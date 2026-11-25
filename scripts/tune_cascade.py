#!/usr/bin/env python3
"""Pick the cascade's tier models and hand-off thresholds on val_llm.

Reads the per-email scores saved by scripts/dump_scores.py. Uses val_llm for
every choice; test_llm scores are only read at the end, to report how the
already-fixed thresholds do.

1. Per tier, the candidate with the best validation recall at 1% FPR wins
   (the cheapest one, if several are within one point of the best).
2. Hand-off thresholds are searched on validation: most recall with the
   cascade's FPR <= 1% and <= 10% of emails reaching the top tier.
3. Writes configs/models/cascade.toml (replays the saved scores through the
   normal harness) and results/cascade_tuning.json (the choice, ablations and
   the recall-vs-escalation curve).

    python scripts/tune_cascade.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from phishbench.cascade_tuning import apply, as_array, last_tier_scores, tune
from phishbench.eval.metrics import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
# Tier 3 (Qwen ~4B, Phase 4) joins automatically once its scores are in results/scores/.
TIERS = {
    "1 · cheap baseline": ["rules", "tfidf-lr"],
    "2 · encoder / System 1": ["laya-english", "modernbert-base", "deberta-v3-base"],
    "3 · small language model": ["qwen3.5-4b-0shot-shared-v1", "qwen3.5-4b-8shot-shared-v1"],
    "4 · frontier LLM": ["claude-haiku-5-5-zeroshot", "claude-haiku-5-5-fewshot",
                         "claude-sonnet-5-5-zeroshot", "claude-sonnet-5-5-fewshot"],
}
SHARE_CAPS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
TARGET_SHARE = 0.10
MAX_FPR = 0.01
NEAR_TIE = 0.01


def load(scores_dir: Path, model: str, split: str) -> dict | None:
    path = scores_dir / f"{model}__{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def aligned(data: dict, ids: list[str]) -> dict[str, np.ndarray]:
    by_id = {p["id"]: p for p in data["predictions"]}
    return {"score": as_array([by_id[i]["score"] if not by_id[i]["error"] else None for i in ids]),
            "latency": np.array([by_id[i]["latency_ms"] for i in ids]),
            "cost": np.array([by_id[i]["cost_usd"] for i in ids])}


def costs(models: list[dict], thresholds: list[tuple[float, float]]) -> dict:
    """Mean latency and cost per email for fixed thresholds (tiers run in sequence)."""
    n = len(models[0]["score"])
    pending = np.ones(n, dtype=bool)
    latency, cost = np.zeros(n), np.zeros(n)
    for m, (lo, hi) in zip(models, [*thresholds, (math.inf, -math.inf)]):
        latency += np.where(pending, m["latency"], 0.0)
        cost += np.where(pending, m["cost"], 0.0)
        pending &= ~((m["score"] >= hi) | (m["score"] <= lo))
    return {"latency_ms_mean": float(latency.mean()), "latency_ms_p50": float(np.median(latency)),
            "latency_ms_p95": float(np.percentile(latency, 95)), "usd_per_million_emails": float(cost.mean() * 1e6)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=ROOT / "results" / "scores")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "champa2023-v1")
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--grid", type=int, default=30, help="threshold candidates per tier")
    parser.add_argument("--config-out", type=Path, default=ROOT / "configs" / "models" / "cascade.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cascade_tuning.json")
    args = parser.parse_args(argv)

    from phishbench.data.splits import load_split
    labels = {name: {e.id: e.is_phishing for e in load_split(args.splits, name, args.processed)}
              for name in ("val_llm", "test_llm")}
    ids = {name: sorted(lab) for name, lab in labels.items()}
    y = {name: np.array([labels[name][i] for i in ids[name]]) for name in ids}

    # 1. Tier models, by validation recall at 1% FPR.
    chosen, selection = [], {}
    for tier, candidates in TIERS.items():
        table, cost = {}, {}
        for model in candidates:
            val = load(args.scores, model, "val_llm")
            if val is None or load(args.scores, model, "test_llm") is None:
                continue
            a = aligned(val, ids["val_llm"])
            eff = last_tier_scores(a["score"], y["val_llm"])
            table[model] = recall_at_fpr(list(y["val_llm"]), list(eff), MAX_FPR)[0]
            cost[model] = (float(a["cost"].mean()), float(a["latency"].mean()))
        # Within NEAR_TIE of the best, validation cannot tell models apart (a point is
        # ~11 phishing emails), so the cheapest of those wins, then the fastest.
        near = [m for m in table if table[m] >= max(table.values()) - NEAR_TIE]
        winner = min(near, key=lambda m: cost[m]) if table else None
        selection[tier] = {"val_recall_at_1pct_fpr": table, "chosen": winner,
                           "rule": f"cheapest within {NEAR_TIE:.0%} of the best validation recall"}
        if table:
            chosen.append(winner)
        print(f"tier {tier}: " + (", ".join(f"{m} {r:.3f}" for m, r in table.items()) or "no scores, tier skipped")
              + (f"  -> {winner}" if table else ""))

    data = {split: [aligned(load(args.scores, m, split), ids[split]) for m in chosen] for split in ids}

    def run(tier_idx: list[int], caps=SHARE_CAPS) -> dict:
        """Tune a cascade over a subset of the chosen tiers on val; apply to test."""
        val = [data["val_llm"][i] for i in tier_idx]
        test = [data["test_llm"][i] for i in tier_idx]
        best = tune([m["score"] for m in val[:-1]], val[-1]["score"], y["val_llm"],
                    max_fpr=MAX_FPR, share_caps=caps, grid=args.grid)
        out = {}
        for cap, o in sorted(best.items()):
            t = apply([m["score"] for m in test[:-1]], test[-1]["score"], y["test_llm"], o.thresholds, o.last_threshold)
            out[cap] = {"val": {**o.as_dict(), **costs(val, o.thresholds)},
                        "test": {**t.as_dict(), **costs(test, o.thresholds)}}
        return out

    full = list(range(len(chosen)))
    curve = run(full)
    main_point = curve[TARGET_SHARE]

    # Best single model: chosen on validation, as the plan's reference for the 95% target.
    best_single_val = {m: recall_at_fpr(list(y["val_llm"]), list(last_tier_scores(d["score"], y["val_llm"])), MAX_FPR)[0]
                       for m, d in zip(chosen, data["val_llm"])}
    ref = max(best_single_val, key=best_single_val.get)
    ref_i = chosen.index(ref)
    ref_val_recall, ref_val_t = recall_at_fpr(
        list(y["val_llm"]), list(last_tier_scores(data["val_llm"][ref_i]["score"], y["val_llm"])), MAX_FPR)
    ref_test = apply([], data["test_llm"][ref_i]["score"], y["test_llm"], [], ref_val_t)

    # Ablations at the target cap: first tier straight to the top tier, and the cascade with each
    # single tier removed. Same grid as the main search, so they are directly comparable.
    ablations = {}
    if len(chosen) > 2:
        subsets = [[0, full[-1]]] + [[i for i in full if i != j] for j in full[:-1]]
        for idx in subsets:
            name = " -> ".join(chosen[i] for i in idx)
            if name not in ablations:
                ablations[name] = run(idx, caps=(TARGET_SHARE,)).get(TARGET_SHARE)

    # Transfer check: the same fixed thresholds on the LLM-written sets (never used for any choice).
    transfer: dict[str, dict] = {}
    for set_name, split in (("synthetic-v1", "synthetic"),):
        syn_dir = ROOT / "data" / "splits" / set_name
        if not (syn_dir / "manifest.json").exists() or not all(load(args.scores, m, split) for m in chosen):
            continue
        syn_lab = {e.id: e.is_phishing for e in load_split(syn_dir, split)}
        syn_ids = sorted(syn_lab)
        ys = np.array([syn_lab[i] for i in syn_ids])
        syn = [aligned(load(args.scores, m, split), syn_ids) for m in chosen]

        def on_syn(idx: list[int], point: dict) -> dict:
            th = [(-math.inf if t["low"] is None else t["low"], math.inf if t["high"] is None else t["high"])
                  for t in point["val"]["thresholds"]]
            last_t = math.inf if point["val"]["last_threshold"] is None else point["val"]["last_threshold"]
            ms = [syn[i] for i in idx]
            o = apply([m["score"] for m in ms[:-1]], ms[-1]["score"], ys, th, last_t)
            return {**o.as_dict(), **costs(ms, th)}

        rows = {" -> ".join(chosen): on_syn(full, main_point)}
        for name, point in ablations.items():
            if point:
                rows[name] = on_syn([chosen.index(m) for m in name.split(" -> ")], point)
        rows[ref + " alone"] = apply([], syn[ref_i]["score"], ys, [], ref_val_t).as_dict()
        transfer[set_name] = rows

    result = {
        "transfer_to_synthetic": transfer,
        "rule": f"all choices on val_llm; cascade FPR <= {MAX_FPR:.0%}; <= {TARGET_SHARE:.0%} of emails to the top tier",
        "tier_selection": selection,
        "tiers": chosen,
        "chosen": main_point,
        "reference_single_model": {"model": ref, "val_recall_at_1pct_fpr": ref_val_recall,
                                   "val_threshold": ref_val_t, "test_at_val_threshold": ref_test.as_dict()},
        "recall_kept_vs_reference": {
            "val": main_point["val"]["recall"] / ref_val_recall,
            "test": main_point["test"]["recall"] / ref_test.recall},
        "curve_by_top_tier_cap": {str(c): v for c, v in curve.items()},
        "ablations_at_target": ablations,
        # Every saved split for the chosen models, so the same config also replays e.g. the synthetic set.
        "score_files": {m: sorted(f"results/scores/{p.name}" for p in args.scores.glob(f"{m}__*.json")) for m in chosen},
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    th = main_point["val"]["thresholds"]
    lines = ["# Cascade tuned on val_llm by scripts/tune_cascade.py (do not edit by hand).",
             "# Tiers replay the per-email scores, latencies and costs saved by scripts/dump_scores.py,",
             "# so the cascade goes through the normal harness without re-running any model.",
             "", "[detector]", 'class = "phishbench.detectors.cascade:CascadeDetector"',
             'data_allowed = ["val_llm: tier choice, hand-off thresholds, final threshold"]',
             "", "[detector.params]", 'name = "cascade"', ""]
    for i, model in enumerate(chosen):
        lines += ["[[detector.params.tiers]]", 'class = "phishbench.detectors.replay:ReplayDetector"']
        if i < len(th):
            # -inf / +inf (tier never clears / never flags) are written as values no score can reach.
            lo = -1.0 if th[i]["low"] is None else th[i]["low"]
            hi = 2.0 if th[i]["high"] is None else th[i]["high"]
            lines += [f"low = {lo!r}", f"high = {hi!r}"]
        files = ", ".join(f'"{f}"' for f in result["score_files"][model])
        lines += [f'params = {{ name = "{model}", score_files = [{files}] }}', ""]
    lines += ["[eval]", "warmup = 0", "batch_size = 2500", ""]
    args.config_out.write_text("\n".join(lines), encoding="utf-8")

    for split in ("val", "test"):
        p = main_point[split]
        print(f"{split}: recall {p['recall']:.3f} at FPR {p['false_positive_rate']:.4f}; exits per tier "
              + " / ".join(f"{s:.1%}" for s in p["exit_share"])
              + f"; ${p['usd_per_million_emails']:.0f} per 1M emails; mean latency {p['latency_ms_mean']:.0f} ms")
    print(f"reference {ref}: val {ref_val_recall:.3f}, test at val threshold {ref_test.recall:.3f} "
          f"(FPR {ref_test.fpr:.4f}); recall kept on test: {result['recall_kept_vs_reference']['test']:.1%}")
    for set_name, rows in transfer.items():
        for name, t in rows.items():
            print(f"{set_name}, {name}: recall {t['recall']:.3f} at FPR {t['false_positive_rate']:.3f}; "
                  f"top tier {t['top_tier_share']:.1%}")
    print(f"wrote {args.config_out} and {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
