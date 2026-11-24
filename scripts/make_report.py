#!/usr/bin/env python3
"""Phase 6 tables and figures, from saved scores and the cascade's results.

Reads results/scores/*__<split>.json (scripts/dump_scores.py), the newest
results/cascade_*.json (scripts/evaluate.py on configs/models/cascade.toml)
and results/cascade_tuning.json. Writes to results/report/:

    summary.md             one row per model, same metric and emails
    pareto_cost.png        recall at 1% FPR vs. dollars per million emails
    pareto_latency.png     recall at 1% FPR vs. median latency
    cascade_curve.png      recall vs. share of emails sent to the top tier

    python scripts/make_report.py            # needs matplotlib
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from phishbench.data.splits import load_split
from phishbench.eval import metrics as m

ROOT = Path(__file__).resolve().parents[1]
# Scatter plots compare every pair of colours, so three groups only; each point is labelled.
GROUPS = {"local": ("Local models", "#2a78d6"), "llm": ("Frontier LLMs (API)", "#eb6834"),
          "cascade": ("Cascade", "#1baf7a")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"rules": "Rules", "tfidf-lr": "TF-IDF + LR", "laya-english": "Laya", "modernbert-base": "ModernBERT",
          "deberta-v3-base": "DeBERTa-v3", "claude-haiku-5-5-zeroshot": "Haiku 5.5 zero-shot",
          "claude-haiku-5-5-fewshot": "Haiku 5.5 few-shot", "claude-sonnet-5-5-zeroshot": "Sonnet 5.5 zero-shot",
          "claude-sonnet-5-5-fewshot": "Sonnet 5.5 few-shot", "claude-opus-5-5-zeroshot": "Opus 5.5 zero-shot",
          "qwen3.5-4b-0shot-shared-v1": "Qwen3.5-4B zero-shot", "qwen3.5-4b-8shot-shared-v1": "Qwen3.5-4B 8-shot",
          "cascade": "Cascade"}


def group_of(model: str) -> str:
    return "cascade" if model == "cascade" else "llm" if model.startswith("claude") else "local"


def _scores(preds: list[dict], labels: dict[str, bool]) -> tuple[list[bool], list[float]]:
    from phishbench.types import Prediction
    y = [labels[p["id"]] for p in preds]
    return y, m.effective_scores(y, [Prediction(score=p["score"], error=p["error"]) for p in preds])


def model_row(model: str, preds: list[dict], labels: dict[str, bool], n_bootstrap: int,
              val: tuple[list[dict], dict[str, bool]] | None = None) -> dict:
    y, scores = _scores(preds, labels)
    fixed = None
    if val:  # the deployable number: threshold fixed on validation, then applied to these emails
        t = m.recall_at_fpr(*_scores(*val), 0.01)[1]
        at = m.at_threshold(y, scores, t)
        fixed = (at.recall, at.fpr)
    recall, _ = m.recall_at_fpr(y, scores, 0.01)
    lo, hi = m.bootstrap_ci(lambda ys, ss: m.recall_at_fpr(ys, ss, 0.01)[0], y, scores, n_bootstrap)
    lat = [p["latency_ms"] for p in preds]
    return {"model": model, "fixed": fixed, "recall": recall, "ci": (lo, hi), "pr_auc": m.average_precision(y, scores),
            "errors": sum(1 for p in preds if p["error"] or p["score"] is None),
            "p50_ms": m.percentile(lat, 50), "p95_ms": m.percentile(lat, 95), "mean_ms": statistics.fmean(lat),
            "usd_per_million": sum(p["cost_usd"] for p in preds) / len(preds) * 1e6}


def scatter(rows: list[dict], x_key: str, x_label: str, floor: float, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, (name, color) in GROUPS.items():
        pts = [r for r in rows if group_of(r["model"]) == key]
        if not pts:
            continue
        xs = [max(r[x_key], floor) for r in pts]
        ax.errorbar(xs, [r["recall"] for r in pts],
                    yerr=[[r["recall"] - r["ci"][0] for r in pts], [r["ci"][1] - r["recall"] for r in pts]],
                    fmt="none", ecolor=color, elinewidth=1, alpha=0.45, zorder=2)
        ax.scatter(xs, [r["recall"] for r in pts], s=70 if key != "cascade" else 130, color=color,
                   marker="D" if key == "cascade" else "o", edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
    # Direct labels; points that sit close together get their labels stacked upwards with a leader line.
    import math
    placed: list[tuple[float, float, int]] = []
    for r in sorted(rows, key=lambda r: (max(r[x_key], floor), -r["recall"])):
        x, yv = max(r[x_key], floor), r["recall"]
        near = [k for px, py, k in placed if abs(math.log10(px) - math.log10(x)) < 0.6 and abs(py - yv) < 0.05]
        k = max(near) + 1 if near else 0
        placed.append((x, yv, k))
        ax.annotate(LABELS.get(r["model"], r["model"]), (x, yv), xytext=(9, 5 + 14 * k),
                    textcoords="offset points", fontsize=8.5, color=INK, zorder=4,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if k else None)
    ax.set_ylim(-0.03, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xscale("log")
    ax.set_xlabel(x_label, color=MUTED)
    ax.set_ylabel("Recall at 1% false-positive rate", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def curve(tuning: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted((v["test"]["top_tier_share"], v["test"]["recall"]) for v in tuning["curve_by_top_tier_cap"].values())
    ref = tuning["reference_single_model"]
    ref_recall = ref["test_at_val_threshold"]["recall"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(ref_recall, color=GROUPS["llm"][1], linewidth=1.5, zorder=2)
    ax.annotate(f"{LABELS.get(ref['model'], ref['model'])} alone ({ref_recall:.1%})", (1.0, ref_recall), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.axhline(0.95 * ref_recall, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("target: 95% of that recall", (1.0, 0.95 * ref_recall), xycoords=("axes fraction", "data"),
                xytext=(0, -11), textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    ax.axvline(0.10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("target: ≤10% to top tier", (0.10, 0), xytext=(5, 4), textcoords="offset points",
                fontsize=8.5, color=MUTED, xycoords=("data", "axes fraction"))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GROUPS["cascade"][1], linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    chosen = tuning["chosen"]["test"]
    ax.annotate(f"chosen: {chosen['recall']:.1%} with {chosen['top_tier_share']:.1%} escalated",
                (chosen["top_tier_share"], chosen["recall"]), xytext=(14, -62), textcoords="offset points",
                fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    right = max(0.3, max(p[0] for p in pts) + 0.05)
    ax.set_xlim(-0.01, right)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of emails that reach the top tier (test_llm)", color=MUTED)
    ax.set_ylabel("Recall, thresholds fixed on validation", color=MUTED)
    ax.set_title("Cascade: recall kept as fewer emails are escalated", loc="left", color=INK, fontsize=12, pad=12)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=ROOT / "results" / "scores")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "champa2023-v1")
    parser.add_argument("--split", default="test_llm")
    parser.add_argument("--val-splits", type=Path, default=ROOT / "data" / "splits" / "champa2023-v1")
    parser.add_argument("--val-split", default="val_llm", help="where each model's threshold was fixed")
    parser.add_argument("--cascade-glob", default="cascade_2*.json", help="harness results for the cascade, in results/")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "report")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    labels = {e.id: e.is_phishing for e in load_split(args.splits, args.split)}
    val_labels = {e.id: e.is_phishing for e in load_split(args.val_splits, args.val_split)}
    rows = []
    for path in sorted(args.scores.glob(f"*__{args.split}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        val_path = args.scores / f"{data['model']}__{args.val_split}.json"
        val = (json.loads(val_path.read_text(encoding="utf-8"))["predictions"], val_labels) if val_path.exists() else None
        rows.append(model_row(data["model"], data["predictions"], labels, args.bootstrap, val))
        if val is None and data.get("validation_threshold") is not None:
            # Imported from a results file: use the validation threshold recorded by that run.
            at = m.at_threshold(*_scores(data["predictions"], labels), data["validation_threshold"])
            rows[-1]["fixed"] = (at.recall, at.fpr)
    cascades = sorted((ROOT / "results").glob(args.cascade_glob))
    exits = None
    for path in reversed(cascades):  # newest cascade run on exactly these emails
        res = json.loads(path.read_text(encoding="utf-8"))
        if set(p["id"] for p in res["predictions"]) != set(labels):
            continue
        rows.append(model_row("cascade", res["predictions"], labels, args.bootstrap))
        at = res["detection"]["at_chosen_threshold"]
        rows[-1]["fixed"] = (at["recall"], at["false_positive_rate"])
        counts: dict[int, int] = {}
        for p in res["predictions"]:
            counts[p["details"]["exit_tier"]] = counts.get(p["details"]["exit_tier"], 0) + 1
        exits = {t: c / len(res["predictions"]) for t, c in sorted(counts.items())}
        break
    rows.sort(key=lambda r: -r["recall"])

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test_llm" else f"_{args.split}"
    lines = [f"# Results on `{args.splits.name}/{args.split}` ({len(labels)} emails, {sum(labels.values())} phishing)", "",
             "| Model | Recall at 1% FPR (95% CI) | Recall / FPR at validation threshold | PR-AUC | Errors "
             "| p50 / mean latency | $ per 1M emails |",
             "| --- | --- | --- | --- | ---: | --- | ---: |"]
    for r in rows:
        lines.append(f"| {LABELS.get(r['model'], r['model'])} | {r['recall']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f}) | "
                     + (f"{r['fixed'][0]:.3f} / {r['fixed'][1]:.2%}" if r["fixed"] else "n/a")
                     + f" | {r['pr_auc']:.3f} | {r['errors']} | {r['p50_ms']:,.1f} / {r['mean_ms']:,.0f} ms | "
                     f"${r['usd_per_million']:,.0f} |")
    if exits:
        lines += ["", "Cascade: share of emails answered by each tier: "
                  + ", ".join(f"tier {t} {s:.1%}" for t, s in exits.items()) + "."]
    lines += ["", "\"Recall at 1% FPR\" picks the best threshold on these emails (how every model is compared); "
              "\"at validation threshold\" applies the threshold fixed on the validation split, which is what a "
              "deployment would get. API and cascade latencies are the recorded per-email times."]
    (args.out / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not args.no_plots:
        scatter(rows, "usd_per_million", "Dollars per million emails (log scale; local models shown at $1)", 1.0,
                "Accuracy vs. cost", args.out / f"pareto_cost{suffix}.png")
        scatter(rows, "mean_ms", "Mean latency per email, ms (log scale)", 0.1,
                "Accuracy vs. latency", args.out / f"pareto_latency{suffix}.png")
        tuning = ROOT / "results" / "cascade_tuning.json"
        if tuning.exists() and args.split == "test_llm":
            curve(json.loads(tuning.read_text(encoding="utf-8")), args.out / "cascade_curve.png")
        print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
