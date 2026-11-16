#!/usr/bin/env python3
"""Evaluate one detector and write a standard results file.

With a frozen split set, --test/--val/--train name splits:

    python scripts/evaluate.py --model configs/models/laya.toml \\
        --splits data/splits/champa2023-v1 --test test_llm --val val_llm

Without --splits they are file paths (JSON or JSONL):

    python scripts/evaluate.py --model configs/models/laya.toml --test data/benchmark_emails.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from phishbench.data import load_emails
from phishbench.data.splits import load_split
from phishbench.detectors import load_detector
from phishbench.eval import evaluate, write_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="model config (TOML)")
    parser.add_argument("--splits", type=Path, help="frozen split directory (data/splits/<version>)")
    parser.add_argument("--test", required=True, help="test split name, or a JSON/JSONL file")
    parser.add_argument("--val", help="validation split or file, used to choose the threshold")
    parser.add_argument("--train", help="training split or file, for models that learn")
    parser.add_argument("--out", type=Path, default=Path("results"), help="results directory")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--max-cost", type=float,
                        help="API models: lower the config's max_total_cost_usd to this many dollars of new spend")
    args = parser.parse_args(argv)

    config = tomllib.loads(args.model.read_text(encoding="utf-8"))
    det_cfg = config["detector"]
    eval_cfg = config.get("eval", {})

    if args.splits:
        manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
        load = lambda name: load_split(args.splits, name)  # noqa: E731
        source_groups = {"true_phishing": manifest.get("true_phishing_sources", [])}
        data_ref = {"splits": str(args.splits), "version": manifest.get("version")}
    else:
        load = load_emails
        source_groups = {}
        data_ref = {}

    test = load(args.test)
    val = load(args.val) if args.val else []
    train = load(args.train) if args.train else []

    params = dict(det_cfg.get("params") or {})
    if args.max_cost is not None:
        params["max_total_cost_usd"] = min(params.get("max_total_cost_usd", args.max_cost), args.max_cost)
    detector = load_detector(det_cfg["class"], params)
    result = evaluate(
        detector,
        test,
        train=train,
        val=val,
        warmup=eval_cfg.get("warmup", 3),
        batch_size=eval_cfg.get("batch_size", 1),
        n_bootstrap=args.bootstrap,
        config={"path": str(args.model), **config,
                "data_ref": {**data_ref, "test": args.test, "val": args.val, "train": args.train}},
        source_groups=source_groups,
    )
    if hasattr(detector, "spent_usd"):
        # Money actually spent in this run; cached responses cost nothing.
        result["api_spend"] = {"new_usd": detector.spent_usd, "api_calls": detector.api_calls,
                               "cap_usd": params.get("max_total_cost_usd")}
    path = write_result(result, args.out)

    det = result["detection"]
    print(f"model: {result['model']}  test emails: {result['data']['test']['n']}")
    if det:
        h = det["headline_recall_at_fpr"]
        print(f"recall at {h['max_fpr']:.0%} FPR: {h['recall']:.3f}  (95% CI {h['ci95'][0]:.3f}-{h['ci95'][1]:.3f})")
        print(f"PR-AUC: {det['pr_auc']:.3f}")
    p50 = result["speed"]["latency_ms_p50"]
    print(f"errors: {result['errors']['count']}  p50 latency: {p50 if p50 is None else round(p50, 1)} ms")
    if "api_spend" in result:
        s = result["api_spend"]
        print(f"new API spend: ${s['new_usd']:.4f} over {s['api_calls']} calls (cap ${s['cap_usd']})")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
