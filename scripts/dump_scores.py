#!/usr/bin/env python3
"""Fit one detector, then save its per-email predictions on several splits.

The cascade is tuned and evaluated from these files, so no model has to be
re-run (or re-paid for) while hand-off thresholds are searched:

    python scripts/dump_scores.py --model configs/models/tfidf_lr.toml \\
        --splits data/splits/champa2023-v1 --train train --val val_llm \\
        --score val_llm test_llm

Writes results/scores/<model>__<split>.json. `--val` is what the detector may
tune on in fit() (Laya's signal weights); thresholds are not chosen here.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from phishbench.data.loader import dataset_fingerprint
from phishbench.data.splits import load_split
from phishbench.detectors import load_detector
from phishbench.eval.environment import describe_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--train", help="training split, for models that learn")
    parser.add_argument("--val", help="validation split passed to fit()")
    parser.add_argument("--score", nargs="+", required=True, help="splits to score")
    parser.add_argument("--score-splits", type=Path,
                        help="split set holding the --score splits, if different from --splits")
    parser.add_argument("--out", type=Path, default=Path("results/scores"))
    parser.add_argument("--max-cost", type=float, help="API models: cap on new spend, USD")
    args = parser.parse_args(argv)

    config = tomllib.loads(args.model.read_text(encoding="utf-8"))
    params = dict(config["detector"].get("params") or {})
    if args.max_cost is not None:
        params["max_total_cost_usd"] = min(params.get("max_total_cost_usd", args.max_cost), args.max_cost)
    detector = load_detector(config["detector"]["class"], params)
    batch_size = config.get("eval", {}).get("batch_size", 1)

    train = load_split(args.splits, args.train) if args.train else []
    val = load_split(args.splits, args.val) if args.val else []
    detector.fit(train, val)

    score_dir = args.score_splits or args.splits
    version = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8")).get("version")
    args.out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(args.score):
        emails = load_split(score_dir, name)
        if i == 0:
            detector.predict(emails[:batch_size])  # warmup, so latencies exclude model loading
        preds = []
        for start in range(0, len(emails), batch_size):
            preds.extend(detector.predict(emails[start : start + batch_size]))
        if len(preds) != len(emails):
            raise RuntimeError(f"{detector.name} returned {len(preds)} predictions for {len(emails)} emails")
        out = {
            "model": detector.name,
            "config": str(args.model),
            "split_set": version,
            "split": name,
            "fingerprint": dataset_fingerprint(emails),
            "fit": {"train": args.train, "val": args.val},
            "environment": describe_environment(),
            "predictions": [
                {"id": e.id, "score": p.score, "latency_ms": p.latency_ms, "cost_usd": p.cost_usd, "error": p.error}
                for e, p in zip(emails, preds)
            ],
        }
        path = args.out / f"{detector.name}__{name}.json"
        path.write_text(json.dumps(out, allow_nan=False) + "\n", encoding="utf-8")
        failed = sum(1 for p in preds if p.failed)
        print(f"{detector.name} on {name}: {len(preds)} emails, {failed} errors -> {path}", flush=True)
    if hasattr(detector, "spent_usd"):
        print(f"new API spend: ${detector.spent_usd:.4f} over {detector.api_calls} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
