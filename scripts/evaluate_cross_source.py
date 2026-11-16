#!/usr/bin/env python3
"""Evaluate generalization across email sources (Leave-One-Source-Out test).

For learned models (TF-IDF + LR, Encoders):
- Retrains on train split excluding source S
- Tunes threshold on val split excluding source S
- Evaluates detection performance on test emails from source S

For zero-shot models (Rules):
- Evaluates directly on test emails from source S

Usage:
    python scripts/evaluate_cross_source.py --model configs/models/tfidf_lr.toml \\
        --splits data/splits/champa2023-v1
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from phishbench.data.splits import load_split
from phishbench.detectors import load_detector
from phishbench.eval.harness import evaluate
from phishbench.types import Email


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="model config (TOML)")
    parser.add_argument("--splits", required=True, type=Path, help="frozen split directory")
    parser.add_argument("--out", type=Path, default=Path("results/cross_source"), help="output directory")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads(args.model.read_text(encoding="utf-8"))
    det_cfg = config["detector"]
    is_learned = any("train" in s for s in det_cfg.get("data_allowed", []))

    print(f"Loading splits from {args.splits} ...")
    train = load_split(args.splits, "train")
    val = load_split(args.splits, "val")
    test = load_split(args.splits, "test")

    sources = sorted({e.source for e in test if e.source})
    print(f"Detected {len(sources)} sources: {', '.join(sources)}")

    results = {}
    for src in sources:
        print(f"\n--- Evaluating held-out source: {src} ---")
        test_src = [e for e in test if e.source == src]
        val_src = [e for e in val if e.source != src] if is_learned else val
        train_src = [e for e in train if e.source != src] if is_learned else []

        phish_count = sum(1 for e in test_src if e.is_phishing)
        legit_count = len(test_src) - phish_count
        print(f"Test samples for {src}: {len(test_src)} ({phish_count} phish, {legit_count} legit)")

        detector = load_detector(det_cfg["class"], det_cfg.get("params"))
        res = evaluate(
            detector,
            test=test_src,
            train=train_src,
            val=val_src,
            warmup=config.get("eval", {}).get("warmup", 0),
            batch_size=config.get("eval", {}).get("batch_size", 100),
            config={"name": f"{detector.name}-leave-out-{src}", "held_out_source": src},
        )
        t_val = res.get("threshold", {}).get("value", 0.5)
        det_block = res.get("detection") or {}
        headline = det_block.get("headline_recall_at_fpr", {})
        recall = headline.get("recall")
        if recall is None and phish_count > 0:
            # For single-class phishing sources, compute recall at chosen threshold
            preds = res.get("predictions", [])
            hit = sum(1 for p in preds if p.get("score") is not None and p["score"] >= t_val)
            recall = hit / phish_count
            print(f"Recall at chosen threshold ({t_val:.4f}) on {src}: {recall:.4f}")
        else:
            print(f"Recall @ 1% FPR on {src}: {recall if recall is not None else 'N/A'}")
        results[src] = res

    summary_file = args.out / f"{Path(args.model).stem}_cross_source.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved cross-source results to {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

