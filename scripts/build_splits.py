#!/usr/bin/env python3
"""Clean, deduplicate and split the curated datasets, then freeze the splits.

    python scripts/download_data.py
    python scripts/build_splits.py

Writes data/processed/champa2023.jsonl (git-ignored) and
data/splits/champa2023-v1/ (committed: id lists + manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from phishbench.data.clean import SOURCES, TRUE_PHISHING_SOURCES, DropCounts, read_source
from phishbench.data.dedup import deduplicate
from phishbench.data.loader import dataset_fingerprint
from phishbench.data.splits import LLM_SUBSAMPLES, freeze, stratified_sample, stratified_split

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "champa2023")
    parser.add_argument("--version", default="champa2023-v1")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--llm-size", type=int, default=2500)
    parser.add_argument("--verify", action="store_true",
                        help="rebuild the processed emails and check them against the committed splits "
                             "instead of freezing new ones")
    args = parser.parse_args(argv)
    split_dir = ROOT / "data" / "splits" / args.version
    if not args.verify and (split_dir / "manifest.json").exists():
        parser.error(f"{split_dir} is already frozen; use --verify to rebuild against it, or a new --version")

    t0 = time.time()
    drops = DropCounts()
    emails, truncated = [], 0
    raw_counts = {}
    for filename in SOURCES:
        before = len(emails)
        for email, was_truncated in read_source(args.raw / filename, drops):
            emails.append(email)
            truncated += was_truncated
        raw_counts[SOURCES[filename]] = len(emails) - before
        print(f"read {filename}: {raw_counts[SOURCES[filename]]} emails")

    print(f"deduplicating {len(emails)} emails ...")
    kept, report = deduplicate(emails)
    print(f"kept {report.kept} ({report.exact_duplicates} exact, {report.near_duplicates} near duplicates, "
          f"{report.mixed_label_emails} in mixed-label groups dropped) in {time.time() - t0:.0f}s")

    processed_rel = Path("data") / "processed" / f"{args.version}.jsonl"
    processed = ROOT / processed_rel
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps({"id": e.id, "label": e.label, "source": e.source, "from": e.sender,
                                "subject": e.subject, "body": e.body, "urls": list(e.urls)}) + "\n")

    splits = stratified_split(kept, args.seed)
    for name, parent in LLM_SUBSAMPLES.items():
        splits[name] = stratified_sample(splits[parent], args.llm_size, args.seed)

    if args.verify:
        frozen = json.loads((split_dir / "manifest.json").read_text())["splits"]
        bad = [n for n, emails in splits.items()
               if dataset_fingerprint(emails) != frozen[n]["content_fingerprint"]]
        if bad:
            print(f"MISMATCH: rebuilt splits differ from {split_dir} for {bad}", file=sys.stderr)
            return 1
        print(f"verified: rebuilt emails match every split in {split_dir}")
        return 0

    source_info = json.loads((args.raw / "SOURCE.json").read_text())
    manifest = freeze(
        splits,
        split_dir,
        {
            "version": args.version,
            "dataset": source_info,
            "processed_file": str(processed_rel),
            "label_mapping": "source label 1 (spam or phishing) -> phishing; 0 -> legitimate",
            "true_phishing_sources": list(TRUE_PHISHING_SOURCES),
            "seed": args.seed,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "llm_subsample_size": args.llm_size,
            "read_per_source": raw_counts,
            "dropped_while_reading": asdict(drops),
            "bodies_truncated": truncated,
            "dedup": asdict(report),
        },
    )
    for name, s in manifest["splits"].items():
        print(f"{name:9} {s['n']:7}  {s['labels']}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
