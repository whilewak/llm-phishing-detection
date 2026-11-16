#!/usr/bin/env python3
"""Freeze the few-shot example pool shared by every LLM.

Draws 8 short examples from the train split (never val or test): 4 phishing
(1 Nazario, 1 Nigerian Fraud, 2 from the spam-heavy corpora) and 4 legitimate
(Enron, Ling, SpamAssassin, TREC), labels alternating. The pool is written with
its content hash and the train split's id hash, so loading it detects drift.

    python scripts/build_fewshot.py --splits data/splits/champa2023-v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

from phishbench.data.splits import load_split
from phishbench.prompts import fewshot_hash
from phishbench.types import PHISHING, Email

MAX_BODY_CHARS = 2_000
# (label, allowed sources) in the order examples are shown.
SLOTS = [
    (PHISHING, ("nazario",)),
    ("legitimate", ("enron",)),
    (PHISHING, ("nigerian",)),
    ("legitimate", ("ling",)),
    (PHISHING, ("ceas08", "trec05", "trec06", "trec07", "spamassassin")),
    ("legitimate", ("spamassassin",)),
    (PHISHING, ("ceas08", "trec05", "trec06", "trec07", "spamassassin")),
    ("legitimate", ("trec05", "trec06", "trec07", "ceas08")),
]


def choose(train: Sequence[Email], seed: int = 0) -> list[Email]:
    rng = random.Random(seed)
    pool = sorted((e for e in train if 200 <= len(e.body) <= MAX_BODY_CHARS and e.subject), key=lambda e: e.id)
    chosen: list[Email] = []
    for label, sources in SLOTS:
        candidates = [e for e in pool if e.label == label and e.source in sources and e not in chosen]
        if not candidates:
            raise ValueError(f"no train email for {label} from {sources}")
        chosen.append(rng.choice(candidates))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path, default=Path("data/splits/champa2023-v1"))
    parser.add_argument("--out", type=Path, help="default: configs/prompts/fewshot-<version>.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    out = args.out or Path("configs/prompts") / f"fewshot-{manifest['version']}.json"
    if out.exists():
        print(f"{out} already exists; few-shot pools are frozen. Delete it deliberately to rebuild.", file=sys.stderr)
        return 1

    examples = choose(load_split(args.splits, "train"), args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split_version": manifest["version"],
                "train_ids_sha256": manifest["splits"]["train"]["ids_sha256"],
                "seed": args.seed,
                "examples_sha256": fewshot_hash(examples),
                "examples": [
                    {"id": e.id, "label": e.label, "source": e.source, "from": e.sender,
                     "subject": e.subject, "body": e.body, "urls": list(e.urls)}
                    for e in examples
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: " + ", ".join(f"{e.source}/{e.label}" for e in examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
