#!/usr/bin/env python3
"""Run several API model configs under one overall dollar budget.

Each run gets `--max-cost` = what is left of the budget, so the suite as a
whole can never spend more than `--budget` on new API calls (cached responses
are free). Stops at the first run that fails or hits its cap.

    python scripts/run_suite.py --budget 55 \\
        configs/models/claude-haiku-5-5-zeroshot.toml configs/models/claude-sonnet-5-5-zeroshot.toml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--budget", type=float, required=True, help="total new API spend allowed, USD")
    parser.add_argument("--already-spent", type=float, default=0.0, help="spend to count against the budget")
    parser.add_argument("--splits", default="data/splits/champa2023-v1")
    parser.add_argument("--val", default="val_llm")
    parser.add_argument("--test", default="test_llm")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    spent = args.already_spent
    for cfg in args.configs:
        remaining = args.budget - spent
        if remaining <= 0.01:
            print(f"budget used up (${spent:.2f} of ${args.budget:.2f}); stopping before {cfg.name}")
            return 1
        print(f"\n=== {cfg.stem}: ${spent:.2f} spent so far, cap for this run ${remaining:.2f}", flush=True)
        before = set(args.out.glob(f"{cfg.stem}_*.json"))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--model", str(cfg),
             "--splits", args.splits, "--train", "train", "--val", args.val, "--test", args.test,
             "--out", str(args.out), "--max-cost", f"{remaining:.4f}"],
            cwd=ROOT,
        )
        new = sorted(set(args.out.glob(f"{cfg.stem}_*.json")) - before)
        if proc.returncode != 0 or not new:
            print(f"{cfg.stem} did not finish (exit {proc.returncode}). Completed calls are cached; "
                  "rerunning the suite resumes for free.")
            return proc.returncode or 1
        spent += json.loads(new[-1].read_text())["api_spend"]["new_usd"]
    print(f"\nall {len(args.configs)} runs finished; new API spend ${spent:.2f} of ${args.budget:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
