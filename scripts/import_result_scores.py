#!/usr/bin/env python3
"""Turn a harness results file into a saved-scores file for the cascade and report.

For models run on another machine whose results JSON is committed but whose
response cache is not (Qwen, Phase 4). A results file only holds the test
split's predictions, so this gives `<model>__<split>.json` for that split only;
tuning the cascade with the model still needs its validation scores.

    python scripts/import_result_scores.py results/qwen3.5-4b-0shot-shared-v1_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(paths: list[str]) -> int:
    out_dir = ROOT / "results" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        split = res["config"]["data_ref"]["test"]
        out = {
            "model": res["model"], "config": res["config"].get("path"), "split": split,
            "split_set": res["config"]["data_ref"].get("version"), "fingerprint": res["data"]["test"]["fingerprint"],
            "imported_from": str(Path(path).name), "validation_threshold": res["threshold"]["value"],
            "environment": res["environment"],
            "predictions": [{"id": p["id"], "score": p["score"], "latency_ms": p["latency_ms"],
                             "cost_usd": p["cost_usd"], "error": p["error"]} for p in res["predictions"]],
        }
        dest = out_dir / f"{res['model']}__{split}.json"
        dest.write_text(json.dumps(out, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{res['model']}: {len(out['predictions'])} {split} predictions -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
