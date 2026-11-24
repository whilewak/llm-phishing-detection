"""Replays predictions saved by scripts/dump_scores.py.

Lets the cascade run through the normal harness on recorded scores, latencies
and costs, so tuning and evaluating it never re-runs (or re-pays for) a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from phishbench.types import Email, Prediction, TrainingCost


class ReplayDetector:
    def __init__(self, name: str, score_files: Sequence[str | Path]):
        self.name = name
        self._preds: dict[str, dict] = {}
        for path in score_files:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data["model"] != name:
                raise ValueError(f"{path} holds scores for {data['model']!r}, not {name!r}")
            for rec in data["predictions"]:
                self._preds[rec["id"]] = rec

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        return TrainingCost()

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        out = []
        for e in emails:
            rec = self._preds.get(e.id)
            if rec is None:
                raise KeyError(f"no recorded {self.name} prediction for email {e.id}")
            out.append(Prediction(score=rec["score"], latency_ms=rec["latency_ms"],
                                  cost_usd=rec["cost_usd"], error=rec["error"]))
        return out
