"""Cascade: cheap tiers answer the emails they are sure about, the rest escalate.

Every tier but the last has two hand-off thresholds, chosen on validation:

- score >= `high`  -> stop, call it phishing
- score <= `low`   -> stop, call it legitimate
- otherwise, or if the tier failed (error, refusal) -> escalate to the next tier

The last tier's failures stay failures. Latency and cost are summed over the
tiers an email visited (tiers run one after another).

The cascade's own 0..1 score keeps that decision and the deciding tier's
ranking, in three bands: early "legitimate" exits in [0, 1/3], emails the last
tier scored in [1/3, 2/3], early "phishing" exits in [2/3, 1]. Within a band
the order is the deciding tier's score.
"""

from __future__ import annotations

from typing import Any, Sequence

from phishbench.detectors import load_detector
from phishbench.types import Detector, Email, Prediction, TrainingCost

BANDS = {"legitimate": 0, "last": 1, "phishing": 2}


def band_score(band: str, score: float) -> float:
    """Place a tier's 0..1 score inside its third of the cascade's score range."""
    return (BANDS[band] + min(max(score, 0.0), 1.0)) / 3


def decide(score: float | None, low: float, high: float) -> str:
    """What a non-final tier does with one email: 'phishing', 'legitimate' or 'escalate'."""
    if score is None:
        return "escalate"
    if score >= high:
        return "phishing"
    if score <= low:
        return "legitimate"
    return "escalate"


class CascadeDetector:
    def __init__(self, tiers: Sequence[dict[str, Any]], name: str = "cascade"):
        """`tiers`: cheapest first. Each is {"class": "module:Class", "params": {...},
        "low": float, "high": float}; or {"detector": obj, ...} in tests. The last
        tier needs no thresholds."""
        if len(tiers) < 2:
            raise ValueError("a cascade needs at least two tiers")
        self.name = name
        self.detectors: list[Detector] = []
        self.thresholds: list[tuple[float, float]] = []
        for i, tier in enumerate(tiers):
            det = tier.get("detector") or load_detector(tier["class"], tier.get("params"))
            self.detectors.append(det)
            if i < len(tiers) - 1:
                low, high = float(tier["low"]), float(tier["high"])
                if low >= high:
                    raise ValueError(f"tier {i + 1} ({det.name}): low {low} must be below high {high}")
                self.thresholds.append((low, high))

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        total = TrainingCost()
        for det in self.detectors:
            cost = det.fit(train, val)
            total.wall_clock_s += cost.wall_clock_s
            total.peak_memory_mb = max(total.peak_memory_mb, cost.peak_memory_mb)
            total.cost_usd += cost.cost_usd
            total.n_train_emails = max(total.n_train_emails, cost.n_train_emails)
            total.hardware = total.hardware or cost.hardware
        return total

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        out: list[Prediction | None] = [None] * len(emails)
        latency = [0.0] * len(emails)
        cost = [0.0] * len(emails)
        pending = list(range(len(emails)))
        last = len(self.detectors) - 1
        for t, det in enumerate(self.detectors):
            if not pending:
                break
            preds = det.predict([emails[i] for i in pending])
            if len(preds) != len(pending):
                raise RuntimeError(f"{det.name} returned {len(preds)} predictions for {len(pending)} emails")
            still: list[int] = []
            for i, p in zip(pending, preds):
                latency[i] += p.latency_ms
                cost[i] += p.cost_usd
                details = {"exit_tier": t + 1, "exit_model": det.name, "tier_score": p.score}
                if t == last:
                    score = None if p.failed else band_score("last", p.score)
                    out[i] = Prediction(score=score, label=p.label, latency_ms=latency[i], cost_usd=cost[i],
                                        error=p.error, details=details)
                    continue
                action = decide(None if p.failed else p.score, *self.thresholds[t])
                if action == "escalate":
                    still.append(i)
                else:
                    out[i] = Prediction(score=band_score(action, p.score), label=action, latency_ms=latency[i], cost_usd=cost[i],
                                        details=details)
            pending = still
        return out  # type: ignore[return-value]
