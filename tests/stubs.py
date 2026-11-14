"""Stub detectors and helpers for testing the harness without real models."""

from __future__ import annotations

from typing import Sequence

from phishbench.types import LEGITIMATE, PHISHING, Email, Prediction, TrainingCost


def make_email(i: int, label: str | None) -> Email:
    return Email(id=f"e{i}", label=label, sender=f"s{i}@example.com", subject=f"subject {i}", body=f"body {i}")


def make_split(n_phish: int, n_legit: int, start: int = 0) -> list[Email]:
    emails = [make_email(start + i, PHISHING) for i in range(n_phish)]
    emails += [make_email(start + n_phish + i, LEGITIMATE) for i in range(n_legit)]
    return emails


class OracleDetector:
    """Scores phishing 0.9 and everything else 0.1."""

    name = "oracle"

    def __init__(self):
        self.fit_calls = 0

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        self.fit_calls += 1
        return TrainingCost(wall_clock_s=1.5, n_train_emails=len(train), hardware="stub")

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        return [Prediction(score=0.9 if e.is_phishing else 0.1, latency_ms=2.0, cost_usd=0.001) for e in emails]


class ScriptedDetector:
    """Returns a fixed score (or an error) per email id."""

    name = "scripted"

    def __init__(self, scores: dict[str, float | None]):
        self.scores = scores

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        return TrainingCost()

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        out = []
        for e in emails:
            s = self.scores[e.id]
            out.append(Prediction(score=s, error=None if s is not None else "refused", latency_ms=1.0))
        return out


class ShortDetector:
    """Broken: returns one prediction fewer than asked."""

    name = "short"

    def fit(self, train, val):
        return TrainingCost()

    def predict(self, emails):
        return [Prediction(score=0.5) for _ in emails][:-1]
