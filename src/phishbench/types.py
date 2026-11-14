"""Shared data types and the Detector interface every model implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

PHISHING = "phishing"
LEGITIMATE = "legitimate"
SPAM = "spam"
LABELS = (PHISHING, LEGITIMATE, SPAM)


@dataclass(frozen=True)
class Email:
    """One email in the shared schema. Models only ever see these fields."""

    id: str
    label: str | None
    sender: str
    subject: str
    body: str
    urls: tuple[str, ...] = ()
    source: str | None = None

    @property
    def is_phishing(self) -> bool:
        """Binary headline task: phishing vs. everything else."""
        return self.label == PHISHING


@dataclass
class Prediction:
    """A model's output for one email.

    `score` is in [0, 1], higher meaning more likely phishing. When `error` is
    set (unparseable output, timeout, refusal, malformed response) `score` is
    None and the harness counts the email as wrong.
    """

    score: float | None
    label: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None or self.score is None


@dataclass
class TrainingCost:
    """What it cost to train or tune a model. All zeros for zero-shot models."""

    wall_clock_s: float = 0.0
    peak_memory_mb: float = 0.0
    hardware: str = ""
    cost_usd: float = 0.0
    n_train_emails: int = 0


class Detector(Protocol):
    name: str

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        """Train or tune on the given splits. Zero-shot models return TrainingCost()."""
        ...

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        """Score emails; must return exactly one Prediction per email, in order."""
        ...
