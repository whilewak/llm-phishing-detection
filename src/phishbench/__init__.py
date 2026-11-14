"""Phishing detection benchmark: one interface, one harness, many models."""

from phishbench.types import (
    LEGITIMATE,
    PHISHING,
    SPAM,
    Detector,
    Email,
    Prediction,
    TrainingCost,
)

__all__ = [
    "Detector",
    "Email",
    "LEGITIMATE",
    "PHISHING",
    "Prediction",
    "SPAM",
    "TrainingCost",
]
