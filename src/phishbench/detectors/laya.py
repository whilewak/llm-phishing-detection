"""Laya System 1 decision model (ModernBERT-large) as a Detector.

Laya answers several yes/no questions about an email plus a three-way
classification in one forward pass. Until Phase 3 fits the signal weights on
validation data, the score is the POC's hand-picked weighted sum of the yes/no
probabilities, scaled to [0, 1].
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from phishbench.types import LEGITIMATE, PHISHING, Email, Prediction, TrainingCost

QUESTIONS: dict[str, dict[str, Any]] = {
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing, scam, or social engineering attempt to steal money, passwords, or data?",
        "criteria": {
            "true": "phishing, scam, or social engineering fraud",
            "false": "a legitimate business or personal email",
        },
    },
    "urgency_pressure": {
        "type": "noul",
        "instructions": "Does this email create artificial time pressure, panic, or threats of account suspension?",
        "criteria": {
            "true": "severe urgency or account termination threat",
            "false": "routine timing or no deadline pressure",
        },
    },
    "credential_harvest": {
        "type": "noul",
        "instructions": "Does this email ask the recipient to verify login credentials, enter passwords, or access an auth portal?",
        "criteria": {
            "true": "requests login credentials, passwords, or MFA tokens",
            "false": "no login or credential request",
        },
    },
    "financial_pressure": {
        "type": "noul",
        "instructions": "Does this email ask to buy gift cards, execute wire transfers, or divert invoice payments?",
        "criteria": {
            "true": "requests direct money transfer or gift cards",
            "false": "no financial payment diversion or gift card request",
        },
    },
    "suspicious_attachment": {
        "type": "noul",
        "instructions": "Does this email urge opening an attachment, enabling macros, or reviewing an attached invoice?",
        "criteria": {
            "true": "prompts to open attachment or enable content",
            "false": "no attachment or safe standard file",
        },
    },
    "classification": {
        "type": "choice",
        "instructions": "What is the security classification of the email?",
        "criteria": {
            "benign": "legitimate, safe business or personal email",
            "promotional": "unsolicited marketing, newsletter, or sales offer",
            "phishing": "fraudulent lure, credential theft, impersonation, or scam",
        },
    },
}

# Hand-picked in the POC; replaced by weights fitted on validation in Phase 3.
POC_WEIGHTS = {
    "is_phishing": 1.0,
    "urgency_pressure": 0.6,
    "credential_harvest": 0.8,
    "financial_pressure": 0.8,
    "suspicious_attachment": 0.7,
}
POC_RISK_THRESHOLD = 1.05


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LayaDetector:
    def __init__(
        self,
        default_model: str = "english",
        device: str | None = None,
        weights: dict[str, float] | None = None,
        risk_threshold: float = POC_RISK_THRESHOLD,
        router: Any = None,
    ):
        """`router` lets tests pass a fake; otherwise laya and torch are loaded here."""
        self.name = f"laya-{default_model}"
        self.weights = dict(weights or POC_WEIGHTS)
        self.risk_threshold = risk_threshold
        if router is None:
            from laya import Router

            self.device = device or _pick_device()
            router = Router(default=default_model, device=self.device)
            router.preload([default_model])
        else:
            self.device = device or "fake"
        self.router = router

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        # Zero-shot until Phase 3 fits the signal weights on validation.
        return TrainingCost(hardware=self.device)

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        return [self._predict_one(e) for e in emails]

    def _predict_one(self, email: Email) -> Prediction:
        state = {"from": email.sender, "subject": email.subject, "body": email.body}
        start = time.perf_counter()
        raw = self.router.predict(state, QUESTIONS)
        latency_ms = (time.perf_counter() - start) * 1000

        answers = raw.get("answers") if isinstance(raw, dict) else None
        signals, choice, class_probs, missing = self._parse(answers)
        if missing:
            return Prediction(
                score=None,
                latency_ms=latency_ms,
                error=f"Laya response missing or malformed: {', '.join(missing)}",
            )

        composite = sum(self.weights[q] * signals[q] for q in self.weights)
        score = composite / sum(self.weights.values())
        # The POC's verdict rule, kept for reference; metrics use `score`.
        flagged = composite >= self.risk_threshold or choice == "phishing"
        return Prediction(
            score=score,
            label=PHISHING if flagged else LEGITIMATE,
            latency_ms=latency_ms,
            details={
                "signals": signals,
                "classification": choice,
                "classification_probabilities": class_probs,
                "composite": composite,
            },
        )

    def _parse(self, answers: Any) -> tuple[dict[str, float], str | None, dict, list[str]]:
        if not isinstance(answers, dict):
            return {}, None, {}, ["answers"]
        signals: dict[str, float] = {}
        missing: list[str] = []
        for q in self.weights:
            value = answers.get(q, {}).get("noul") if isinstance(answers.get(q), dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                signals[q] = float(value)
            else:
                missing.append(f"{q}.noul")
        cls = answers.get("classification")
        choice = cls.get("choice") if isinstance(cls, dict) else None
        if not isinstance(choice, str):
            missing.append("classification.choice")
        probs = cls.get("probabilities", {}) if isinstance(cls, dict) else {}
        return signals, choice, probs, missing
