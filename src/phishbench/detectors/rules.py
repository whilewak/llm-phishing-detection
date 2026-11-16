"""Handwritten heuristic rules detector for phishing emails.

Evaluates multi-vector threat signals:
- Urgency and coercive language
- Credential and authentication harvesting
- Financial pressure, wire transfer, and invoice fraud
- Suspicious URL patterns (IP addresses, high-risk TLDs, URL shorteners, excessive subdomains)
- Subject line anomalies and display name spoofing cues

Produces a normalized continuous risk score in [0.0, 1.0].
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Sequence
from urllib.parse import urlparse

from phishbench.types import Email, Prediction, TrainingCost

# Regex patterns for high-confidence phishing indicators
_URGENCY_PATTERNS = [
    re.compile(r"\b(?:immediate(?:ly)?\s+action|action\s+required)\b", re.IGNORECASE),
    re.compile(r"\b(?:account|access)\s+(?:suspended|restricted|terminated|disabled|frozen)\b", re.IGNORECASE),
    re.compile(r"\bwithin\s+(?:24|48|12|hours?)\b", re.IGNORECASE),
    re.compile(r"\b(?:urgent|critical|security\s+alert|final\s+notice)\b", re.IGNORECASE),
    re.compile(r"\bfailure\s+to\s+(?:verify|respond|confirm)\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\s+(?:access|activity|login|sign-in)\b", re.IGNORECASE),
]

_CREDENTIAL_PATTERNS = [
    re.compile(r"\b(?:verify|confirm|validate|update)\s+your\s+(?:account|identity|information|email|profile)\b", re.IGNORECASE),
    re.compile(r"\b(?:reset|change|update)\s+your\s+password\b", re.IGNORECASE),
    re.compile(r"\b(?:enter|provide)\s+(?:your\s+)?(?:credentials|password|passcode|pin|security\s+code)\b", re.IGNORECASE),
    re.compile(r"\b(?:sign\s+in|log\s+in)\s+to\s+(?:verify|confirm|unlock|review)\b", re.IGNORECASE),
    re.compile(r"\b(?:mfa|2fa|otp|one-time\s+passcode|token)\s+verification\b", re.IGNORECASE),
]

_FINANCIAL_PATTERNS = [
    re.compile(r"\b(?:wire\s+transfer|remittance|direct\s+deposit)\b", re.IGNORECASE),
    re.compile(r"\b(?:overdue|unpaid|past\s+due)\s+invoice\b", re.IGNORECASE),
    re.compile(r"\b(?:gift\s+cards?|itunes\s+card|steam\s+card)\b", re.IGNORECASE),
    re.compile(r"\b(?:bitcoin|cryptocurrency|crypto\s+wallet|btc)\b", re.IGNORECASE),
    re.compile(r"\b(?:lottery|inheritance|next\s+of\s+kin|beneficiary|funds\s+transfer|diplomatic\s+courier)\b", re.IGNORECASE),
    re.compile(r"\b(?:payment\s+confirmation|payment\s+status|transaction\s+declined)\b", re.IGNORECASE),
]

_HIGH_RISK_TLDS = {
    ".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".club",
    ".work", ".icu", ".cam", ".rest", ".fit", ".bid", ".live", ".loan"
}

_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at"
}

_IP_URL_RE = re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/.*)?$", re.IGNORECASE)


class RulesDetector:
    """Heuristic rule-based phishing detector."""

    def __init__(
        self,
        urgency_weight: float = 1.2,
        credential_weight: float = 1.8,
        financial_weight: float = 1.5,
        url_weight: float = 1.4,
        scale: float = 1.1,
        offset: float = 1.8,
    ):
        self.name = "rules"
        self.urgency_weight = urgency_weight
        self.credential_weight = credential_weight
        self.financial_weight = financial_weight
        self.url_weight = url_weight
        self.scale = scale
        self.offset = offset

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        """Rules are zero-shot."""
        return TrainingCost()

    def _score_urls(self, urls: Sequence[str]) -> tuple[float, list[str]]:
        if not urls:
            return 0.0, []
        points = 0.0
        reasons = []
        for u in urls:
            lower = u.lower().strip()
            # IP address host
            if _IP_URL_RE.match(lower):
                points += 1.5
                reasons.append("url_ip_host")
            try:
                parsed = urlparse(lower)
                hostname = parsed.hostname or ""
                # High risk TLD
                for tld in _HIGH_RISK_TLDS:
                    if hostname.endswith(tld):
                        points += 1.2
                        reasons.append(f"high_risk_tld:{tld}")
                        break
                # URL shortener
                if hostname in _SHORTENER_DOMAINS:
                    points += 0.8
                    reasons.append("url_shortener")
                # @ in URL (auth spoofing)
                if "@" in parsed.netloc:
                    points += 1.8
                    reasons.append("url_auth_spoofing")
                # Deep subdomains (e.g., paypal.com.attacker.com)
                if hostname.count(".") >= 4:
                    points += 0.6
                    reasons.append("excessive_subdomains")
            except Exception:
                pass
        return points, reasons

    def score_email(self, email: Email) -> tuple[float, dict[str, Any]]:
        text = f"{email.subject} {email.body}"
        triggered: list[str] = []
        raw_score = 0.0

        # 1. Urgency / Coercion
        urgency_hits = sum(1 for p in _URGENCY_PATTERNS if p.search(text))
        if urgency_hits:
            pts = min(2.5, urgency_hits * 0.7) * self.urgency_weight
            raw_score += pts
            triggered.append(f"urgency_hits:{urgency_hits}")

        # 2. Credential harvesting
        cred_hits = sum(1 for p in _CREDENTIAL_PATTERNS if p.search(text))
        if cred_hits:
            pts = min(3.0, cred_hits * 0.9) * self.credential_weight
            raw_score += pts
            triggered.append(f"credential_hits:{cred_hits}")

        # 3. Financial / Wire
        fin_hits = sum(1 for p in _FINANCIAL_PATTERNS if p.search(text))
        if fin_hits:
            pts = min(3.0, fin_hits * 0.85) * self.financial_weight
            raw_score += pts
            triggered.append(f"financial_hits:{fin_hits}")

        # 4. URLs
        url_pts, url_reasons = self._score_urls(email.urls)
        if url_pts > 0:
            raw_score += min(3.0, url_pts) * self.url_weight
            triggered.extend(url_reasons)

        # 5. Subject urgency check
        subj_lower = email.subject.lower()
        if any(w in subj_lower for w in ("urgent", "action required", "important notice", "account alert")):
            raw_score += 0.8
            triggered.append("urgent_subject")

        # Map to continuous [0.0, 1.0] using logistic sigmoid
        # When raw_score == offset, prob = 0.5
        score = 1.0 / (1.0 + math.exp(-self.scale * (raw_score - self.offset)))
        # Clean rounding to avoid floating point noise
        score = min(1.0, max(0.0, float(score)))

        return score, {"raw_score": raw_score, "triggered": triggered}

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        results = []
        for e in emails:
            t0 = time.perf_counter()
            score, details = self.score_email(e)
            latency = (time.perf_counter() - t0) * 1000.0
            results.append(
                Prediction(
                    score=score,
                    latency_ms=latency,
                    cost_usd=0.0,
                    details=details,
                )
            )
        return results

