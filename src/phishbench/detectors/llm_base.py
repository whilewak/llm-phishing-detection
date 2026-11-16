"""Shared machinery for API-hosted LLM detectors.

Providers implement `_call`, which sends one request and turns the response
into a `CallResult`. This base adds what every LLM needs: the shared prompt,
concurrency, an on-disk response cache (reruns cost nothing), cost accounting,
a per-run cost guard and the two scoring modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from phishbench import prompts
from phishbench.types import LEGITIMATE, PHISHING, Email, Prediction, TrainingCost

DEFAULT_CACHE_DIR = Path("data/cache/llm")


class RiskAssessment(BaseModel):
    """The structured answer every 0-100 model returns."""

    risk_score: int = Field(ge=0, le=100)
    verdict: Literal["phishing", "legitimate"]


class CostLimitExceeded(RuntimeError):
    """Raised instead of sending a request that could take spend past the cap."""


class TransientAPIError(RuntimeError):
    """Throttling, server or network failure left after the SDK's retries.
    Counted as an error for this run but not cached, so a rerun retries it."""


@dataclass
class Usage:
    input: int = 0  # uncached input tokens
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0  # includes reasoning tokens
    reasoning: int = 0

    def cost(self, prices: dict[str, float]) -> float:
        return (
            self.input * prices["input"]
            + self.cache_write * prices.get("cache_write", prices["input"])
            + self.cache_read * prices.get("cache_read", prices["input"])
            + self.output * prices["output"]
        ) / 1e6


@dataclass
class CallResult:
    """One provider response, already scored. `error` set means no usable score."""

    score: float | None
    label: str | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    details: dict[str, Any] = field(default_factory=dict)


def score_from_json(parsed: RiskAssessment | None, raw_text: str | None = None) -> CallResult:
    if parsed is None and raw_text:
        try:
            parsed = RiskAssessment.model_validate_json(raw_text)
        except ValidationError as exc:
            return CallResult(score=None, error=f"unparseable answer: {exc.errors()[0]['msg']}")
    if parsed is None:
        return CallResult(score=None, error="no answer returned")
    return CallResult(
        score=parsed.risk_score / 100,
        label=PHISHING if parsed.verdict == "phishing" else LEGITIMATE,
        details={"risk_score": parsed.risk_score, "verdict": parsed.verdict},
    )


def _norm(token: str) -> str:
    return token.strip().lower()


def score_from_logprobs(top_logprobs: Sequence[tuple[str, float]]) -> CallResult:
    """P(phishing) / (P(phishing) + P(legitimate)) at the first answer token.

    Tokenisers may split the words ("phish" + "ing", "legit" + "imate"), so a
    candidate counts if it is a non-empty prefix of one answer word only.
    """
    p = {"phishing": 0.0, "legitimate": 0.0}
    for token, logprob in top_logprobs:
        t = _norm(token)
        if not t:
            continue
        matches = [w for w in p if w.startswith(t)]
        if len(matches) == 1:
            p[matches[0]] += math.exp(logprob)
    total = p["phishing"] + p["legitimate"]
    if total == 0:
        return CallResult(score=None, error="neither answer token among the top logprobs")
    score = p["phishing"] / total
    return CallResult(
        score=score,
        label=PHISHING if score >= 0.5 else LEGITIMATE,
        details={"p_phishing": p["phishing"], "p_legitimate": p["legitimate"]},
    )


class LLMDetector(ABC):
    """Base for API LLM detectors. Subclasses set `provider` and implement `_call`."""

    provider: str = "llm"

    def __init__(
        self,
        name: str,
        model: str,
        prices: dict[str, float],
        answer_mode: str = prompts.SCORE_0_100,
        few_shot: str | None = None,
        split_dir: str | None = None,
        max_output_tokens: int = 256,
        max_input_chars: int = prompts.MAX_INPUT_CHARS,
        concurrency: int = 8,
        max_total_cost_usd: float | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        request_params: dict[str, Any] | None = None,
    ):
        self.name = name
        self.model = model
        self.prices = dict(prices)
        for key in ("input", "output"):
            if key not in self.prices:
                raise ValueError(f"prices must include {key!r}")
        self.answer_mode = answer_mode
        self.few_shot_path = few_shot
        self.few_shot = prompts.load_fewshot(few_shot, split_dir) if few_shot else []
        self.system = prompts.system_prompt(answer_mode, self.few_shot)
        self.max_output_tokens = max_output_tokens
        self.max_input_chars = max_input_chars
        self.concurrency = max(1, concurrency)
        self.max_total_cost_usd = max_total_cost_usd
        self.cache_dir = Path(cache_dir) / self.provider if cache_dir else None
        self.request_params = dict(request_params or {})
        self.spent_usd = 0.0
        self.api_calls = 0
        self._lock = threading.Lock()

    # --- provider hook -------------------------------------------------------

    @abstractmethod
    def _call(self, user_text: str) -> CallResult:
        """Send one request for one rendered email and score the response."""

    # --- Detector interface --------------------------------------------------

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        return TrainingCost(hardware="API (Amazon Bedrock)", n_train_emails=len(self.few_shot))

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        if self.concurrency == 1 or len(emails) <= 1:
            return [self._predict_one(e) for e in emails]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._predict_one, emails))

    # --- internals -----------------------------------------------------------

    def _cache_key(self, user_text: str) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "answer_mode": self.answer_mode,
            "max_output_tokens": self.max_output_tokens,
            "params": self.request_params,
            "prompt_version": prompts.PROMPT_VERSION,
            "system": self.system,
            "user": user_text,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _worst_case_cost(self, user_text: str) -> float:
        # ~3 characters per token is a deliberately high token estimate.
        in_tokens = (len(self.system) + len(user_text)) / 3
        return Usage(input=int(in_tokens), output=self.max_output_tokens).cost(self.prices)

    def _reserve(self, estimate: float) -> None:
        with self._lock:
            if self.max_total_cost_usd is not None and self.spent_usd + estimate > self.max_total_cost_usd:
                raise CostLimitExceeded(
                    f"{self.name}: spent ${self.spent_usd:.2f}; the next request could pass the "
                    f"${self.max_total_cost_usd:.2f} cap. Raise max_total_cost_usd to continue "
                    "(cached responses are reused for free)."
                )
            self.spent_usd += estimate

    def _settle(self, estimate: float, actual: float) -> None:
        with self._lock:
            self.spent_usd += actual - estimate
            self.api_calls += 1

    def _predict_one(self, email: Email) -> Prediction:
        user_text, truncated = prompts.render_email(email, self.max_input_chars)
        key = self._cache_key(user_text)
        path = self._cache_path(key)

        if path is not None and path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            return self._to_prediction(rec, truncated, cached=True)

        estimate = self._worst_case_cost(user_text)
        self._reserve(estimate)
        start = time.perf_counter()
        try:
            result = self._call(user_text)
        except TransientAPIError as exc:
            self._settle(estimate, 0.0)
            return Prediction(
                score=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"transient API error (not cached): {exc}",
                details={"truncated": truncated, "cached": False},
            )
        except BaseException:
            self._settle(estimate, 0.0)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        cost = result.usage.cost(self.prices)
        self._settle(estimate, cost)

        rec = {
            "model": self.model,
            "prompt_version": prompts.PROMPT_VERSION,
            "score": result.score,
            "label": result.label,
            "error": result.error,
            "usage": asdict(result.usage),
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "details": result.details,
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            tmp.replace(path)
        return self._to_prediction(rec, truncated, cached=False)

    def _to_prediction(self, rec: dict, truncated: bool, cached: bool) -> Prediction:
        usage = rec["usage"]
        return Prediction(
            score=rec["score"] if rec["error"] is None else None,
            label=rec["label"],
            latency_ms=rec["latency_ms"],
            cost_usd=rec["cost_usd"],
            error=rec["error"],
            details={
                **rec["details"],
                "truncated": truncated,
                "cached": cached,
                "prompt_version": rec["prompt_version"],
                "input_tokens": usage["input"],
                "cache_write_tokens": usage["cache_write"],
                "cache_read_tokens": usage["cache_read"],
                "output_tokens": usage["output"],
                "reasoning_tokens": usage["reasoning"],
            },
        )


def bedrock_api_key(env_var: str = "AWS_BEARER_TOKEN_BEDROCK") -> str:
    import os

    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"set {env_var} to an Amazon Bedrock API key (see docs/phase5-plan.md)")
    return key


def bedrock_base_url(region: str, path: str) -> str:
    return f"https://bedrock-runtime.{region}.amazonaws.com/{path}"
