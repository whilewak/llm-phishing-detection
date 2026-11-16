"""Claude (Sonnet 5.5, Haiku 5.5) as a Detector.

Two backends, same request:
- "anthropic" (default): Anthropic's API, key from ANTHROPIC_API_KEY (or an
  `ant auth login` profile). Model ids like "claude-haiku-5-5".
- "bedrock": Amazon Bedrock's Messages API, key from AWS_BEARER_TOKEN_BEDROCK.
  Model ids like "global.anthropic.claude-haiku-5-5".

Scores are the model's 0-100 risk score / 100 (Claude returns no token
probabilities). Thinking is configured per model in the TOML: Sonnet 5.5 turns
it off with {"type": "between_tools"}, Haiku 5.5 with {"type": "disabled"}.
Current Claude models reject `temperature`, so none is sent. Refusals are
errors; no refusal fallback is configured, since it would answer with a
different model.
"""

from __future__ import annotations

from typing import Any

from phishbench import prompts
from phishbench.detectors.llm_base import (
    CallResult,
    LLMDetector,
    RiskAssessment,
    TransientAPIError,
    Usage,
    bedrock_api_key,
    bedrock_base_url,
    score_from_json,
)

BACKENDS = {
    "anthropic": ("anthropic", "API (Anthropic)"),
    "bedrock": ("anthropic-bedrock", "API (Amazon Bedrock)"),
}


class ClaudeDetector(LLMDetector):
    def __init__(
        self,
        model: str,
        prices: dict[str, float],
        thinking: dict[str, Any] | None = None,
        effort: str | None = "low",
        backend: str = "anthropic",
        region: str = "us-west-1",
        name: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {backend!r}")
        if kwargs.get("answer_mode", prompts.SCORE_0_100) != prompts.SCORE_0_100:
            raise ValueError("Claude returns no logprobs; only the 0-100 score mode is supported")
        # Separate cache namespaces, so responses from the two backends never mix.
        self.provider, self.hardware = BACKENDS[backend]
        self.backend = backend
        self.thinking = thinking
        self.effort = effort
        self.region = region
        params = {"thinking": thinking, "effort": effort}
        super().__init__(name=name or model.split(".")[-1], model=model, prices=prices,
                         request_params=params, **kwargs)
        if client is None:
            import anthropic

            if backend == "bedrock":
                client = anthropic.Anthropic(
                    base_url=bedrock_base_url(region, "anthropic"),
                    api_key=bedrock_api_key(),
                    max_retries=5,
                )
            else:
                client = anthropic.Anthropic(max_retries=5)
        self.client = client

    def _request_kwargs(self, user_text: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_text}],
            "output_format": RiskAssessment,
        }
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def _call(self, user_text: str) -> CallResult:
        try:
            response = self.client.messages.parse(**self._request_kwargs(user_text))
        except Exception as exc:  # noqa: BLE001 - classified below
            if _is_transient(exc):
                raise TransientAPIError(f"{type(exc).__name__}: {exc}") from exc
            raise

        u = response.usage
        usage = Usage(
            input=u.input_tokens or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            output=u.output_tokens or 0,
        )
        stop = response.stop_reason
        if stop == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return CallResult(score=None, error=f"refusal (category: {category})", usage=usage,
                              details={"stop_reason": stop, "refusal_category": category})
        if stop == "max_tokens":
            return CallResult(score=None, error="hit max_tokens before answering", usage=usage,
                              details={"stop_reason": stop})

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        result = score_from_json(getattr(response, "parsed_output", None), text)
        result.usage = usage
        result.details["stop_reason"] = stop
        return result


def _is_transient(exc: Exception) -> bool:
    try:
        import anthropic
    except ImportError:  # fake clients in tests
        return isinstance(exc, (ConnectionError, TimeoutError))
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
