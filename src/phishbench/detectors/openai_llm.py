"""OpenAI GPT-6.1 Sol and GPT-6 Luna through Amazon Bedrock's Responses API.

Two scoring modes:
- `score_0_100`: structured JSON risk score / 100, same as Claude.
- `answer_token`: one-word answer, scored from the first token's top
  logprobs. OpenAI allows `temperature` and logprobs only at reasoning effort
  "none" (Luna, Terra), so this mode requires it.
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
    score_from_logprobs,
)

TOP_LOGPROBS = 20


class OpenAIDetector(LLMDetector):
    provider = "openai-bedrock"

    def __init__(
        self,
        model: str,
        prices: dict[str, float],
        effort: str = "none",
        temperature: float | None = None,
        cache_prefix: bool = False,
        region: str = "us-east-1",
        name: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        mode = kwargs.get("answer_mode", prompts.SCORE_0_100)
        if effort != "none" and (temperature is not None or mode == prompts.ANSWER_TOKEN):
            raise ValueError("temperature and logprobs are only accepted at reasoning effort 'none'")
        self.effort = effort
        self.temperature = temperature
        self.cache_prefix = cache_prefix
        self.region = region
        params = {"effort": effort, "temperature": temperature, "cache_prefix": cache_prefix}
        super().__init__(name=name or model.split(".")[-1], model=model, prices=prices,
                         request_params=params, **kwargs)
        if client is None:
            import openai

            client = openai.OpenAI(
                base_url=bedrock_base_url(region, "openai/v1"),
                api_key=bedrock_api_key(),
                max_retries=5,
            )
        self.client = client

    def _instructions(self) -> Any:
        if not self.cache_prefix:
            return {"instructions": self.system}
        # Bedrock explicit prompt caching: mark the fixed prefix as a breakpoint.
        return {
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": self.system,
                         "prompt_cache_breakpoint": {"mode": "explicit"}},
                    ],
                }
            ]
        }

    def _request_kwargs(self, user_text: str) -> dict[str, Any]:
        prefix = self._instructions()
        user_msg = {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "reasoning": {"effort": self.effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if "instructions" in prefix:
            kwargs["instructions"] = prefix["instructions"]
            kwargs["input"] = [user_msg]
        else:
            kwargs["input"] = prefix["input"] + [user_msg]
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.answer_mode == prompts.SCORE_0_100:
            kwargs["text_format"] = RiskAssessment
        else:
            kwargs["top_logprobs"] = TOP_LOGPROBS
            kwargs["include"] = ["message.output_text.logprobs"]
        return kwargs

    def _call(self, user_text: str) -> CallResult:
        kwargs = self._request_kwargs(user_text)
        try:
            if self.answer_mode == prompts.SCORE_0_100:
                response = self.client.responses.parse(**kwargs)
            else:
                response = self.client.responses.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below
            if _is_transient(exc):
                raise TransientAPIError(f"{type(exc).__name__}: {exc}") from exc
            raise

        usage = _usage(response.usage)
        texts, refusal, logprobs = [], None, None
        for item in response.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in item.content or []:
                if part.type == "refusal":
                    refusal = part.refusal
                elif part.type == "output_text":
                    texts.append(part.text)
                    if logprobs is None and getattr(part, "logprobs", None):
                        logprobs = part.logprobs

        details = {"status": response.status}
        if refusal is not None:
            return CallResult(score=None, error=f"refusal: {refusal[:200]}", usage=usage, details=details)
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            return CallResult(score=None, error=f"incomplete response ({reason})", usage=usage, details=details)

        if self.answer_mode == prompts.SCORE_0_100:
            result = score_from_json(getattr(response, "output_parsed", None), "".join(texts))
        elif not logprobs:
            result = CallResult(score=None, error="response carried no logprobs")
        else:
            first = logprobs[0]
            top = [(c.token, c.logprob) for c in (first.top_logprobs or [])] or [(first.token, first.logprob)]
            result = score_from_logprobs(top)
            result.details["answer_text"] = "".join(texts)[:40]
        result.usage = usage
        result.details.update(details)
        return result


def _usage(u: Any) -> Usage:
    if u is None:
        return Usage()
    inp = getattr(u, "input_tokens_details", None)
    cached = (getattr(inp, "cached_tokens", 0) or 0) if inp else 0
    written = (getattr(inp, "cache_write_tokens", 0) or 0) if inp else 0
    out = getattr(u, "output_tokens_details", None)
    reasoning = (getattr(out, "reasoning_tokens", 0) or 0) if out else 0
    return Usage(
        input=max(0, (u.input_tokens or 0) - cached - written),
        cache_write=written,
        cache_read=cached,
        output=u.output_tokens or 0,
        reasoning=reasoning,
    )


def _is_transient(exc: Exception) -> bool:
    try:
        import openai
    except ImportError:
        return isinstance(exc, (ConnectionError, TimeoutError))
    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500
