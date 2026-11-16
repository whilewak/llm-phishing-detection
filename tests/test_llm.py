"""LLM detectors with fake clients: no API key or network needed."""

import importlib.util
import math
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from phishbench import prompts
from phishbench.detectors.llm_base import (
    CallResult,
    CostLimitExceeded,
    LLMDetector,
    RiskAssessment,
    TransientAPIError,
    Usage,
    score_from_json,
    score_from_logprobs,
)
from tests.stubs import make_email, make_split

HAS_SDKS = importlib.util.find_spec("anthropic") is not None
PRICES = {"input": 2.0, "cache_write": 2.5, "cache_read": 0.1, "output": 10.0}


class ScriptedLLM(LLMDetector):
    provider = "scripted"

    def __init__(self, results=None, delay=0.0, **kw):
        self.results = results
        self.delay = delay
        self.calls = []
        kw.setdefault("cache_dir", None)
        super().__init__(name="scripted", model="m", prices=PRICES, **kw)

    def _call(self, user_text):
        self.calls.append(user_text)
        if self.delay:
            time.sleep(self.delay * (1 + (hash(user_text) % 3)))
        r = self.results(user_text) if callable(self.results) else self.results
        if isinstance(r, Exception):
            raise r
        return r or CallResult(score=0.5, usage=Usage(input=100, output=10))


class ScoringTest(unittest.TestCase):
    def test_json(self):
        r = score_from_json(RiskAssessment(risk_score=87, verdict="phishing"))
        self.assertAlmostEqual(r.score, 0.87)
        self.assertEqual(r.label, "phishing")
        r = score_from_json(None, '{"risk_score": 3, "verdict": "legitimate"}')
        self.assertAlmostEqual(r.score, 0.03)
        for bad in ('{"risk_score": 140, "verdict": "phishing"}', "not json", '{"risk_score": 5}'):
            self.assertIsNone(score_from_json(None, bad).score)
            self.assertIsNotNone(score_from_json(None, bad).error)
        self.assertIsNotNone(score_from_json(None, None).error)

    def test_logprobs(self):
        r = score_from_logprobs([("phishing", math.log(0.6)), ("legitimate", math.log(0.2)), ("maybe", math.log(0.2))])
        self.assertAlmostEqual(r.score, 0.75)
        r = score_from_logprobs([(" Phish", math.log(0.3)), ("legit", math.log(0.1))])
        self.assertAlmostEqual(r.score, 0.75)
        self.assertAlmostEqual(score_from_logprobs([("legitimate", 0.0)]).score, 0.0)
        self.assertIsNotNone(score_from_logprobs([("yes", 0.0), ("", 0.0)]).error)

    def test_cost(self):
        u = Usage(input=1_000_000, cache_write=1_000_000, cache_read=1_000_000, output=1_000_000)
        self.assertAlmostEqual(u.cost(PRICES), 2.0 + 2.5 + 0.1 + 10.0)


class BaseDetectorTest(unittest.TestCase):
    def test_order_kept_under_concurrency(self):
        det = ScriptedLLM(results=lambda t: CallResult(score=int(t.split("body ")[1].split()[0]) / 100),
                          delay=0.002, concurrency=8)
        emails = [make_email(i, "phishing") for i in range(30)]
        preds = det.predict(emails)
        self.assertEqual([p.score for p in preds], [i / 100 for i in range(30)])

    def test_cache_hit_skips_request_and_keeps_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            det = ScriptedLLM(cache_dir=d, results=CallResult(score=0.9, usage=Usage(input=1000, output=20)))
            (first,) = det.predict([make_email(1, "phishing")])
            (second,) = det.predict([make_email(1, "phishing")])
            self.assertEqual(len(det.calls), 1)
            self.assertFalse(first.details["cached"])
            self.assertTrue(second.details["cached"])
            self.assertEqual(first.latency_ms, second.latency_ms)
            self.assertEqual(first.cost_usd, second.cost_usd)
            self.assertAlmostEqual(first.cost_usd, (1000 * 2.0 + 20 * 10.0) / 1e6)
            # A fresh detector (new process) reuses the same cache.
            det2 = ScriptedLLM(cache_dir=d)
            det2.predict([make_email(1, "phishing")])
            self.assertEqual(det2.calls, [])

    def test_errors_cached_but_transient_errors_not(self):
        with tempfile.TemporaryDirectory() as d:
            det = ScriptedLLM(cache_dir=d, results=CallResult(score=None, error="refusal"))
            det.predict([make_email(1, "phishing")])
            (p,) = det.predict([make_email(1, "phishing")])
            self.assertEqual(len(det.calls), 1)
            self.assertTrue(p.failed)

            det = ScriptedLLM(cache_dir=d, results=TransientAPIError("503"))
            (p,) = det.predict([make_email(2, "phishing")])
            self.assertTrue(p.failed)
            self.assertIn("not cached", p.error)
            det.predict([make_email(2, "phishing")])
            self.assertEqual(len(det.calls), 2)

    def test_cost_guard_stops_before_sending(self):
        det = ScriptedLLM(max_total_cost_usd=0.0001, max_output_tokens=1000)
        with self.assertRaises(CostLimitExceeded):
            det.predict([make_email(1, "phishing")])
        self.assertEqual(det.calls, [])

    def test_cost_guard_tracks_actual_spend(self):
        det = ScriptedLLM(max_total_cost_usd=0.05, max_output_tokens=10,
                          results=CallResult(score=0.5, usage=Usage(input=0, output=1000)))  # $0.01 each
        det.predict([make_email(i, "phishing") for i in range(4)])
        self.assertAlmostEqual(det.spent_usd, 0.04)
        with self.assertRaises(CostLimitExceeded):
            det.predict([make_email(i, "phishing") for i in range(10, 13)])

    def test_details_and_truncation_recorded(self):
        det = ScriptedLLM(max_input_chars=10, results=CallResult(score=0.2, usage=Usage(input=5, reasoning=3, output=7)))
        e = make_email(1, "legitimate")
        long = type(e)(id="long", label="legitimate", sender="s", subject="t", body="x " * 100)
        p_short, p_long = det.predict([e, long])
        self.assertFalse(p_short.details["truncated"])
        self.assertTrue(p_long.details["truncated"])
        self.assertEqual(p_short.details["reasoning_tokens"], 3)
        self.assertEqual(p_short.details["prompt_version"], prompts.PROMPT_VERSION)

    def test_runs_through_harness(self):
        from phishbench.eval import evaluate

        det = ScriptedLLM(results=lambda t: CallResult(score=0.9 if "body 1" in t and "body 10" not in t else 0.1))
        result = evaluate(det, make_split(3, 3), val=make_split(3, 3, start=50), warmup=1, n_bootstrap=20)
        self.assertEqual(result["errors"]["count"], 0)
        self.assertEqual(result["training_cost"]["hardware"], "API")


# --- Claude ------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self, response=None, error=None):
        self.requests = []
        self.response, self.error = response, error
        self.messages = NS(parse=self._parse)

    def _parse(self, **kw):
        self.requests.append(kw)
        if self.error:
            raise self.error
        return self.response


def claude_response(score=91, verdict="phishing", stop="end_turn", category=None):
    parsed = RiskAssessment(risk_score=score, verdict=verdict) if stop == "end_turn" else None
    return NS(
        stop_reason=stop,
        stop_details=NS(category=category) if category else None,
        parsed_output=parsed,
        content=[NS(type="text", text=parsed.model_dump_json() if parsed else "")],
        usage=NS(input_tokens=300, cache_creation_input_tokens=0, cache_read_input_tokens=200, output_tokens=15),
    )


@unittest.skipUnless(HAS_SDKS, "anthropic SDK not installed")
class ClaudeTest(unittest.TestCase):
    def make(self, client, **kw):
        from phishbench.detectors.claude import ClaudeDetector

        kw.setdefault("model", "claude-sonnet-5-5")
        kw.setdefault("cache_dir", None)
        return ClaudeDetector(prices=PRICES, thinking={"type": "between_tools"}, client=client, **kw)

    def test_request_shape_and_score(self):
        client = FakeAnthropic(claude_response())
        det = self.make(client)
        (p,) = det.predict([make_email(1, "phishing")])
        self.assertAlmostEqual(p.score, 0.91)
        self.assertEqual(det.name, "claude-sonnet-5-5")
        req = client.requests[0]
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["thinking"], {"type": "between_tools"})
        self.assertEqual(req["output_config"], {"effort": "low"})
        self.assertNotIn("temperature", req)
        self.assertEqual(req["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIs(req["output_format"], RiskAssessment)
        self.assertAlmostEqual(p.cost_usd, (300 * 2.0 + 200 * 0.1 + 15 * 10.0) / 1e6)

    def test_refusal_and_max_tokens_are_errors(self):
        (p,) = self.make(FakeAnthropic(claude_response(stop="refusal", category="cyber"))).predict([make_email(1, "phishing")])
        self.assertTrue(p.failed)
        self.assertIn("cyber", p.error)
        (p,) = self.make(FakeAnthropic(claude_response(stop="max_tokens"))).predict([make_email(1, "phishing")])
        self.assertIn("max_tokens", p.error)

    def test_transient_and_fatal_api_errors(self):
        import anthropic
        import httpx2

        req = httpx2.Request("POST", "https://x")
        transient = anthropic.InternalServerError("boom", response=httpx2.Response(503, request=req), body=None)
        (p,) = self.make(FakeAnthropic(error=transient)).predict([make_email(1, "phishing")])
        self.assertIn("not cached", p.error)
        fatal = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body=None)
        with self.assertRaises(anthropic.BadRequestError):
            self.make(FakeAnthropic(error=fatal)).predict([make_email(1, "phishing")])

    def test_hardware_label(self):
        self.assertEqual(self.make(FakeAnthropic()).fit([], []).hardware, "API (Anthropic)")

    def test_rejects_logprob_mode(self):
        with self.assertRaises(ValueError):
            self.make(FakeAnthropic(), answer_mode=prompts.ANSWER_TOKEN)


@unittest.skipUnless(HAS_SDKS, "anthropic SDK not installed")
class ConfigsTest(unittest.TestCase):
    def test_phase5_configs_build_with_fake_clients(self):
        import tomllib

        from phishbench.detectors import load_detector

        root = Path(__file__).resolve().parents[1]
        configs = sorted((root / "configs" / "models").glob("claude-*.toml"))
        self.assertEqual(len(configs), 5)
        pool = root / "configs" / "prompts" / "fewshot-champa2023-v1.json"
        for path in configs:
            cfg = tomllib.loads(path.read_text())["detector"]
            params = dict(cfg["params"], client=object(), cache_dir=None)
            if "few_shot" in params and not pool.exists():
                continue
            cwd = os.getcwd()
            try:
                os.chdir(root)
                det = load_detector(cfg["class"], params)
            finally:
                os.chdir(cwd)
            self.assertEqual(det.name, path.stem)


@unittest.skipUnless(HAS_SDKS and os.environ.get("ANTHROPIC_API_KEY"), "needs the SDK and ANTHROPIC_API_KEY")
class AnthropicIntegrationTest(unittest.TestCase):
    def test_sample_emails(self):
        import tomllib

        from phishbench.data import load_emails
        from phishbench.detectors import load_detector

        root = Path(__file__).resolve().parents[1]
        emails = load_emails(root / "data" / "benchmark_emails.json")
        cfg = tomllib.loads((root / "configs" / "models" / "claude-haiku-5-5-zeroshot.toml").read_text())["detector"]
        det = load_detector(cfg["class"], cfg["params"])
        preds = det.predict(emails)
        self.assertEqual([p.error for p in preds], [None] * len(emails))

if __name__ == "__main__":
    unittest.main()
