import math
import os
import tempfile
import unittest
from collections import UserDict
from pathlib import Path

from phishbench import prompts
from phishbench.detectors.qwen import MLXBackend, QwenDetector
from phishbench.types import Email
from tests.stubs import make_email, make_split

ROOT = Path(__file__).resolve().parents[1]
FEW_SHOT = ROOT / "configs/prompts/fewshot-champa2023-v1.json"
SPLITS = ROOT / "data/splits/champa2023-v1"


class FakeBackend:
    def __init__(self):
        self.calls = 0
        self.messages = None

    def render(self, messages):
        self.messages = messages
        return list(str(messages).encode())

    def score(self, tokens, prefix_length=0):
        self.calls += 1
        return math.log(.6), math.log(.2), {}


class QwenTest(unittest.TestCase):
    def detector(self, **kwargs):
        return QwenDetector(backend=FakeBackend(), cache_dir=None, **kwargs)

    def few_detector(self):
        return self.detector(n_shots=8, few_shot=str(FEW_SHOT), split_dir=str(SPLITS),
                             max_input_tokens=20000)

    def test_chat_template_accepts_tokenizer_mapping(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return UserDict({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        backend = object.__new__(MLXBackend)
        backend.tokenizer = Tokenizer()
        self.assertEqual(backend.render([{"role": "user", "content": "email"}]), [1, 2, 3])

    def test_conditional_probability(self):
        p, = self.detector().predict([make_email(0, "phishing")])
        self.assertAlmostEqual(p.score, .75)
        self.assertAlmostEqual(p.details["answer_probability_mass"], .8)

    def test_zero_shot_uses_shared_task_and_email(self):
        det = self.detector()
        email = make_email(0, "phishing")
        det.predict([email])
        self.assertEqual(det.backend.messages[0]["content"],
                         prompts.TASK_INSTRUCTIONS + "\n\n" +
                         "Reply with exactly A for phishing or B for legitimate.")
        self.assertEqual(det.backend.messages[1]["content"], prompts.render_email(email)[0])
        self.assertEqual([m["role"] for m in det.backend.messages], ["system", "user"])

    def test_test_label_never_enters_prompt(self):
        det = self.detector()
        a, = det.predict([make_email(0, "phishing")])
        seen = det.backend.messages
        b, = det.predict([make_email(0, "legitimate")])
        self.assertEqual(det.backend.messages, seen)
        self.assertEqual(a.score, b.score)

    def test_eight_shot_uses_claudes_frozen_pool(self):
        examples = prompts.load_fewshot(FEW_SHOT, SPLITS)
        det = self.few_detector()
        self.assertEqual(det.fit(examples, []).n_train_emails, 8)
        self.assertEqual(det.examples, examples)
        self.assertEqual([e.id for e in examples], [
            "nazario:267", "enron:3375", "nigerian:1272", "ling:2068",
            "trec07:18384", "spamassassin:2820", "trec06:1105", "trec06:10097",
        ])
        p, = det.predict([make_email(200, "phishing")])
        self.assertFalse(p.failed, p.error)
        self.assertEqual(p.details["example_ids"], [e.id for e in examples])
        system = det.backend.messages[0]["content"]
        self.assertTrue(system.startswith(prompts.TASK_INSTRUCTIONS + "\n\n"))
        self.assertIn("\n\nExamples:\n\nExample 1:\n", system)
        for i, example in enumerate(examples, 1):
            answer = "A" if example.is_phishing else "B"
            self.assertIn(f"Example {i}:\n{prompts.render_email(example)[0]}\nAnswer: {answer}", system)
        with self.assertRaises(ValueError):
            det.predict([examples[0]])

    def test_eight_shot_requires_frozen_train_examples(self):
        det = self.few_detector()
        with self.assertRaises(RuntimeError):
            det.predict([make_email(99, "phishing")])
        with self.assertRaises(ValueError):
            det.fit(make_split(4, 4), [])
        examples = prompts.load_fewshot(FEW_SHOT, SPLITS)
        with self.assertRaises(ValueError):
            det.fit(examples, examples)
        with self.assertRaises(ValueError):
            QwenDetector(n_shots=8)
        with self.assertRaises(ValueError):
            QwenDetector(n_shots=4)

    def test_truncation_preserves_headers_urls_and_wrapper(self):
        det = self.detector(max_input_tokens=1800)
        email = Email("long", "phishing", "sender@example.com", "Subject", "x" * 10000,
                      ("https://example.com/path",))
        p, = det.predict([email])
        self.assertFalse(p.failed, p.error)
        self.assertTrue(p.details["truncated"])
        self.assertLessEqual(p.details["input_tokens"], 1800)
        user = det.backend.messages[1]["content"]
        self.assertIn("From: sender@example.com", user)
        self.assertIn("https://example.com/path", user)
        self.assertIn("[... truncated]", user)
        self.assertTrue(user.endswith("</email>"))

    def test_shared_body_cap_is_reported(self):
        det = self.detector(max_input_tokens=40000)
        p, = det.predict([Email("long", "phishing", "", "", "x" * 35000)])
        self.assertFalse(p.failed, p.error)
        self.assertTrue(p.details["truncated"])
        self.assertIn("[... truncated]", det.backend.messages[1]["content"])

    def test_cache_invalidation_and_original_timing(self):
        with tempfile.TemporaryDirectory() as root:
            det = QwenDetector(backend=FakeBackend(), cache_dir=root)
            a, = det.predict([make_email(0, "phishing")])
            b, = det.predict([make_email(0, "phishing")])
            self.assertEqual(det.backend.calls, 1)
            self.assertEqual(a.latency_ms, b.latency_ms)
            self.assertTrue(b.details["cache_hit"])
            det.revision = "1" * 40
            det.predict([make_email(0, "phishing")])
            self.assertEqual(det.backend.calls, 2)

    def test_backend_failure_is_error(self):
        det = self.detector()
        det.backend.score = lambda *args: (_ for _ in ()).throw(TimeoutError("timeout"))
        p, = det.predict([make_email(0, "phishing")])
        self.assertTrue(p.failed)
        self.assertIn("timeout", p.error)

    def test_nonfinite_logprobs_are_errors(self):
        det = self.detector()
        det.backend.score = lambda *args: (math.nan, -1, {})
        self.assertTrue(det.predict([make_email(0, "phishing")])[0].failed)


# Explicit opt-in prevents unit tests from downloading weights or using a GPU.
@unittest.skipUnless(os.environ.get('PHISHBENCH_QWEN_INTEGRATION') == '1',
                     'set PHISHBENCH_QWEN_INTEGRATION=1 for local model integration')
class QwenIntegrationTest(unittest.TestCase):
    def test_real_zero_and_eight_shot(self):
        from phishbench.data.splits import load_split
        emails = load_split(SPLITS, "test_llm")[:2]
        train = load_split(SPLITS, "train")
        zero = QwenDetector(cache_dir=None).predict(emails)
        eight_detector = QwenDetector(n_shots=8, few_shot=str(FEW_SHOT),
                                      split_dir=str(SPLITS), cache_dir=None)
        eight_detector.fit(train, [])
        eight = eight_detector.predict(emails)
        for p in zero + eight:
            self.assertFalse(p.failed, p.error)
            self.assertGreaterEqual(p.score, 0)
            self.assertLessEqual(p.score, 1)
            self.assertLessEqual(p.details['answer_probability_mass'], 1.00001)
