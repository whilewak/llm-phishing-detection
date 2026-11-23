import math
import tempfile
import unittest
from phishbench.detectors.qwen import QwenDetector
from phishbench.detectors.prompts import select_examples
from tests.stubs import make_split, make_email


class FakeBackend:
    def __init__(self):
        self.calls = 0
        self.messages = None
    def encode(self, text):
        return list(text.encode())
    def decode(self, tokens):
        return bytes(tokens).decode(errors="ignore")
    def render(self, messages):
        self.messages = messages
        return self.encode(str(messages))
    def score(self, tokens, prefix_length=0):
        self.calls += 1
        return math.log(.6), math.log(.2), {}


class QwenTest(unittest.TestCase):
    def detector(self, **kwargs):
        return QwenDetector(backend=FakeBackend(), cache_dir=None, **kwargs)
    def test_conditional_probability(self):
        det = self.detector()
        p, = det.predict([make_email(0, "phishing")])
        self.assertAlmostEqual(p.score, .75)
        self.assertAlmostEqual(p.details["answer_probability_mass"], .8)
    def test_test_label_never_enters_prompt(self):
        det = self.detector()
        a, = det.predict([make_email(0, "phishing")])
        prompt = det.backend.messages
        b, = det.predict([make_email(0, "legitimate")])
        self.assertEqual(det.backend.messages, prompt)
        self.assertEqual(a.score, b.score)
    def test_examples_balanced_deterministic_and_train_only(self):
        train = make_split(10, 10)
        chosen = select_examples(train, 4, 2)
        self.assertEqual(chosen, select_examples(list(reversed(train)), 4, 2))
        self.assertEqual(sum(e.is_phishing for e in chosen), 2)
        det = self.detector(n_shots=4)
        det.fit(train, make_split(2, 2, 100))
        with self.assertRaises(ValueError):
            det.predict([det.examples[0]])
        with self.assertRaises(ValueError):
            det.fit(train, train)
    def test_few_shot_requires_fit_and_enough_examples(self):
        det = self.detector(n_shots=4)
        with self.assertRaises(RuntimeError):
            det.predict([make_email(99, "phishing")])
        with self.assertRaises(ValueError):
            det.fit(make_split(1, 1), [])
    def test_truncation_preserves_task_and_prefix(self):
        det = self.detector(max_input_tokens=1024)
        from phishbench.types import Email
        p, = det.predict([Email("long", "phishing", "", "", "x"*10000)])
        self.assertFalse(p.failed)
        self.assertTrue(p.details["truncated"])
        self.assertLessEqual(p.details["input_tokens"], 1024)
        self.assertIn("Classify", det.backend.messages[0]["content"])
    def test_cache_invalidation_and_original_timing(self):
        with tempfile.TemporaryDirectory() as root:
            det = QwenDetector(backend=FakeBackend(), cache_dir=root)
            a, = det.predict([make_email(0, "phishing")])
            b, = det.predict([make_email(0, "phishing")])
            self.assertEqual(det.backend.calls, 1)
            self.assertEqual(a.latency_ms, b.latency_ms)
            self.assertTrue(b.details["cache_hit"])
            det.revision = "1"*40
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
import os
@unittest.skipUnless(os.environ.get('PHISHBENCH_QWEN_INTEGRATION') == '1',
                     'set PHISHBENCH_QWEN_INTEGRATION=1 for local model integration')
class QwenIntegrationTest(unittest.TestCase):
    def test_real_zero_and_few_shot(self):
        from phishbench.data import load_emails
        from pathlib import Path
        emails = load_emails(Path(__file__).resolve().parents[1] / 'data/benchmark_emails.json')
        det = QwenDetector(cache_dir=None)
        zero = det.predict(emails[:2])
        det.n_shots = 2
        det.fit(emails[2:], [])
        few = det.predict(emails[:2])
        for p in zero + few:
            self.assertFalse(p.failed, p.error)
            self.assertGreaterEqual(p.score, 0)
            self.assertLessEqual(p.score, 1)
            self.assertLessEqual(p.details['answer_probability_mass'], 1.00001)
