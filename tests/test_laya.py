import importlib.util
import unittest
from pathlib import Path

from phishbench.data import load_emails
from phishbench.detectors.laya import POC_WEIGHTS, LayaDetector
from tests.stubs import make_email

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "benchmark_emails.json"
HAS_LAYA = importlib.util.find_spec("laya") is not None and importlib.util.find_spec("torch") is not None


def _answers(p: float, choice: str = "benign") -> dict:
    out = {q: {"noul": p} for q in POC_WEIGHTS}
    out["classification"] = {"choice": choice, "probabilities": {choice: 1.0}}
    return {"answers": out}


class FakeRouter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def predict(self, state, questions):
        self.calls.append(state)
        return self.response


class LayaDetectorTest(unittest.TestCase):
    def test_score_is_scaled_composite(self):
        det = LayaDetector(router=FakeRouter(_answers(1.0)))
        (pred,) = det.predict([make_email(0, "phishing")])
        self.assertIsNone(pred.error)
        self.assertAlmostEqual(pred.score, 1.0)
        self.assertEqual(pred.label, "phishing")
        self.assertEqual(pred.details["classification"], "benign")

    def test_low_signals_are_legitimate(self):
        det = LayaDetector(router=FakeRouter(_answers(0.05)))
        (pred,) = det.predict([make_email(0, "legitimate")])
        self.assertAlmostEqual(pred.score, 0.05)
        self.assertEqual(pred.label, "legitimate")

    def test_classification_head_can_flag(self):
        det = LayaDetector(router=FakeRouter(_answers(0.05, choice="phishing")))
        (pred,) = det.predict([make_email(0, "phishing")])
        self.assertEqual(pred.label, "phishing")

    def test_missing_answers_are_errors_not_zeros(self):
        response = _answers(0.9)
        del response["answers"]["credential_harvest"]
        det = LayaDetector(router=FakeRouter(response))
        (pred,) = det.predict([make_email(0, "phishing")])
        self.assertIsNone(pred.score)
        self.assertIn("credential_harvest.noul", pred.error)

    def test_wrong_answer_key_is_error(self):
        det = LayaDetector(router=FakeRouter({"answers": {q: {"prob": 0.9} for q in POC_WEIGHTS}}))
        (pred,) = det.predict([make_email(0, "phishing")])
        self.assertTrue(pred.failed)
        self.assertIn("classification.choice", pred.error)

    def test_malformed_response_is_error(self):
        det = LayaDetector(router=FakeRouter(None))
        (pred,) = det.predict([make_email(0, "phishing")])
        self.assertTrue(pred.failed)

    def test_router_receives_email_fields(self):
        router = FakeRouter(_answers(0.5))
        LayaDetector(router=router).predict([make_email(3, "phishing")])
        self.assertEqual(router.calls[0], {"from": "s3@example.com", "subject": "subject 3", "body": "body 3"})


@unittest.skipUnless(HAS_LAYA, "laya and torch not installed")
class LayaIntegrationTest(unittest.TestCase):
    def test_sample_emails_run_without_errors(self):
        det = LayaDetector()
        preds = det.predict(load_emails(SAMPLE))
        self.assertEqual(len(preds), 7)
        self.assertEqual([p.error for p in preds], [None] * 7)
        for p in preds:
            self.assertGreaterEqual(p.score, 0.0)
            self.assertLessEqual(p.score, 1.0)


if __name__ == "__main__":
    unittest.main()
