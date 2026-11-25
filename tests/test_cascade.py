import json
import tempfile
import unittest
from pathlib import Path

from phishbench.detectors.cascade import CascadeDetector, decide
from phishbench.detectors.replay import ReplayDetector
from phishbench.eval import evaluate
from phishbench.types import LEGITIMATE, PHISHING, Prediction, TrainingCost
from tests.stubs import ScriptedDetector, make_email, make_split


class Counting(ScriptedDetector):
    """Scripted scores with a fixed latency and cost, recording what it was asked."""

    def __init__(self, name, scores, latency_ms, cost_usd):
        super().__init__(scores)
        self.name, self.latency_ms, self.cost_usd, self.seen = name, latency_ms, cost_usd, []

    def predict(self, emails):
        self.seen += [e.id for e in emails]
        return [Prediction(score=self.scores[e.id], error=None if self.scores[e.id] is not None else "refused",
                           latency_ms=self.latency_ms, cost_usd=self.cost_usd) for e in emails]


class CascadeTest(unittest.TestCase):
    def setUp(self):
        self.emails = [make_email(i, PHISHING if i < 3 else LEGITIMATE) for i in range(6)]
        # e0 sure phishing, e5 sure legitimate, e1/e4 unsure, e2 tier 1 fails, e3 unsure twice
        self.t1 = Counting("cheap", {"e0": 0.99, "e1": 0.5, "e2": None, "e3": 0.4, "e4": 0.6, "e5": 0.01}, 1.0, 0.0)
        self.t2 = Counting("mid", {"e1": 0.95, "e2": 0.5, "e3": 0.5, "e4": 0.02}, 10.0, 0.0)
        self.t3 = Counting("top", {"e2": 0.9, "e3": None}, 100.0, 0.01)
        self.cascade = CascadeDetector([
            {"detector": self.t1, "low": 0.1, "high": 0.9},
            {"detector": self.t2, "low": 0.1, "high": 0.9},
            {"detector": self.t3},
        ])

    def test_decide(self):
        self.assertEqual(decide(0.9, 0.1, 0.9), "phishing")
        self.assertEqual(decide(0.1, 0.1, 0.9), "legitimate")
        self.assertEqual(decide(0.5, 0.1, 0.9), "escalate")
        self.assertEqual(decide(None, 0.1, 0.9), "escalate")

    def test_routing_scores_and_costs(self):
        preds = self.cascade.predict(self.emails)
        self.assertEqual([p.details["exit_tier"] for p in preds], [1, 2, 3, 3, 2, 1])
        expect = [(2 + 0.99) / 3, (2 + 0.95) / 3, (1 + 0.9) / 3, None, 0.02 / 3, 0.01 / 3]
        for p, want in zip(preds, expect):
            self.assertEqual(p.score, want) if want is None else self.assertAlmostEqual(p.score, want)
        self.assertEqual(self.t2.seen, ["e1", "e2", "e3", "e4"])
        self.assertEqual(self.t3.seen, ["e2", "e3"])
        self.assertEqual([p.latency_ms for p in preds], [1.0, 11.0, 111.0, 111.0, 11.0, 1.0])
        self.assertAlmostEqual(sum(p.cost_usd for p in preds), 0.02)
        # An early tier's failure escalates; the last tier's failure stays a failure.
        self.assertFalse(preds[2].failed)
        self.assertTrue(preds[3].failed)

    def test_runs_through_the_harness(self):
        result = evaluate(self.cascade, self.emails, val=self.emails, warmup=0, n_bootstrap=20)
        self.assertEqual(result["errors"]["count"], 1)
        self.assertAlmostEqual(result["inference_cost_usd"]["total"], 0.02)

    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            CascadeDetector([{"detector": self.t1}])
        with self.assertRaises(ValueError):
            CascadeDetector([{"detector": self.t1, "low": 0.9, "high": 0.1}, {"detector": self.t3}])


class ReplayTest(unittest.TestCase):
    def test_replays_saved_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m__val.json"
            path.write_text(json.dumps({"model": "m", "predictions": [
                {"id": "e0", "score": 0.7, "latency_ms": 3.0, "cost_usd": 0.1, "error": None},
                {"id": "e1", "score": None, "latency_ms": 4.0, "cost_usd": 0.0, "error": "refusal"}]}))
            det = ReplayDetector("m", [path])
            self.assertEqual(det.fit([], []), TrainingCost())
            a, b = det.predict([make_email(0, PHISHING), make_email(1, LEGITIMATE)])
            self.assertEqual((a.score, a.latency_ms, a.cost_usd), (0.7, 3.0, 0.1))
            self.assertTrue(b.failed)
            with self.assertRaises(KeyError):
                det.predict([make_email(9, PHISHING)])
            with self.assertRaises(ValueError):
                ReplayDetector("other", [path])


try:
    import numpy as np
except ImportError:  # the tuner is the only part that needs numpy
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class TuningTest(unittest.TestCase):
    def test_tune_respects_limits_and_matches_apply(self):
        from phishbench.cascade_tuning import apply, tune

        rng = np.random.default_rng(0)
        y = np.array([e.is_phishing for e in make_split(300, 700)])
        cheap = np.clip(np.where(y, 0.7, 0.3) + rng.normal(0, 0.25, len(y)), 0, 1)
        cheap[:5] = np.nan  # failures escalate
        top = np.clip(np.where(y, 0.95, 0.05) + rng.normal(0, 0.05, len(y)), 0, 1)
        out = tune([cheap], top, y, max_fpr=0.01, share_caps=(0.3, 1.0), grid=15)
        for cap, o in out.items():
            self.assertLessEqual(o.fpr, 0.01 + 1e-9)
            self.assertLessEqual(o.exit_share[-1], cap + 1e-9)
            again = apply([cheap], top, y, o.thresholds, o.last_threshold)
            self.assertAlmostEqual(again.recall, o.recall)
            self.assertAlmostEqual(again.fpr, o.fpr)
            self.assertAlmostEqual(sum(o.exit_share), 1.0)
        self.assertGreaterEqual(out[1.0].recall, out[0.3].recall)
        self.assertGreater(out[0.3].recall, 0.5)


if __name__ == "__main__":
    unittest.main()
