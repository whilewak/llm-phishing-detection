import math
import unittest

from phishbench.eval import metrics as m
from phishbench.types import Prediction


class RecallAtFprTest(unittest.TestCase):
    def test_perfect_separation(self):
        y = [True, True, False, False]
        recall, t = m.recall_at_fpr(y, [0.9, 0.8, 0.2, 0.1], 0.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(t, 0.8)

    def test_stops_at_fpr_budget(self):
        # 100 legitimate emails, so 1% FPR allows exactly one false positive.
        y = [False] * 100 + [True] * 4
        scores = [0.99, 0.98] + [0.1] * 98 + [0.97, 0.95, 0.5, 0.05]
        recall, t = m.recall_at_fpr(y, scores, 0.01)
        # Threshold 0.99 flags one legit email (1%); 0.98 would be 2%.
        # Flagging the phishing scored 0.97 needs threshold <= 0.97, past 0.98.
        self.assertEqual(recall, 0.0)
        self.assertEqual(t, 0.99)

    def test_ties_move_together(self):
        y = [True, False]
        recall, t = m.recall_at_fpr(y, [0.5, 0.5], 0.0)
        self.assertEqual(recall, 0.0)
        self.assertTrue(math.isinf(t))

    def test_needs_both_classes(self):
        with self.assertRaises(ValueError):
            m.recall_at_fpr([True, True], [0.1, 0.2], 0.01)


class FailedPredictionsTest(unittest.TestCase):
    def test_failures_always_wrong(self):
        y = [True, False, True, False]
        preds = [
            Prediction(score=None, error="timeout"),
            Prediction(score=None, error="refused"),
            Prediction(score=0.9),
            Prediction(score=0.1),
        ]
        scores = m.effective_scores(y, preds)
        for t in (0.0, 0.5, 1.0):
            r = m.at_threshold(y, scores, t)
            self.assertEqual(r.fn, 1 if t <= 0.9 else 2)  # failed phishing is never flagged
            self.assertGreaterEqual(r.fp, 1)  # failed legitimate is always flagged

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            m.effective_scores([True], [])


class OtherMetricsTest(unittest.TestCase):
    def test_average_precision(self):
        self.assertAlmostEqual(m.average_precision([True, False], [0.9, 0.1]), 1.0)
        # Ranking: phish, legit, phish -> AP = 0.5*1 + 0.5*(2/3)
        self.assertAlmostEqual(m.average_precision([True, False, True], [0.9, 0.5, 0.1]), 0.5 + 1 / 3)

    def test_threshold_metrics(self):
        r = m.at_threshold([True, True, False, False], [0.9, 0.3, 0.6, 0.1], 0.5)
        self.assertEqual((r.tp, r.fn, r.fp, r.tn), (1, 1, 1, 1))
        self.assertAlmostEqual(r.f1, 0.5)
        # recall 0.5, fpr 0.5 at 1% prevalence: 0.005 / (0.005 + 0.495)
        self.assertAlmostEqual(r.precision_at_prevalence(0.01), 0.01)

    def test_ece(self):
        self.assertAlmostEqual(m.expected_calibration_error([True, False], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(m.expected_calibration_error([False, False], [0.95, 0.95]), 0.95)
        self.assertTrue(math.isnan(m.expected_calibration_error([], [])))

    def test_tie_share(self):
        self.assertEqual(m.tie_share([0.1, 0.2, 0.3]), 0.0)
        self.assertEqual(m.tie_share([0.5, 0.5, 0.3, 0.1]), 0.5)

    def test_percentile(self):
        values = list(range(1, 101))
        self.assertEqual(m.percentile(values, 50), 50)
        self.assertEqual(m.percentile(values, 95), 95)
        self.assertEqual(m.percentile([7.0], 95), 7.0)

    def test_bootstrap_is_deterministic_and_brackets(self):
        y = [True] * 20 + [False] * 20
        scores = [0.8 + i * 0.001 for i in range(20)] + [0.2 + i * 0.001 for i in range(20)]
        ci1 = m.bootstrap_ci(m.average_precision, y, scores, n_resamples=200)
        ci2 = m.bootstrap_ci(m.average_precision, y, scores, n_resamples=200)
        self.assertEqual(ci1, ci2)
        self.assertEqual(ci1, (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
