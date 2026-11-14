import json
import tempfile
import unittest
from pathlib import Path

from phishbench.eval import evaluate, write_result
from tests.stubs import OracleDetector, ScriptedDetector, ShortDetector, make_split


class HarnessTest(unittest.TestCase):
    def test_oracle_end_to_end(self):
        det = OracleDetector()
        train, val, test = make_split(5, 5), make_split(10, 10, start=100), make_split(10, 10, start=200)
        result = evaluate(det, test, train=train, val=val, warmup=2, n_bootstrap=50, config={"name": "oracle"})

        self.assertEqual(det.fit_calls, 1)
        self.assertEqual(result["model"], "oracle")
        self.assertEqual(result["threshold"]["value"], 0.9)
        self.assertTrue(result["threshold"]["chosen_on"].startswith("validation"))
        self.assertEqual(result["detection"]["headline_recall_at_fpr"]["recall"], 1.0)
        self.assertEqual(result["detection"]["pr_auc"], 1.0)
        self.assertEqual(result["errors"]["count"], 0)
        self.assertEqual(result["training_cost"]["n_train_emails"], 10)
        self.assertEqual(result["data"]["test"]["labels"], {"phishing": 10, "legitimate": 10})
        self.assertEqual(len(result["predictions"]), 20)
        self.assertAlmostEqual(result["inference_cost_usd"]["per_million_emails"], 1000.0)
        self.assertEqual(result["speed"]["latency_ms_p50"], 2.0)
        self.assertIn("python", result["environment"])

    def test_default_threshold_without_validation(self):
        result = evaluate(OracleDetector(), make_split(2, 2), warmup=0, n_bootstrap=10)
        self.assertEqual(result["threshold"]["value"], 0.5)
        self.assertTrue(result["threshold"]["chosen_on"].startswith("default"))

    def test_errors_counted_wrong(self):
        test = make_split(2, 2)  # e0, e1 phishing; e2, e3 legitimate
        det = ScriptedDetector({"e0": None, "e1": 0.9, "e2": None, "e3": 0.1})
        result = evaluate(det, test, warmup=0, n_bootstrap=10)
        self.assertEqual(result["errors"]["count"], 2)
        self.assertEqual(result["errors"]["examples"], ["refused"])
        at_t = result["detection"]["at_chosen_threshold"]
        self.assertEqual((at_t["tp"], at_t["fn"], at_t["fp"], at_t["tn"]), (1, 1, 1, 1))

    def test_score_ties_reported(self):
        test = make_split(2, 2)
        det = ScriptedDetector({"e0": 0.5, "e1": 0.5, "e2": 0.5, "e3": 0.1})
        result = evaluate(det, test, warmup=0, n_bootstrap=10)
        self.assertEqual(result["score_ties"]["tie_share"], 0.75)

    def test_single_class_test_skips_detection(self):
        result = evaluate(OracleDetector(), make_split(3, 0), warmup=0)
        self.assertIsNone(result["detection"])
        self.assertIn("detection_note", result)

    def test_breakdown_by_source_and_group(self):
        from phishbench.types import Email

        test = [
            Email(id="n1", label="phishing", sender="", subject="", body="", source="nazario"),
            Email(id="n2", label="phishing", sender="", subject="", body="", source="nazario"),
            Email(id="t1", label="phishing", sender="", subject="", body="", source="trec"),
            Email(id="t2", label="legitimate", sender="", subject="", body="", source="trec"),
        ]
        det = ScriptedDetector({"n1": 0.9, "n2": 0.2, "t1": 0.8, "t2": 0.1})
        result = evaluate(det, test, warmup=0, n_bootstrap=10, source_groups={"true_phishing": ["nazario"]})
        by = result["detection"]["by_source"]
        self.assertEqual(by["nazario"]["n_phishing"], 2)
        self.assertEqual(by["true_phishing"]["at_chosen_threshold"]["recall"], 0.5)
        self.assertIsNone(by["nazario"]["at_chosen_threshold"]["false_positive_rate"])
        self.assertEqual(by["trec"]["at_chosen_threshold"], {"recall": 1.0, "false_positive_rate": 0.0})

    def test_wrong_prediction_count_raises(self):
        with self.assertRaises(RuntimeError):
            evaluate(ShortDetector(), make_split(2, 2), warmup=0, batch_size=4)

    def test_write_result_is_strict_json(self):
        result = evaluate(OracleDetector(), make_split(3, 3), warmup=0, n_bootstrap=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(result, tmp)
            loaded = json.loads(Path(path).read_text())
        self.assertEqual(loaded["model"], "oracle")


if __name__ == "__main__":
    unittest.main()
