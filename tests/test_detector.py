"""
Unit tests for LayaPhishingDetector.
"""

import unittest
from src.detector import LayaPhishingDetector


class TestLayaPhishingDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize detector once for testing
        cls.detector = LayaPhishingDetector(default_model="english", risk_threshold=1.05)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.router)
        self.assertIn(self.detector.device, ["mps", "cuda", "cpu"])
        self.assertEqual(self.detector.risk_threshold, 1.05)

    def test_default_questions_structure(self):
        questions = self.detector.DEFAULT_QUESTIONS
        self.assertIn("is_phishing", questions)
        self.assertIn("urgency_pressure", questions)
        self.assertIn("credential_harvest", questions)
        self.assertIn("financial_pressure", questions)
        self.assertIn("classification", questions)

        for qid, qdef in questions.items():
            self.assertIn("type", qdef)
            self.assertIn("instructions", qdef)

    def test_credential_phish_prediction(self):
        sample = {
            "from": "security@micros0ft-login-verify.com",
            "subject": "Urgent: Verify Your Password",
            "body": "Your account is locked. Click here immediately to submit your password."
        }
        res = self.detector.predict(sample)
        
        self.assertIn("verdict", res)
        self.assertIn("composite_risk_score", res)
        self.assertIn("signals", res)
        self.assertEqual(res["verdict"], "PHISHING")
        self.assertGreaterEqual(res["composite_risk_score"], 1.05)
        self.assertGreater(res["signals"]["p_credential_harvest"], 0.4)

    def test_legitimate_email_prediction(self):
        sample = {
            "from": "newsletter@engineering-weekly.org",
            "subject": "This Week in Software Engineering #42",
            "body": "Here are the top articles on distributed systems and compiler design for this week."
        }
        res = self.detector.predict(sample)
        
        self.assertIn("verdict", res)
        self.assertEqual(res["verdict"], "BENIGN")
        self.assertLess(res["composite_risk_score"], 1.05)

    def test_evaluate_batch(self):
        samples = [
            {
                "id": "1",
                "ground_truth": "PHISHING",
                "from": "attacker@fake.com",
                "subject": "Immediate password reset",
                "body": "Reset password now or lose access."
            },
            {
                "id": "2",
                "ground_truth": "BENIGN",
                "from": "team@corp.com",
                "subject": "Lunch today",
                "body": "Who wants tacos for lunch?"
            }
        ]
        summary = self.detector.evaluate_batch(samples)
        
        self.assertEqual(summary["total_samples"], 2)
        self.assertIn("confusion_matrix", summary)
        self.assertIn("metrics", summary)
        self.assertIn("latency", summary)
        self.assertEqual(summary["metrics"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
