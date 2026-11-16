import unittest
from phishbench.detectors.rules import RulesDetector
from phishbench.types import Email


class RulesDetectorTest(unittest.TestCase):
    def setUp(self):
        self.detector = RulesDetector()

    def test_phishing_email_scores_high(self):
        email = Email(
            id="phish-1",
            label="phishing",
            sender="security-alert@random-server.xyz",
            subject="URGENT: Unauthorized access detected. Verify your account immediately",
            body="Your account will be suspended within 24 hours. Please click below to reset your password and verify your credentials.",
            urls=("http://192.168.1.1/login.php", "https://auth.verify-service.xyz/reset"),
        )
        preds = self.detector.predict([email])
        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertFalse(pred.failed)
        self.assertGreater(pred.score, 0.85)
        self.assertIn("urgency_hits:3", pred.details["triggered"])
        self.assertIn("url_ip_host", pred.details["triggered"])

    def test_legitimate_email_scores_low(self):
        email = Email(
            id="ham-1",
            label="legitimate",
            sender="colleague@company.com",
            subject="Lunch meeting tomorrow",
            body="Hey, let's meet at noon tomorrow to discuss the quarterly project roadmap. Let me know if that works.",
            urls=(),
        )
        preds = self.detector.predict([email])
        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertFalse(pred.failed)
        self.assertLess(pred.score, 0.25)
        self.assertEqual(len(pred.details["triggered"]), 0)

    def test_fit_returns_training_cost(self):
        cost = self.detector.fit([], [])
        self.assertEqual(cost.n_train_emails, 0)
        self.assertEqual(cost.cost_usd, 0.0)

    def test_empty_email_handled(self):
        email = Email(id="empty", label=None, sender="", subject="", body="", urls=())
        preds = self.detector.predict([email])
        self.assertEqual(len(preds), 1)
        self.assertFalse(preds[0].failed)
        self.assertLess(preds[0].score, 0.2)


if __name__ == "__main__":
    unittest.main()

