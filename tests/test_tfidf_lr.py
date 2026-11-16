import unittest
from phishbench.detectors.tfidf_lr import TfidfLogisticRegressionDetector
from phishbench.types import Email


class TfidfLogisticRegressionTest(unittest.TestCase):
    def test_unfitted_returns_error(self):
        det = TfidfLogisticRegressionDetector()
        email = Email(id="1", label=None, sender="", subject="test", body="hello", urls=())
        preds = det.predict([email])
        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0].failed)
        self.assertIn("not fitted", preds[0].error.lower())

    def test_fit_and_predict_flow(self):
        train = [
            Email(id="p1", label="phishing", sender="", subject="URGENT: account suspended", body="verify password now", urls=("http://bad.xyz",)),
            Email(id="p2", label="phishing", sender="", subject="Wire transfer invoice overdue", body="pay invoice immediately to bank account", urls=()),
            Email(id="h1", label="legitimate", sender="", subject="Meeting notes from lunch", body="great discussion on the quarterly roadmap", urls=()),
            Email(id="h2", label="legitimate", sender="", subject="Weekend plans and dinner", body="let us grab dinner on Saturday evening", urls=()),
        ]
        val = train[:2]
        det = TfidfLogisticRegressionDetector(max_features=500)
        cost = det.fit(train, val)

        self.assertEqual(cost.n_train_emails, 4)
        self.assertGreater(cost.wall_clock_s, 0.0)

        test = [
            Email(id="tp", label="phishing", sender="", subject="Immediate action: your account is suspended", body="verify password", urls=()),
            Email(id="th", label="legitimate", sender="", subject="Dinner roadmap discussion", body="weekend plans and dinner", urls=()),
        ]
        preds = det.predict(test)
        self.assertEqual(len(preds), 2)
        self.assertFalse(preds[0].failed)
        self.assertFalse(preds[1].failed)
        self.assertGreater(preds[0].score, preds[1].score)
        self.assertTrue(0.0 <= preds[0].score <= 1.0)
        self.assertTrue(0.0 <= preds[1].score <= 1.0)


if __name__ == "__main__":
    unittest.main()

