import json
import tempfile
import unittest
from pathlib import Path

from phishbench.data import dataset_fingerprint, load_emails
from phishbench.data.loader import extract_urls, normalize_label

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "benchmark_emails.json"


class LoaderTest(unittest.TestCase):
    def test_sample_file(self):
        emails = load_emails(SAMPLE)
        self.assertEqual(len(emails), 7)
        self.assertEqual(sum(e.is_phishing for e in emails), 4)
        first = emails[0]
        self.assertEqual(first.id, "CASE_1_CREDENTIAL_HARVEST")
        self.assertEqual(first.label, "phishing")
        self.assertEqual(first.urls, ("http://login.micros0ft-portal-verify.com/auth",))
        self.assertEqual(emails[1].label, "legitimate")

    def test_jsonl_and_duplicates(self):
        rows = [{"id": "a", "label": "ham", "body": "hi"}, {"id": "b", "label": "phish", "body": "x"}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows))
            self.assertEqual([e.label for e in load_emails(p)], ["legitimate", "phishing"])
            p.write_text("\n".join(json.dumps(r) for r in rows + rows[:1]))
            with self.assertRaises(ValueError):
                load_emails(p)

    def test_labels(self):
        self.assertEqual(normalize_label("BENIGN"), "legitimate")
        self.assertEqual(normalize_label("Promotional"), "spam")
        self.assertIsNone(normalize_label("UNKNOWN"))
        with self.assertRaises(ValueError):
            normalize_label("maybe")

    def test_extract_urls(self):
        self.assertEqual(extract_urls("see https://a.com/x. and http://b.org)"), ("https://a.com/x", "http://b.org"))

    def test_fingerprint_ignores_order(self):
        emails = load_emails(SAMPLE)
        self.assertEqual(dataset_fingerprint(emails), dataset_fingerprint(list(reversed(emails))))
        self.assertNotEqual(dataset_fingerprint(emails), dataset_fingerprint(emails[1:]))


if __name__ == "__main__":
    unittest.main()
