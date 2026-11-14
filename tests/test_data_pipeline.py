import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from phishbench.data.clean import DropCounts, clean_body, read_source
from phishbench.data.splits import freeze, load_split, stratified_sample, stratified_split
from phishbench.types import LEGITIMATE, PHISHING, Email

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def email(i: int, label: str, source: str = "s", body: str | None = None) -> Email:
    # Emails given an explicit body share a subject, so only the body decides duplication.
    return Email(id=f"{source}:{i}", label=label, sender="", subject="hello" if body is not None else f"subject {i}",
                 body=body if body is not None else f"unique body number {i} " + "word " * (i % 7), source=source)


class CleanTest(unittest.TestCase):
    def test_header_lines_removed(self):
        body = "Received: from x by y\nMessage-ID: <a@b>\nHello there,\nplease read.\nX-Spam-Score: 5"
        cleaned, truncated = clean_body(body)
        self.assertEqual(cleaned, "Hello there,\nplease read.")
        self.assertFalse(truncated)

    def test_flattened_message_kept(self):
        body = "Date:   Wed, 21 Aug 2002 10:54:46 -0500  From: Chris  Message-ID:  <1@x>  | " + "real text " * 40
        cleaned, _ = clean_body(body)
        self.assertIn("real text", cleaned)
        self.assertNotIn("2002", cleaned)
        self.assertNotIn("<1@x>", cleaned)

    def test_tokenised_header_removed(self):
        cleaned, _ = clean_body("content - length : 3386 apple research center")
        self.assertEqual(cleaned, "apple research center")

    def test_truncation(self):
        cleaned, truncated = clean_body("a" * 200_000)
        self.assertTrue(truncated)
        self.assertEqual(len(cleaned), 100_000)

    def test_read_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Nazario.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sender", "receiver", "date", "subject", "body", "urls", "label"])
                w.writerow(["a@x", "", "", "Verify account", "Click http://evil.example/login now", "1", "1"])
                w.writerow(["MAILER", "", "", "DON'T DELETE -- FOLDER INTERNAL DATA", "x", "0", "1"])
                w.writerow(["b@x", "", "", "", "", "0", "1"])
                w.writerow(["c@x", "", "", "hi", "body", "0", ""])
            drops = DropCounts()
            rows = [e for e, _ in read_source(path, drops)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, "nazario:0")
        self.assertEqual(rows[0].label, PHISHING)
        self.assertEqual(rows[0].urls, ("http://evil.example/login",))
        self.assertEqual((drops.mbox_artifact, drops.empty, drops.bad_label), (1, 1, 1))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class DedupTest(unittest.TestCase):
    def test_exact_near_and_mixed(self):
        from phishbench.data.dedup import deduplicate

        long = "please verify your account details at the link below within one day " * 5
        emails = [
            email(0, PHISHING, body=long),
            email(1, PHISHING, body=long.upper() + " !!!"),  # exact after normalising tokens
            email(2, PHISHING, body=long + " thanks"),  # near duplicate
            email(3, LEGITIMATE, body="the quarterly planning meeting moved to thursday " * 5),
            email(4, PHISHING, body="the quarterly planning meeting moved to thursday " * 5),  # mixed labels
            email(5, LEGITIMATE, body="lunch on friday? tacos or pizza, let me know by noon " * 3),
        ]
        kept, report = deduplicate(emails)
        self.assertEqual([e.id for e in kept], ["s:0", "s:5"])
        self.assertEqual(report.exact_duplicates, 2)  # 1 copies 0; 4 copies 3
        self.assertEqual(report.near_duplicates, 1)  # 2 is near 0
        self.assertEqual(report.mixed_label_groups, 1)
        self.assertEqual(report.mixed_label_emails, 2)

    def test_tokenised_and_raw_copies_match(self):
        from phishbench.data.dedup import deduplicate

        raw = "Hi team, the report for May 25, 2001 is attached (hplno525.xls). Thanks!"
        tokenised = "hi team , the report for may 25 , 2001 is attached ( hplno525 . xls ) . thanks !"
        kept, _ = deduplicate([email(0, LEGITIMATE, "trec05", raw), email(1, LEGITIMATE, "enron", tokenised)])
        self.assertEqual(len(kept), 1)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.emails = [email(i, PHISHING if i % 3 == 0 else LEGITIMATE, source="a" if i % 2 else "b")
                       for i in range(400)]

    def test_split_is_stratified_disjoint_and_deterministic(self):
        s1 = stratified_split(self.emails, seed=7)
        s2 = stratified_split(self.emails, seed=7)
        self.assertEqual({k: [e.id for e in v] for k, v in s1.items()}, {k: [e.id for e in v] for k, v in s2.items()})
        ids = [e.id for v in s1.values() for e in v]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 400)
        self.assertEqual(len(s1["train"]), 280)
        phish_share = sum(e.is_phishing for e in s1["test"]) / len(s1["test"])
        self.assertAlmostEqual(phish_share, sum(e.is_phishing for e in self.emails) / 400, delta=0.02)

    def test_sample_size_and_proportions(self):
        sample = stratified_sample(self.emails, 100, seed=1)
        self.assertEqual(len(sample), 100)
        self.assertEqual(len({e.id for e in sample}), 100)
        self.assertAlmostEqual(sum(e.is_phishing for e in sample), 100 * 134 / 400, delta=1)

    def test_freeze_and_load_roundtrip(self):
        splits = stratified_split(self.emails, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed" / "x.jsonl"
            processed.parent.mkdir(parents=True)
            processed.write_text("\n".join(json.dumps(
                {"id": e.id, "label": e.label, "source": e.source, "from": e.sender, "subject": e.subject,
                 "body": e.body}) for e in self.emails))
            split_dir = root / "data" / "splits" / "x-v1"
            freeze(splits, split_dir, {"processed_file": "data/processed/x.jsonl"})

            test = load_split(split_dir, "test")
            self.assertEqual(sorted(e.id for e in test), sorted(e.id for e in splits["test"]))

            with self.assertRaises(FileExistsError):
                freeze(splits, split_dir, {})

            # Changing one email's text must be caught.
            changed = processed.read_text().replace("unique body number", "edited body number", 1)
            processed.write_text(changed)
            first_id = json.loads(changed.splitlines()[0])["id"]
            name = next(n for n, v in splits.items() if any(e.id == first_id for e in v))
            with self.assertRaises(ValueError):
                load_split(split_dir, name)


if __name__ == "__main__":
    unittest.main()
