import json
import tempfile
import unittest
from pathlib import Path

from phishbench import prompts
from phishbench.types import Email
from tests.stubs import make_email

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits" / "champa2023-v1"
POOL = ROOT / "configs" / "prompts" / "fewshot-champa2023-v1.json"


def _write_pool(path: Path, emails, train_hash="h"):
    path.write_text(json.dumps({
        "train_ids_sha256": train_hash,
        "examples_sha256": prompts.fewshot_hash(emails),
        "examples": [{"id": e.id, "label": e.label, "source": e.source, "from": e.sender,
                      "subject": e.subject, "body": e.body, "urls": list(e.urls)} for e in emails],
    }))


class RenderTest(unittest.TestCase):
    def test_includes_every_field(self):
        e = Email(id="x", label="phishing", sender="a@b.c", subject="Hi", body="Click", urls=("http://u.v",))
        text, truncated = prompts.render_email(e)
        for part in ("<email>", "From: a@b.c", "Subject: Hi", "http://u.v", "Click", "</email>"):
            self.assertIn(part, text)
        self.assertFalse(truncated)

    def test_truncates_long_bodies(self):
        e = Email(id="x", label=None, sender="", subject="", body="word " * 10_000)
        text, truncated = prompts.render_email(e, max_chars=1000)
        self.assertTrue(truncated)
        self.assertIn("[... truncated]", text)
        self.assertLess(len(text), 1200)

    def test_untrusted_notice_and_modes(self):
        for mode in prompts.ANSWER_MODES:
            s = prompts.system_prompt(mode)
            self.assertIn("untrusted data", s)
        self.assertIn("risk_score", prompts.system_prompt(prompts.SCORE_0_100))
        with self.assertRaises(ValueError):
            prompts.system_prompt("nope")

    def test_few_shot_examples_carry_gold_answers(self):
        shots = [make_email(1, "phishing"), make_email(2, "legitimate")]
        s = prompts.system_prompt(prompts.ANSWER_TOKEN, shots)
        self.assertIn("Example 1", s)
        self.assertIn("Answer: phishing", s)
        self.assertIn("Answer: legitimate", s)


class FewShotPoolTest(unittest.TestCase):
    def test_roundtrip_and_tamper_detection(self):
        shots = [make_email(1, "phishing"), make_email(2, "legitimate")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pool.json"
            _write_pool(path, shots)
            self.assertEqual([e.id for e in prompts.load_fewshot(path)], ["e1", "e2"])
            data = json.loads(path.read_text())
            data["examples"][0]["body"] = "changed"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                prompts.load_fewshot(path)

    def test_rejects_examples_outside_train(self):
        manifest = json.loads((SPLITS / "manifest.json").read_text())
        shots = [make_email(1, "phishing")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pool.json"
            _write_pool(path, shots, manifest["splits"]["train"]["ids_sha256"])
            with self.assertRaisesRegex(ValueError, "not in train"):
                prompts.load_fewshot(path, SPLITS)

    @unittest.skipUnless(POOL.exists(), "few-shot pool not built yet (scripts/build_fewshot.py)")
    def test_frozen_pool(self):
        shots = prompts.load_fewshot(POOL, SPLITS)
        self.assertEqual(len(shots), 8)
        self.assertEqual(sum(e.label == "phishing" for e in shots), 4)


class BuildFewShotTest(unittest.TestCase):
    def test_choose_is_deterministic_and_balanced(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("build_fewshot", ROOT / "scripts" / "build_fewshot.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        train = []
        sources = {"phishing": ["nazario", "nigerian", "ceas08", "trec05", "trec06"],
                   "legitimate": ["enron", "ling", "spamassassin", "trec07", "trec06"]}
        for label, srcs in sources.items():
            for s in srcs:
                for i in range(3):
                    train.append(Email(id=f"{s}-{label}-{i}", label=label, sender="x", subject="s",
                                       body="b" * 300, source=s))
        a, b = mod.choose(train), mod.choose(train)
        self.assertEqual([e.id for e in a], [e.id for e in b])
        self.assertEqual([e.label for e in a], ["phishing", "legitimate"] * 4)
        self.assertEqual(len({e.id for e in a}), 8)


if __name__ == "__main__":
    unittest.main()
