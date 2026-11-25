#!/usr/bin/env python3
"""Write the prompts for the LLM-written test set (synthetic-v1).

300 email specs (150 phishing, 150 legitimate) are drawn from a fixed grid and
split evenly between two writers that are not being benchmarked (Grok and
ChatGPT), 150 each. Each writer gets 6 prompts of 25 mixed specs; paste one
prompt into a fresh chat and save the reply as

    data/synthetic/raw/<writer>/batch_<nn>.jsonl

then run scripts/build_synthetic.py. Labels come from the spec, never from the
writer's output.

    python scripts/build_synthetic_prompts.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
SEED = 20261122
WRITERS = ("grok", "chatgpt")
BATCH = 25

# (category, what the email is) -> number of emails. 150 phishing, 150 legitimate.
PHISHING = {
    "credential theft": ("asks the reader to sign in, verify or reset an account through a link to a fake login page", 25),
    "CEO fraud / business email compromise": ("impersonates an executive, colleague or supplier to get a payment, gift cards, or changed bank details; usually no link", 25),
    "invoice fraud": ("a fake invoice, overdue notice or payment-details change from a supposed vendor", 25),
    "delivery scam": ("a fake parcel, customs fee or missed-delivery notice asking for a small payment or personal details", 25),
    "malicious attachment": ("pushes the reader to open an attached or linked document (invoice, scan, voicemail, shared file); name the attachment in the text", 25),
    "advance-fee fraud": ("promises money (inheritance, prize, grant, refund, job) in return for a fee or personal details", 25),
}
LEGITIMATE = {
    "genuine security notice": ("a real password reset, new-sign-in alert or MFA notice from the organisation's own domain; may contain urgency and a link", 15),
    "genuine invoice or payment reminder": ("a real invoice, receipt, renewal or overdue reminder from a real supplier relationship", 15),
    "urgent request from a manager": ("a real boss or colleague asking for something quickly (a file, a report, an approval, a call); pressure but nothing fraudulent", 15),
    "genuine delivery notification": ("a real shipping, tracking or missed-delivery email from a carrier or shop", 15),
    "HR or IT administrative notice": ("benefits enrolment, policy acknowledgement, mandatory training, system maintenance, device update", 15),
    "marketing or newsletter": ("a promotion, product update or newsletter the reader signed up for, with offers and deadlines", 15),
    "ordinary work thread": ("colleagues discussing a project, a document or a decision", 12),
    "personal email": ("friends or family: plans, news, a favour", 12),
    "meeting or scheduling": ("arranging, moving or following up on a meeting or event", 12),
    "customer support reply": ("a support agent answering a ticket the reader opened", 12),
    "university or school email": ("course announcements, deadlines, advising, club or department mail", 12),
}
SECTORS = ["a retail bank", "a university", "a software (SaaS) company", "a logistics company", "a hospital group",
           "an online retailer", "a city government office", "a small accounting firm", "a telecom provider",
           "a payroll / HR services company", "an airline", "a manufacturing company"]
TONES = ["formal", "casual", "terse", "friendly", "bureaucratic"]
LENGTHS = ["short (40-80 words)", "medium (100-180 words)", "long (220-350 words)"]
QUALITY = ["polished", "polished", "a few typos or awkward phrases", "written by a non-native English speaker"]
DIFFICULTY = ["obvious (several classic red flags)", "moderate (one or two red flags)",
              "subtle (well written, plausible, only the request or the sender domain gives it away)"]

PREAMBLE = """\
I'm a university student working on a course research project that benchmarks phishing-email DETECTORS \
(a rules baseline, TF-IDF + logistic regression, fine-tuned BERT-style classifiers and LLM classifiers). \
The public phishing corpora we use are 15-20 years old and are almost certainly in the training data of \
the models we test, so we need a small, fresh test set that no detector has seen. Your output is used only \
as labelled test data for those detectors; none of it is ever sent to anyone.

Please write the {n} test emails specified below. About half are phishing and half are legitimate; the \
legitimate ones matter just as much, because they measure false alarms.

Safety rules (these keep the set harmless):
- Every organisation, person, product and address must be fictional. Do not use real brands, real companies or real people.
- Every domain in a sender address or link must end in ".example" (a reserved top-level domain that can never resolve), for phishing and legitimate emails alike. Example: billing@northwind-freight.example, https://portal.northwind-freight.example/invoices
- No real phone numbers (use 555 numbers), no real account numbers, no working payloads. Attachments are only mentioned by file name.

Realism rules:
- Write each email the way that sender would really write it. Follow the spec's tone, length and writing quality.
- A phishing email should impersonate a fictional organisation the way real phishing does (for example a lookalike domain such as northwind-freight-billing.example for northwind-freight.example). Never label it, warn about it or add a disclaimer inside the email.
- A legitimate email must be genuinely legitimate: sender domain, links and request all consistent with the organisation.
- Vary names, organisations, greetings, sign-offs and structure between emails. Do not reuse an organisation name within this batch.
- No placeholders such as [Name] or <link>; fill everything in.

Output format: exactly {n} lines of JSON (JSONL) in one code block, nothing else, one object per email, in the order given:
{{"id": "<the spec id>", "from": "Display Name <address@domain.example>", "subject": "...", "body": "..."}}
Use \\n for line breaks inside "body".

Specs:
"""


def build_specs() -> list[dict]:
    rng = random.Random(SEED)
    specs = []
    for label, table in (("phishing", PHISHING), ("legitimate", LEGITIMATE)):
        for category, (what, n) in table.items():
            for _ in range(n):
                spec = {"label": label, "category": category, "what": what, "sector": rng.choice(SECTORS),
                        "tone": rng.choice(TONES), "length": rng.choice(LENGTHS), "quality": rng.choice(QUALITY)}
                if label == "phishing":
                    spec["difficulty"] = rng.choice(DIFFICULTY)
                specs.append(spec)
    # Deal each (label, category) evenly to the writers, then shuffle within a writer.
    specs.sort(key=lambda s: (s["label"], s["category"]))
    per_writer: dict[str, list[dict]] = {w: [] for w in WRITERS}
    for i, spec in enumerate(specs):
        per_writer[WRITERS[i % len(WRITERS)]].append(spec)
    out = []
    for writer, items in per_writer.items():
        rng.shuffle(items)
        for i, spec in enumerate(items):
            out.append({"id": f"syn-{writer}-{i + 1:03d}", "writer": writer, "batch": i // BATCH + 1, **spec})
    return out


def spec_line(s: dict) -> str:
    parts = [f"{s['label'].upper()}, {s['category']}: {s['what']}"]
    if s["category"] != "personal email":  # personal mail has no organisation behind it
        parts.append(f"organisation: {s['sector']}")
    parts += [f"tone: {s['tone']}", f"length: {s['length']}", f"writing: {s['quality']}"]
    if "difficulty" in s:
        parts.append(f"how detectable: {s['difficulty']}")
    return f"- {s['id']} | " + " | ".join(parts)


def main() -> int:
    specs = build_specs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "specs.json").write_text(json.dumps(specs, indent=1) + "\n", encoding="utf-8")
    for writer in WRITERS:
        (OUT / "prompts" / writer).mkdir(parents=True, exist_ok=True)
        (OUT / "raw" / writer).mkdir(parents=True, exist_ok=True)
        mine = [s for s in specs if s["writer"] == writer]
        for b in sorted({s["batch"] for s in mine}):
            batch = [s for s in mine if s["batch"] == b]
            text = PREAMBLE.format(n=len(batch)) + "\n".join(spec_line(s) for s in batch) + "\n"
            (OUT / "prompts" / writer / f"batch_{b:02d}.txt").write_text(text, encoding="utf-8")
        n_ph = sum(1 for s in mine if s["label"] == "phishing")
        print(f"{writer}: {len(mine)} specs ({n_ph} phishing, {len(mine) - n_ph} legitimate), "
              f"{len({s['batch'] for s in mine})} prompts in {OUT / 'prompts' / writer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
