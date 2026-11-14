"""Load emails from JSON or JSONL into the shared schema."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from phishbench.types import LEGITIMATE, PHISHING, SPAM, Email

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

# Label spellings seen in source files, mapped to the shared labels.
_LABEL_ALIASES = {
    "phishing": PHISHING,
    "phish": PHISHING,
    "legitimate": LEGITIMATE,
    "benign": LEGITIMATE,
    "ham": LEGITIMATE,
    "safe": LEGITIMATE,
    "spam": SPAM,
    "promotional": SPAM,
}


def normalize_label(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in ("", "unknown"):
        return None
    if key not in _LABEL_ALIASES:
        raise ValueError(f"unknown label {raw!r}; expected one of {sorted(_LABEL_ALIASES)}")
    return _LABEL_ALIASES[key]


def extract_urls(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(m.rstrip(".,;:") for m in _URL_RE.findall(text)))


def _to_email(record: dict[str, Any], index: int) -> Email:
    body = record.get("body") or ""
    urls = record.get("urls")
    return Email(
        id=str(record.get("id", index)),
        label=normalize_label(record.get("label", record.get("ground_truth"))),
        sender=record.get("from") or record.get("sender") or "",
        subject=record.get("subject") or "",
        body=body,
        urls=tuple(urls) if urls is not None else extract_urls(body),
        source=record.get("source"),
    )


def load_emails(path: str | Path) -> list[Email]:
    """Read a JSON list or a JSONL file of email records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    emails = [_to_email(r, i) for i, r in enumerate(records)]
    ids = [e.id for e in emails]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate email ids in {path}")
    return emails


def dataset_fingerprint(emails: Iterable[Email]) -> str:
    """SHA-256 over ids, labels and text, independent of file order or format."""
    h = hashlib.sha256()
    for e in sorted(emails, key=lambda e: e.id):
        for part in (e.id, e.label or "", e.sender, e.subject, e.body):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        h.update(b"\x1e")
    return h.hexdigest()
