"""Read the curated source CSVs into the shared schema and clean them.

Cleaning removes things that identify the source corpus rather than the email:
header lines and message IDs pasted into bodies, RFC 2822 dates, and mailbox
artifacts. It does not change wording otherwise.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from phishbench.data.loader import extract_urls
from phishbench.types import LEGITIMATE, PHISHING, Email

MAX_BODY_CHARS = 100_000

# Source file -> short source name used in ids and reports. Order matters:
# deduplication keeps the copy read first, so the lowercased, tokenised corpora
# (Enron, Ling) come last and raw copies of the same email (TREC-05) win.
SOURCES = {
    "CEAS_08.csv": "ceas08",
    "Nazario.csv": "nazario",
    "Nigerian_Fraud.csv": "nigerian",
    "SpamAssasin.csv": "spamassassin",
    "TREC_05.csv": "trec05",
    "TREC_06.csv": "trec06",
    "TREC_07.csv": "trec07",
    "Enron.csv": "enron",
    "Ling.csv": "ling",
}
# Corpora made only of phishing / fraud, used for the "true phishing" breakdown.
TRUE_PHISHING_SOURCES = ("nazario", "nigerian")

_HEADER_NAMES = (
    r"message-id|date|from|to|cc|bcc|subject|received|return-path|reply-to|sender|"
    r"in-reply-to|references|mime-version|content-type|content-transfer-encoding|"
    r"content-length|x-[a-z0-9-]+"
)
# A header line at the start of a body line: "Received: from ...". Only short
# lines are removed: some corpora flatten a whole message onto one line that
# happens to start with "Date:".
_HEADER_LINE = re.compile(rf"^[ \t]*(?:{_HEADER_NAMES})[ \t]*:[^\n]{{0,200}}$", re.IGNORECASE | re.MULTILINE)
# Header fields pasted inline (SpamAssassin): "Message-ID:  <...>".
_INLINE_MESSAGE_ID = re.compile(r"message-id\s*:\s*<[^>]*>", re.IGNORECASE)
# Ling/Enron are tokenised: "content - length : 3386".
_TOKENISED_HEADER = re.compile(r"\bcontent\s*-\s*(?:length|type)\s*:\s*\S+", re.IGNORECASE)
_RFC2822_DATE = re.compile(
    r"\b(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+"
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[+-]\d{4})?(?:\s*\([A-Z]{2,5}\))?",
    re.IGNORECASE,
)
_MBOX_ARTIFACT = "FOLDER INTERNAL DATA"
_BLANK_RUNS = re.compile(r"\n{3,}")


@dataclass
class DropCounts:
    mbox_artifact: int = 0
    empty: int = 0
    bad_label: int = 0


def clean_body(body: str) -> tuple[str, bool]:
    """Return the cleaned body and whether it had to be truncated."""
    text = _HEADER_LINE.sub("", body)
    text = _INLINE_MESSAGE_ID.sub("", text)
    text = _TOKENISED_HEADER.sub("", text)
    text = _RFC2822_DATE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text).strip()
    if len(text) > MAX_BODY_CHARS:
        return text[:MAX_BODY_CHARS], True
    return text, False


def read_source(path: Path, drops: DropCounts) -> Iterator[tuple[Email, bool]]:
    """Yield (email, truncated) for one source CSV. Row numbers make the ids."""
    source = SOURCES[path.name]
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row_num, row in enumerate(csv.DictReader(f)):
            label_raw = (row.get("label") or "").strip()
            if label_raw not in ("0", "1"):
                drops.bad_label += 1
                continue
            subject = (row.get("subject") or "").strip()
            if _MBOX_ARTIFACT in subject:
                drops.mbox_artifact += 1
                continue
            body, truncated = clean_body(row.get("body") or "")
            if not body and not subject:
                drops.empty += 1
                continue
            yield (
                Email(
                    id=f"{source}:{row_num}",
                    label=PHISHING if label_raw == "1" else LEGITIMATE,
                    sender=(row.get("sender") or "").strip(),
                    subject=subject,
                    body=body,
                    urls=extract_urls(body),
                    source=source,
                ),
                truncated,
            )
