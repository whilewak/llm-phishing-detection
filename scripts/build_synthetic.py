#!/usr/bin/env python3
"""Check the writers' output and freeze the LLM-written test set.

Reads every file in data/synthetic/raw/<writer>/ (the replies to the prompts
from scripts/build_synthetic_prompts.py, pasted as-is), matches every email to
its spec, and writes

    data/synthetic/synthetic-v1.jsonl      the emails (committed)
    data/splits/synthetic-v1/              frozen split set with one split, `synthetic`

data/synthetic/replacements.json lists emails that were replaced after review
(new id -> the id it replaces); the replaced originals are left out. The
original 300 are frozen separately as `synthetic-old` (`--original` rebuilds
that set).

Labels come from the spec. `source` is the writer, so results break down per
writer. Nothing is ever tuned on this set.

    python scripts/build_synthetic.py --check     # report problems only
    python scripts/build_synthetic.py             # freeze (refuses if specs are missing)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from phishbench.data.loader import _to_email, extract_urls
from phishbench.data.splits import freeze

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
_DOMAIN_RE = re.compile(r"@([A-Za-z0-9.-]+)|https?://([A-Za-z0-9.-]+)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[(?:name|link|company|date|your [a-z ]+)\]|<link>|lorem ipsum", re.IGNORECASE)
_DISCLAIMER_RE = re.compile(r"\b(this is a (simulated|sample|fictional|test) |for (research|training|educational) purposes)", re.IGNORECASE)


def parse_reply(path: Path) -> tuple[list[dict], list[str]]:
    """Every JSON object in a pasted reply; tolerates code fences, stray prose and
    replies pasted back to back with no newline between them."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records, problems, pos = [], [], 0
    while (start := text.find("{", pos)) != -1:
        try:
            obj, pos = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: line {text.count(chr(10), 0, start) + 1}: not valid JSON ({exc.msg})")
            pos = start + 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records, problems


def check_email(rec: dict, spec: dict) -> list[str]:
    issues = []
    body = rec.get("body") or ""
    if len(body.split()) < 15:
        issues.append("body shorter than 15 words")
    if not (rec.get("from") or "").strip() or not (rec.get("subject") or "").strip():
        issues.append("missing sender or subject")
    text = f"{rec.get('from', '')}\n{body}"
    bad = sorted({(a or b).lower().rstrip(".") for a, b in _DOMAIN_RE.findall(text)
                  if not (a or b).lower().rstrip(".").endswith(".example")})
    if bad:
        issues.append(f"domains outside .example: {', '.join(bad[:3])}")
    if _PLACEHOLDER_RE.search(body):
        issues.append("unfilled placeholder")
    if _DISCLAIMER_RE.search(body):
        issues.append("contains a disclaimer / says it is simulated")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="synthetic-v1")
    parser.add_argument("--original", action="store_true",
                        help="build the original set (no replacements) as synthetic-old")
    parser.add_argument("--check", action="store_true", help="report only; write nothing")
    parser.add_argument("--allow-missing", action="store_true", help="freeze even if some specs have no email")
    args = parser.parse_args(argv)

    specs = {s["id"]: s for s in json.loads((SYN / "specs.json").read_text(encoding="utf-8"))}
    replaced: dict[str, str] = {}
    superseded: set[str] = set()
    split_name = "synthetic"
    if args.original:
        args.version = split_name = "synthetic-old"
    elif (SYN / "replacements.json").exists():
        rep = json.loads((SYN / "replacements.json").read_text(encoding="utf-8"))
        replaced, superseded = rep["replaced"], set(rep.get("superseded", {}))
    dropped = set(replaced.values()) | superseded
    for new_id, old_id in replaced.items():
        specs[new_id] = {**specs[old_id], "id": new_id, "replaces": old_id}
    for old_id in set(replaced.values()):
        del specs[old_id]
    got: dict[str, dict] = {}
    problems: list[str] = []
    for path in sorted((SYN / "raw").glob("*/*.*")):
        writer = path.parent.name
        records, bad = parse_reply(path)
        problems += bad
        for rec in records:
            rid = str(rec.get("id", ""))
            if rid in dropped or (args.original and rid.rsplit("-", 1)[-1] in ("r1", "r2")):
                continue  # not part of this set
            if rid not in specs or specs[rid]["writer"] != writer:
                problems.append(f"{writer}/{path.name}: id {rid!r} is not one of {writer}'s specs")
            elif rid in got:
                problems.append(f"{writer}/{path.name}: duplicate id {rid}")
            else:
                got[rid] = rec

    missing = sorted(set(specs) - set(got))
    flagged = {rid: issues for rid in sorted(got) if (issues := check_email(got[rid], specs[rid]))}
    print(f"{len(got)} of {len(specs)} emails found; {len(missing)} missing; "
          f"{len(flagged)} flagged; {len(problems)} file problems")
    for p in problems:
        print("  file:", p)
    for rid, issues in flagged.items():
        print(f"  {rid} ({specs[rid]['label']}): {'; '.join(issues)}")
    if missing:
        print("  missing:", ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    if args.check:
        return 0
    if problems or flagged or (missing and not args.allow_missing):
        print("not frozen: fix or regenerate the emails above (or remove them from raw/ and pass --allow-missing)")
        return 1

    records = []
    for rid in sorted(got):
        spec, rec = specs[rid], got[rid]
        body = rec["body"].replace("\r\n", "\n").strip()
        records.append({"id": rid, "source": spec["writer"], "label": spec["label"], "from": rec["from"].strip(),
                        "subject": rec["subject"].strip(), "body": body, "urls": list(extract_urls(body)),
                        "category": spec["category"], "difficulty": spec.get("difficulty"),
                        **({"replaces": spec["replaces"]} if "replaces" in spec else {})})
    # Same-writer near-copies would count one email twice; report them rather than silently drop.
    seen = Counter(" ".join(r["body"].lower().split()) for r in records)
    dupes = sum(c - 1 for c in seen.values() if c > 1)
    if dupes:
        print(f"not frozen: {dupes} emails have identical bodies")
        return 1

    processed_rel = Path("data") / "synthetic" / f"{args.version}.jsonl"
    emails = [_to_email(r, i) for i, r in enumerate(records)]
    manifest = freeze({split_name: emails}, ROOT / "data" / "splits" / args.version, {
        "version": args.version,
        "dataset": {"title": "LLM-written phishing and legitimate emails", "writers": sorted({r["source"] for r in records}),
                    "specs": "data/synthetic/specs.json", "prompts": "data/synthetic/prompts/"},
        "processed_file": str(processed_rel),
        "label_mapping": "from the spec each email was written to, not from the writer's output",
        "true_phishing_sources": sorted({r["source"] for r in records}),
        "missing_specs": missing,
        "replaced": replaced,
    })
    (ROOT / processed_rel).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"froze {args.version}: {manifest['splits'][split_name]['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
