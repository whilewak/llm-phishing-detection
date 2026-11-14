"""Build, freeze and load stratified splits.

A frozen split set is a directory holding one id list per split plus a
manifest with counts and content fingerprints. Loading a split re-reads the
processed emails and refuses to continue if the content no longer matches.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from phishbench.data.loader import dataset_fingerprint, load_emails
from phishbench.types import Email

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
LLM_SUBSAMPLES = {"val_llm": "val", "test_llm": "test"}


def _stratum(e: Email) -> tuple[str, str]:
    return (e.source or "", e.label or "")


def _allocate(total: int, weights: dict, ) -> dict:
    """Largest-remainder rounding of total * weight per key."""
    raw = {k: total * w for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[:remainder]:
        out[k] += 1
    return out


def stratified_split(emails: Sequence[Email], seed: int) -> dict[str, list[Email]]:
    """70/15/15 within every (source, label) stratum."""
    strata: dict[tuple, list[Email]] = defaultdict(list)
    for e in emails:
        strata[_stratum(e)].append(e)
    out: dict[str, list[Email]] = {name: [] for name in SPLIT_FRACTIONS}
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda e: e.id)
        random.Random(f"{seed}:{key}").shuffle(members)
        sizes = _allocate(len(members), SPLIT_FRACTIONS)
        start = 0
        for name in SPLIT_FRACTIONS:
            out[name].extend(members[start : start + sizes[name]])
            start += sizes[name]
    return out


def stratified_sample(emails: Sequence[Email], size: int, seed: int) -> list[Email]:
    """Proportional sample by (source, label)."""
    if size >= len(emails):
        return list(emails)
    strata: dict[tuple, list[Email]] = defaultdict(list)
    for e in emails:
        strata[_stratum(e)].append(e)
    counts = _allocate(size, {k: len(v) / len(emails) for k, v in strata.items()})
    out: list[Email] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda e: e.id)
        out.extend(random.Random(f"{seed}:sample:{key}").sample(members, counts[key]))
    return out


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _counts(emails: Sequence[Email]) -> dict:
    by = Counter(f"{e.source}/{e.label}" for e in emails)
    labels = Counter(e.label for e in emails)
    return {"n": len(emails), "labels": dict(sorted(labels.items())), "by_source_label": dict(sorted(by.items()))}


def freeze(splits: dict[str, list[Email]], out_dir: Path, info: dict) -> dict:
    """Write id lists and the manifest. Refuses to overwrite an existing split set."""
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} is already frozen; build a new version instead")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**info, "splits": {}}
    for name, emails in splits.items():
        ids = sorted(e.id for e in emails)
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        manifest["splits"][name] = {
            **_counts(emails),
            "ids_sha256": _ids_hash(ids),
            "content_fingerprint": dataset_fingerprint(emails),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split(split_dir: str | Path, name: str, processed: str | Path | None = None) -> list[Email]:
    """Load one frozen split and verify its ids and content against the manifest."""
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["splits"][name]
    ids = [line for line in (split_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line]
    if _ids_hash(ids) != entry["ids_sha256"]:
        raise ValueError(f"{name}.txt does not match the manifest's id hash")
    # Split sets live at <root>/data/splits/<version>; processed_file is relative to <root>.
    processed = Path(processed) if processed else split_dir.resolve().parents[2] / manifest["processed_file"]
    wanted = set(ids)
    emails = [e for e in load_emails(processed) if e.id in wanted]
    if len(emails) != len(ids):
        raise ValueError(f"{len(ids) - len(emails)} ids in {name} are missing from {processed}")
    if dataset_fingerprint(emails) != entry["content_fingerprint"]:
        raise ValueError(f"content of {name} differs from when it was frozen; rebuild {processed}")
    return emails
