"""Exact and near-duplicate removal (MinHash + LSH).

Text is normalised to lowercase alphanumeric tokens first, so the tokenised
Enron/Ling copies and the raw TREC-05 copies of the same Enron email match.
Within a duplicate group one email is kept; if the group mixes labels, the
whole group is dropped as label noise.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from phishbench.types import Email

_TOKEN = re.compile(r"[a-z0-9]+")
_MERSENNE = np.uint64((1 << 61) - 1)
SHINGLE = 5
MAX_TOKENS = 3000


def normalized_tokens(email: Email) -> list[str]:
    return _TOKEN.findall(f"{email.subject} {email.body}".lower())[:MAX_TOKENS]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupReport:
    exact_duplicates: int = 0
    near_duplicates: int = 0
    mixed_label_groups: int = 0
    mixed_label_emails: int = 0
    kept: int = 0
    params: dict = field(default_factory=dict)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, tokens: Sequence[str]) -> np.ndarray:
        if len(tokens) < SHINGLE:
            shingles = [" ".join(tokens)]
        else:
            shingles = [" ".join(tokens[i : i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)]
        x = np.fromiter((zlib.crc32(s.encode()) for s in set(shingles)), dtype=np.uint64)
        # (a * x + b) mod p: a, x < 2^32 so a * x fits in uint64.
        h = (x[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE
        return h.min(axis=0)


def deduplicate(
    emails: Sequence[Email],
    num_perm: int = 128,
    bands: int = 16,
    threshold: float = 0.8,
) -> tuple[list[Email], DedupReport]:
    """Return the emails to keep (in input order) and what was removed."""
    rows = num_perm // bands
    report = DedupReport(params={"shingle_tokens": SHINGLE, "num_perm": num_perm, "bands": bands,
                                 "rows_per_band": rows, "jaccard_threshold": threshold})
    n = len(emails)
    uf = _UnionFind(n)

    tokens = [normalized_tokens(e) for e in emails]

    # Exact duplicates on the normalised token stream.
    first_seen: dict[str, int] = {}
    for i, toks in enumerate(tokens):
        key = " ".join(toks)
        if key in first_seen:
            uf.union(first_seen[key], i)
            report.exact_duplicates += 1
        else:
            first_seen[key] = i
    representatives = sorted(first_seen.values())

    # Near duplicates among exact-unique emails.
    hasher = MinHasher(num_perm)
    sigs = {i: hasher.signature(tokens[i]) for i in representatives}
    for band in range(bands):
        buckets: dict[bytes, int] = {}
        lo, hi = band * rows, (band + 1) * rows
        for i in representatives:
            key = sigs[i][lo:hi].tobytes()
            j = buckets.setdefault(key, i)
            if j != i and uf.find(i) != uf.find(j):
                if float(np.mean(sigs[i] == sigs[j])) >= threshold:
                    uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    keep: list[int] = []
    for members in groups.values():
        labels = {emails[i].label for i in members}
        if len(labels) > 1:
            report.mixed_label_groups += 1
            report.mixed_label_emails += len(members)
            continue
        keep.append(min(members))
    report.near_duplicates = n - report.exact_duplicates - len(groups)
    keep.sort()
    report.kept = len(keep)
    return [emails[i] for i in keep], report
