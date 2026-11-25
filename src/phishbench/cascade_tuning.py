"""Choose a cascade's hand-off thresholds from saved validation scores.

Works on arrays, one per tier, aligned on the same emails (needs numpy). A
failed prediction is NaN in an early tier (it escalates) and is counted wrong
in the last tier, exactly as in `detectors/cascade.py` and the harness.

The search maximises validation recall subject to two limits: the cascade's
false-positive rate stays within `max_fpr`, and at most `share_cap` of emails
reach the last tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phishbench.eval.metrics import FAILED_LEGITIMATE_SCORE, FAILED_PHISHING_SCORE


def as_array(scores: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if s is None else s for s in scores], dtype=float)


def last_tier_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Failures become scores that are wrong at every threshold."""
    return np.where(np.isnan(scores), np.where(y, FAILED_PHISHING_SCORE, FAILED_LEGITIMATE_SCORE), scores)


@dataclass
class Outcome:
    thresholds: list[tuple[float, float]]  # (low, high) per early tier
    last_threshold: float  # flag at the last tier when score >= this; inf = flag nothing
    recall: float
    fpr: float
    exit_share: list[float]  # share of emails answered by each tier, last tier included

    def as_dict(self) -> dict:
        fin = lambda v: None if math.isinf(v) else float(v)  # noqa: E731
        return {"thresholds": [{"low": fin(lo), "high": fin(hi)} for lo, hi in self.thresholds],
                "last_threshold": fin(self.last_threshold), "recall": self.recall,
                "false_positive_rate": self.fpr, "exit_share": self.exit_share,
                "top_tier_share": self.exit_share[-1]}


def apply(early: Sequence[np.ndarray], last: np.ndarray, y: np.ndarray,
          thresholds: Sequence[tuple[float, float]], last_threshold: float) -> Outcome:
    """Run fixed thresholds over saved scores (used for the test split)."""
    pending = np.ones(len(y), dtype=bool)
    flagged = np.zeros(len(y), dtype=bool)
    shares = []
    for scores, (lo, hi) in zip(early, thresholds):
        phish = pending & (scores >= hi)
        legit = pending & (scores <= lo)
        flagged |= phish
        shares.append(float((phish | legit).mean()))
        pending &= ~(phish | legit)
    flagged |= pending & (last_tier_scores(last, y) >= last_threshold)
    shares.append(float(pending.mean()))
    return Outcome(list(thresholds), last_threshold, float((flagged & y).sum() / y.sum()),
                   float((flagged & ~y).sum() / (~y).sum()), shares)


def _candidates(scores: np.ndarray, grid: int) -> list[tuple[float, float]]:
    ok = scores[~np.isnan(scores)]
    qs = np.unique(np.quantile(ok, np.linspace(0.02, 0.98, grid)))
    lows = [-math.inf, *qs]  # -inf: this tier never clears an email as legitimate
    highs = [*qs, math.inf]  # +inf: this tier never flags on its own
    return [(lo, hi) for lo in lows for hi in highs if lo < hi]


def tune(early: Sequence[np.ndarray], last: np.ndarray, y: np.ndarray, *, max_fpr: float = 0.01,
         share_caps: Sequence[float] = (0.10,), grid: int = 30) -> dict[float, Outcome]:
    """Best thresholds on these (validation) scores for each cap on the top-tier share."""
    y = np.asarray(y, dtype=bool)
    n, pos, neg = len(y), int(y.sum()), int((~y).sum())
    budget = int(math.floor(max_fpr * neg + 1e-9))
    eff = last_tier_scores(last, y)
    order = np.argsort(-eff, kind="stable")
    s_sorted, y_sorted = eff[order], y[order]
    group_end = np.append(s_sorted[1:] != s_sorted[:-1], True)
    cands = [_candidates(s, grid) for s in early]
    best: dict[float, tuple[tuple, Outcome]] = {}

    def leaf(pending: np.ndarray, tp: int, fp: int, chosen: list, shares: list) -> None:
        share = float(pending.mean())
        caps = [c for c in share_caps if share <= c + 1e-12]
        if not caps:
            return
        m = pending[order]
        ctp, cfp = np.cumsum(y_sorted & m), np.cumsum(~y_sorted & m)
        valid = np.flatnonzero(group_end & (cfp <= budget - fp))
        if len(valid):
            i = valid[-1]
            tp_all, fp_all, thr = tp + int(ctp[i]), fp + int(cfp[i]), float(s_sorted[i])
        else:
            tp_all, fp_all, thr = tp, fp, math.inf
        key = (tp_all, -share)  # most recall, then fewest emails at the top tier
        for c in caps:
            if c not in best or key > best[c][0]:
                best[c] = (key, Outcome(list(chosen), thr, tp_all / pos, fp_all / neg, [*shares, share]))

    def walk(t: int, pending: np.ndarray, tp: int, fp: int, chosen: list, shares: list) -> None:
        if t == len(early):
            leaf(pending, tp, fp, chosen, shares)
            return
        scores = early[t]
        for lo, hi in cands[t]:
            phish = pending & (scores >= hi)
            fp2 = fp + int((phish & ~y).sum())
            if fp2 > budget:
                continue
            legit = pending & (scores <= lo)
            done = phish | legit
            walk(t + 1, pending & ~done, tp + int((phish & y).sum()), fp2,
                 [*chosen, (float(lo), float(hi))], [*shares, float(done.sum() / n)])

    walk(0, np.ones(n, dtype=bool), 0, 0, [], [])
    return {c: o for c, (_, o) in best.items()}
