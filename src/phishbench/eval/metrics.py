"""Detection, calibration and latency metrics.

Conventions:
- `y` is a sequence of bools (True = phishing).
- `scores` are floats; an email is flagged when score >= threshold.
- Failed predictions are always counted wrong (see `effective_scores`).
- A threshold of +inf means "flag nothing".
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from phishbench.types import Prediction

# Scores outside [0, 1] that put a failed prediction on the wrong side of every
# threshold in [0, 1]: a failed phishing email is never flagged (a miss), a
# failed legitimate email is always flagged (a false positive).
FAILED_PHISHING_SCORE = -1.0
FAILED_LEGITIMATE_SCORE = 2.0


def effective_scores(y: Sequence[bool], preds: Sequence[Prediction]) -> list[float]:
    if len(y) != len(preds):
        raise ValueError(f"{len(y)} labels but {len(preds)} predictions")
    out = []
    for is_phish, p in zip(y, preds):
        if p.failed:
            out.append(FAILED_PHISHING_SCORE if is_phish else FAILED_LEGITIMATE_SCORE)
        else:
            out.append(float(p.score))
    return out


def _counts(y: Sequence[bool]) -> tuple[int, int]:
    pos = sum(1 for v in y if v)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise ValueError("need at least one phishing and one non-phishing email")
    return pos, neg


def operating_points(y: Sequence[bool], scores: Sequence[float]) -> list[tuple[float, int, int]]:
    """(threshold, true positives, false positives) for every distinct score,
    from the highest threshold down. Tied scores move together."""
    pairs = sorted(zip(scores, y), key=lambda t: -t[0])
    points = []
    tp = fp = 0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == t:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append((t, tp, fp))
    return points


def recall_at_fpr(y: Sequence[bool], scores: Sequence[float], max_fpr: float) -> tuple[float, float]:
    """Highest recall whose false-positive rate is <= max_fpr, and its threshold."""
    pos, neg = _counts(y)
    best_recall, best_t = 0.0, math.inf
    for t, tp, fp in operating_points(y, scores):
        if fp / neg > max_fpr + 1e-12:
            break
        best_recall, best_t = tp / pos, t
    return best_recall, best_t


def average_precision(y: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (step-wise, ties grouped)."""
    pos, _ = _counts(y)
    ap, prev_recall = 0.0, 0.0
    for _, tp, fp in operating_points(y, scores):
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def precision_at_prevalence(self, prevalence: float) -> float:
        """Precision if phishing made up `prevalence` of all mail."""
        hit = self.recall * prevalence
        false = self.fpr * (1 - prevalence)
        return hit / (hit + false) if hit + false else 0.0

    def as_dict(self, prevalence: float = 0.01) -> dict:
        return {
            "threshold": None if math.isinf(self.threshold) else self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.fpr,
            f"precision_at_{prevalence:g}_prevalence": self.precision_at_prevalence(prevalence),
        }


def at_threshold(y: Sequence[bool], scores: Sequence[float], threshold: float) -> ThresholdMetrics:
    tp = fp = tn = fn = 0
    for is_phish, s in zip(y, scores):
        flagged = s >= threshold
        if is_phish and flagged:
            tp += 1
        elif is_phish:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return ThresholdMetrics(threshold, tp, fp, tn, fn)


def expected_calibration_error(y: Sequence[bool], probs: Sequence[float], n_bins: int = 10) -> float:
    """ECE over equal-width bins. Pass only successful predictions."""
    if not probs:
        return math.nan
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, label in zip(probs, y):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, label))
    total = len(probs)
    ece = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(1 for _, label in b if label) / len(b)
            ece += len(b) / total * abs(conf - acc)
    return ece


def tie_share(scores: Sequence[float]) -> float:
    """Share of emails whose score is shared with at least one other email."""
    if not scores:
        return 0.0
    counts = Counter(scores)
    return sum(c for c in counts.values() if c > 1) / len(scores)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_ci(
    metric: Callable[[Sequence[bool], Sequence[float]], float],
    y: Sequence[bool],
    scores: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval. Resamples lacking either class are redrawn."""
    rng = random.Random(seed)
    n = len(y)
    stats: list[float] = []
    attempts = 0
    while len(stats) < n_resamples and attempts < n_resamples * 10:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [y[i] for i in idx]
        if all(ys) or not any(ys):
            continue
        stats.append(metric(ys, [scores[i] for i in idx]))
    if not stats:
        return math.nan, math.nan
    stats.sort()
    lo = stats[int(alpha / 2 * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return lo, hi
