"""One evaluation harness for every detector.

The harness never contains model-specific code: it calls `fit`, picks a
threshold on validation, times `predict` on the test emails and computes the
same metrics for every model.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from phishbench.data.loader import dataset_fingerprint
from phishbench.eval import metrics as m
from phishbench.eval.environment import describe_environment, gpu_memory_mb, peak_rss_mb
from phishbench.types import Detector, Email, Prediction

HEADLINE_FPR = 0.01
SECONDARY_FPR = 0.001
DEFAULT_THRESHOLD = 0.5


def _split_info(emails: Sequence[Email]) -> dict:
    labels: dict[str, int] = {}
    for e in emails:
        labels[e.label or "unknown"] = labels.get(e.label or "unknown", 0) + 1
    return {"n": len(emails), "labels": labels, "fingerprint": dataset_fingerprint(emails)}


def _has_both_classes(emails: Sequence[Email]) -> bool:
    flags = {e.is_phishing for e in emails if e.label is not None}
    return flags == {True, False}


def _predict_all(detector: Detector, emails: Sequence[Email], batch_size: int) -> tuple[list[Prediction], float]:
    preds: list[Prediction] = []
    start = time.perf_counter()
    for i in range(0, len(emails), batch_size):
        batch = emails[i : i + batch_size]
        out = detector.predict(batch)
        if len(out) != len(batch):
            raise RuntimeError(f"{detector.name} returned {len(out)} predictions for {len(batch)} emails")
        preds.extend(out)
    return preds, time.perf_counter() - start


def _finite(x: float) -> float | None:
    return None if x is None or math.isinf(x) or math.isnan(x) else x


def _choose_threshold(detector: Detector, val: Sequence[Email], batch_size: int) -> dict:
    labelled = [e for e in val if e.label is not None]
    if not _has_both_classes(labelled):
        return {"value": DEFAULT_THRESHOLD, "chosen_on": "default (no usable validation data)"}
    preds, _ = _predict_all(detector, labelled, batch_size)
    y = [e.is_phishing for e in labelled]
    _, t = m.recall_at_fpr(y, m.effective_scores(y, preds), HEADLINE_FPR)
    return {"value": t, "chosen_on": f"validation, recall at {HEADLINE_FPR:.0%} false-positive rate"}


def _breakdown(
    emails: list[Email],
    scores: list[float],
    thresholds: dict[str, float],
    source_groups: dict[str, list[str]],
) -> dict:
    """Recall on phishing and false-positive rate on the rest, per source and per
    named group of sources, at each named threshold."""
    groups: dict[str, set[str]] = {s: {s} for s in sorted({e.source for e in emails if e.source})}
    groups.update({name: set(members) for name, members in source_groups.items()})
    out: dict[str, Any] = {}
    for gname, members in groups.items():
        rows = [(e.is_phishing, s) for e, s in zip(emails, scores) if e.source in members]
        n_pos = sum(1 for p, _ in rows if p)
        n_neg = len(rows) - n_pos
        entry: dict[str, Any] = {"n_phishing": n_pos, "n_not_phishing": n_neg}
        for tname, t in thresholds.items():
            entry[tname] = {
                "recall": sum(1 for p, s in rows if p and s >= t) / n_pos if n_pos else None,
                "false_positive_rate": sum(1 for p, s in rows if not p and s >= t) / n_neg if n_neg else None,
            }
        out[gname] = entry
    return out


def _detection_metrics(y: list[bool], scores: list[float], threshold: float, n_bootstrap: int) -> dict:
    out: dict[str, Any] = {}
    for name, fpr in (("headline", HEADLINE_FPR), ("secondary", SECONDARY_FPR)):
        recall, t = m.recall_at_fpr(y, scores, fpr)
        lo, hi = m.bootstrap_ci(lambda ys, ss, fpr=fpr: m.recall_at_fpr(ys, ss, fpr)[0], y, scores, n_bootstrap)
        out[f"{name}_recall_at_fpr"] = {
            "max_fpr": fpr,
            "recall": recall,
            "ci95": [lo, hi],
            "threshold": _finite(t),
        }
    out["pr_auc"] = m.average_precision(y, scores)
    out["at_chosen_threshold"] = m.at_threshold(y, scores, threshold).as_dict()
    return out


def evaluate(
    detector: Detector,
    test: Sequence[Email],
    *,
    train: Sequence[Email] = (),
    val: Sequence[Email] = (),
    warmup: int = 3,
    batch_size: int = 1,
    n_bootstrap: int = 1000,
    config: dict | None = None,
    source_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Run one detector through fit, threshold choice and test scoring."""
    training_cost = detector.fit(train, val)

    threshold = _choose_threshold(detector, val, batch_size)

    if warmup and test:
        for _ in range(warmup):
            detector.predict(test[:batch_size])

    preds, wall_s = _predict_all(detector, test, batch_size)

    labelled = [(e, p) for e, p in zip(test, preds) if e.label is not None]
    y = [e.is_phishing for e, _ in labelled]
    lpreds = [p for _, p in labelled]
    scores = m.effective_scores(y, lpreds)
    ok = [(yi, p.score) for yi, p in zip(y, lpreds) if not p.failed]

    failures = [p for p in preds if p.failed]
    latencies = [p.latency_ms for p in preds]
    cost = sum(p.cost_usd for p in preds)

    result: dict[str, Any] = {
        "model": detector.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config or {},
        "data": {"train": _split_info(train), "val": _split_info(val), "test": _split_info(test)},
        "threshold": {**threshold, "value": _finite(threshold["value"])},
        "errors": {
            "count": len(failures),
            "rate": len(failures) / len(preds) if preds else 0.0,
            "examples": sorted({p.error or "no score" for p in failures})[:5],
            "rule": "always counted wrong: a miss if phishing, a false positive if not",
        },
        "score_ties": {"tie_share": m.tie_share([s for _, s in ok])},
        "calibration": {
            "ece": _finite(m.expected_calibration_error([yi for yi, _ in ok], [s for _, s in ok]))
        },
        "speed": {
            "latency_ms_p50": _finite(m.percentile(latencies, 50)),
            "latency_ms_p95": _finite(m.percentile(latencies, 95)),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "throughput_per_s": len(preds) / wall_s if wall_s > 0 else None,
            "batch_size": batch_size,
            "warmup_calls": warmup,
        },
        "inference_cost_usd": {
            "total": cost,
            "per_million_emails": cost / len(preds) * 1e6 if preds else 0.0,
        },
        "training_cost": asdict(training_cost),
        "resources": {"peak_rss_mb": peak_rss_mb(), "gpu": gpu_memory_mb()},
        "environment": describe_environment(),
    }

    if _has_both_classes([e for e, _ in labelled]):
        det = _detection_metrics(y, scores, threshold["value"], n_bootstrap)
        headline_t = det["headline_recall_at_fpr"]["threshold"]
        det["by_source"] = _breakdown(
            [e for e, _ in labelled],
            scores,
            {
                "at_chosen_threshold": threshold["value"],
                "at_headline_threshold": math.inf if headline_t is None else headline_t,
            },
            source_groups or {},
        )
        result["detection"] = det
    else:
        result["detection"] = None
        result["detection_note"] = "test data lacks phishing or non-phishing emails; detection metrics skipped"

    result["predictions"] = [
        {
            "id": e.id,
            "label": e.label,
            "score": p.score,
            "model_label": p.label,
            "latency_ms": p.latency_ms,
            "cost_usd": p.cost_usd,
            "error": p.error,
            "details": p.details,
        }
        for e, p in zip(test, preds)
    ]
    return result


def write_result(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(result["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{result['model']}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
