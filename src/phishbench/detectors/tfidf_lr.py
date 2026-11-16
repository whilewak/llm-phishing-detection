"""TF-IDF + Logistic Regression baseline detector.

Learned baseline model:
- Tokenizes subject, body, and URLs with n-grams (1, 2)
- Applies sublinear term-frequency scaling
- Classifies with L2-regularized logistic regression
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from phishbench.eval.environment import peak_rss_mb
from phishbench.types import Email, Prediction, TrainingCost


def _extract_text(email: Email) -> str:
    url_str = " ".join(email.urls) if email.urls else ""
    return f"{email.subject} {email.body} {url_str}"


class TfidfLogisticRegressionDetector:
    """TF-IDF with Logistic Regression classifier."""

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        c_reg: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 1000,
    ):
        self.name = "tfidf-lr"
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.sublinear_tf = sublinear_tf
        self.c_reg = c_reg
        self.class_weight = class_weight
        self.max_iter = max_iter

        self.vectorizer: Any = None
        self.classifier: Any = None
        self._is_fitted = False

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        if not train:
            return TrainingCost()

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        t0 = time.perf_counter()
        initial_rss = peak_rss_mb()

        texts = [_extract_text(e) for e in train]
        y = [1 if e.is_phishing else 0 for e in train]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            stop_words="english",
        )
        x_train = self.vectorizer.fit_transform(texts)

        self.classifier = LogisticRegression(
            C=self.c_reg,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self.classifier.fit(x_train, y)
        self._is_fitted = True

        wall_clock = time.perf_counter() - t0
        final_rss = peak_rss_mb()
        peak_mb = max(final_rss, initial_rss)

        return TrainingCost(
            wall_clock_s=wall_clock,
            peak_memory_mb=peak_mb,
            hardware="Apple M4 CPU",
            cost_usd=0.0,
            n_train_emails=len(train),
        )

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        if not self._is_fitted or self.vectorizer is None or self.classifier is None:
            return [
                Prediction(
                    score=None,
                    error="Model not fitted",
                    latency_ms=0.0,
                    cost_usd=0.0,
                )
                for _ in emails
            ]

        t0 = time.perf_counter()
        texts = [_extract_text(e) for e in emails]
        x = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(x)[:, 1]
        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency = elapsed_total_ms / len(emails) if emails else 0.0

        return [
            Prediction(
                score=float(prob),
                latency_ms=per_item_latency,
                cost_usd=0.0,
                details={"model": "tfidf-lr"},
            )
            for prob in probs
        ]

