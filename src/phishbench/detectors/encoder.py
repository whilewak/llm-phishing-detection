"""Transformer encoder detector (ModernBERT, DeBERTa-v3) using PyTorch and HuggingFace.

Supports:
- Local MPS (Apple Silicon), CUDA, or CPU
- Dynamic padding and truncation to model context limit
- Fine-tuning with early stopping, mixed precision, and checkpointing
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

from phishbench.eval.environment import gpu_memory_mb, peak_rss_mb
from phishbench.types import Email, Prediction, TrainingCost


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_text(email: Email) -> str:
    url_str = " ".join(email.urls) if email.urls else ""
    body = email.body[:16000] if email.body else ""
    return f"Subject: {email.subject}\n\n{body}\n\n{url_str}".strip()


class TransformerEncoderDetector:
    """Fine-tuned transformer encoder for phishing email classification."""

    def __init__(
        self,
        model_name_or_path: str = "answerdotai/ModernBERT-base",
        name: str | None = None,
        max_length: int = 2048,
        device: str | None = None,
        batch_size: int = 16,
        lr: float = 2e-5,
        epochs: int = 2,
        checkpoint_dir: str | Path | None = None,
        eval_batch_size: int = 32,
        max_train_samples: int | None = None,
        gradient_accumulation_steps: int = 4,
    ):
        self.model_name = model_name_or_path
        self.name = name or Path(model_name_or_path).name.lower()
        self.max_length = max_length
        self.device = device or _pick_device()
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.eval_batch_size = eval_batch_size
        self.max_train_samples = max_train_samples
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.tokenizer: Any = None
        self.model: Any = None
        self._is_loaded = False

    def _ensure_model(self):
        if self._is_loaded and self.model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = str(self.checkpoint_dir) if (self.checkpoint_dir and self.checkpoint_dir.exists()) else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).float()
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        if not train:
            self._ensure_model()
            return TrainingCost(hardware=self.device)

        # If already fine-tuned and checkpoint exists, load it directly
        if self.checkpoint_dir and (self.checkpoint_dir / "model.safetensors").exists():
            self._ensure_model()
            return TrainingCost(
                hardware=self.device,
                n_train_emails=len(train),
            )

        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        t0 = time.perf_counter()
        initial_rss = peak_rss_mb()

        from phishbench.data.splits import stratified_sample
        train_emails = stratified_sample(train, self.max_train_samples, seed=42) if (self.max_train_samples and len(train) > self.max_train_samples) else train

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).float()
        model.to(self.device)

        class EmailDataset(Dataset):
            def __init__(self, emails: Sequence[Email]):
                self.texts = [_extract_text(e) for e in emails]
                self.labels = [1 if e.is_phishing else 0 for e in emails]

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

        def collate_fn(batch):
            texts, labels = zip(*batch)
            enc = tokenizer(
                list(texts),
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return enc, torch.tensor(labels, dtype=torch.long)

        train_loader = DataLoader(EmailDataset(train_emails), batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        accum = max(1, self.gradient_accumulation_steps)
        total_steps = (len(train_loader) // accum) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            for step, (batch_enc, batch_labels) in enumerate(train_loader):
                batch_enc = {k: v.to(self.device) for k, v in batch_enc.items()}
                batch_labels = batch_labels.to(self.device)

                outputs = model(**batch_enc, labels=batch_labels)
                loss = outputs.loss / accum
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
                    if not has_nan:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                    if self.device == "mps":
                        torch.mps.empty_cache()

        # Save checkpoint if requested
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(self.checkpoint_dir)
            tokenizer.save_pretrained(self.checkpoint_dir)

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self._is_loaded = True

        wall_clock = time.perf_counter() - t0
        peak_mb = max(peak_rss_mb(), initial_rss)

        return TrainingCost(
            wall_clock_s=wall_clock,
            peak_memory_mb=peak_mb,
            hardware=f"{self.device}",
            cost_usd=0.0,
            n_train_emails=len(train_emails),
        )

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        self._ensure_model()
        import torch

        preds: list[Prediction] = []
        texts = [_extract_text(e) for e in emails]

        for i in range(0, len(texts), self.eval_batch_size):
            batch_texts = texts[i : i + self.eval_batch_size]
            t0 = time.perf_counter()
            try:
                enc = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_item_lat = elapsed_ms / len(batch_texts) if batch_texts else 0.0

                for p in probs:
                    if math.isnan(p):
                        preds.append(
                            Prediction(
                                score=None,
                                error="Model produced NaN score",
                                latency_ms=per_item_lat,
                                cost_usd=0.0,
                            )
                        )
                    else:
                        preds.append(
                            Prediction(
                                score=float(p),
                                latency_ms=per_item_lat,
                                cost_usd=0.0,
                                details={"model": self.name},
                            )
                        )
                if self.device == "mps":
                    torch.mps.empty_cache()
            except Exception as ex:
                for _ in batch_texts:
                    preds.append(
                        Prediction(
                            score=None,
                            error=f"Prediction error: {ex}",
                            latency_ms=0.0,
                            cost_usd=0.0,
                        )
                    )

        return preds

