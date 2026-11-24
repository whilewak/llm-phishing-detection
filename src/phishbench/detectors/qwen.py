"""Pinned Qwen on MLX, scored from the next answer token's log probabilities.

A means phishing and B means legitimate. The score is P(A | A or B), with
unconditional answer probabilities retained for auditing. No text is generated
or parsed. Imports and model loading are deferred until inference is required.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Sequence

from phishbench import prompts
from phishbench.types import Email, Prediction, TrainingCost

MODEL = "mlx-community/Qwen3.5-4B-4bit"
REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
ANSWER_FORMAT = "Reply with exactly A for phishing or B for legitimate."


class MLXBackend:
    def __init__(self, model: str, revision: str):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.prefix = []
        self.prefix_cache = None
        self.model, self.tokenizer = load(model, revision=revision)
        self.answer_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B")]
        if any(len(ids) != 1 for ids in self.answer_ids):
            raise ValueError("A and B must each encode as one answer token")

    def render(self, messages):
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        return tokens["input_ids"] if isinstance(tokens, Mapping) else tokens

    def score(self, tokens, prefix_length=0):
        from mlx_lm.generate import generate_step
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        prefix = tokens[:prefix_length]
        if prefix != self.prefix or self.prefix_cache is None:
            self.prefix_cache = make_prompt_cache(self.model)
            for i in range(0, len(prefix), 512):
                self.model(mx.array(prefix[i:i+512])[None], cache=self.prefix_cache)
                mx.eval([c.state for c in self.prefix_cache])
            self.prefix = prefix
        # float32 avoids rounded bf16 log-softmax values (including mass > 1).
        gen = generate_step(mx.array(tokens[prefix_length:]), self.model, max_tokens=1,
                    prompt_cache=copy.deepcopy(self.prefix_cache), prefill_step_size=512,
                    logits_processors=[lambda _, logits: logits.astype(mx.float32)])
        try:
            greedy, logprobs = next(gen)
            a, b = [float(logprobs[ids[0]].item()) for ids in self.answer_ids]
            mx.synchronize()
            return a, b, {"greedy_token": self.tokenizer.decode([greedy]),
                          "mlx_peak_memory_mb": mx.get_peak_memory() / 2**20}
        finally:
            gen.close()
            mx.clear_cache()


class QwenDetector:
    def __init__(self, model: str = MODEL, revision: str = REVISION, n_shots: int = 0,
                 few_shot: str | None = None, split_dir: str | None = None,
                 max_input_tokens: int = 8192,
                 cache_dir: str | None = "data/cache/qwen-shared-v1",
                 backend=None):
        if n_shots not in (0, 8) or max_input_tokens < 256:
            raise ValueError("invalid shot count or token budget")
        if n_shots == 8 and (not few_shot or not split_dir):
            raise ValueError("eight-shot mode requires the frozen few-shot file and split directory")
        if n_shots == 0 and few_shot:
            raise ValueError("zero-shot mode cannot include few-shot examples")
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError("revision must be an immutable 40-character commit hash")
        self.name = f"qwen3.5-4b-{n_shots}shot-shared-v1"
        self.model, self.revision = model, revision
        self.n_shots = n_shots
        self.few_shot, self.split_dir = few_shot, split_dir
        self.max_input_tokens = max_input_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.runtime = {}
        for package in ("mlx", "mlx-lm", "transformers", "tokenizers"):
            try:
                self.runtime[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                pass
        self.backend = backend
        self.examples: list[Email] = []
        self.fitted = n_shots == 0

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        if {e.id for e in train} & {e.id for e in val}:
            raise ValueError("training and validation overlap")
        if self.n_shots:
            examples = prompts.load_fewshot(self.few_shot, self.split_dir)
            if len(examples) != 8 or sum(e.is_phishing for e in examples) != 4:
                raise ValueError("frozen few-shot pool must contain four examples per class")
            train_by_id = {e.id: e for e in train}
            if any(train_by_id.get(e.id) != e for e in examples):
                raise ValueError("few-shot examples must match the supplied training split")
            if {e.id for e in examples} & {e.id for e in val}:
                raise ValueError("few-shot examples overlap validation")
            self.examples = examples
        else:
            self.examples = []
        self.fitted = True
        return TrainingCost(hardware="Apple Silicon MLX (prompting only)",
                            n_train_emails=len(self.examples))

    def _backend(self):
        if self.backend is None:
            self.backend = MLXBackend(self.model, self.revision)
        return self.backend

    def _tokens(self, email):
        backend = self._backend()
        parts = [prompts.TASK_INSTRUCTIONS, ANSWER_FORMAT]
        if self.examples:
            parts.append("Examples:")
            for i, example in enumerate(self.examples, 1):
                example_text, _ = prompts.render_email(example)
                answer = "A" if example.is_phishing else "B"
                parts.append(f"Example {i}:\n{example_text}\nAnswer: {answer}")
        messages = [{"role": "system", "content": "\n\n".join(parts)}]
        text, truncated = prompts.render_email(email)
        def render(value):
            return backend.render(messages + [{"role": "user", "content": value}])
        empty = render("")
        tokens = render(text)
        original = len(tokens)
        # Keep Claude's email structure, examples and URLs intact. If Qwen's
        # tokenizer needs more room, reduce only the target email's body cap.
        if len(tokens) > self.max_input_tokens:
            low, high = 0, min(len(email.body), prompts.MAX_INPUT_CHARS)
            best_text = None
            while low <= high:
                cap = (low + high) // 2
                candidate, _ = prompts.render_email(email, cap)
                candidate_tokens = render(candidate)
                if len(candidate_tokens) <= self.max_input_tokens:
                    best_text = candidate
                    low = cap + 1
                else:
                    high = cap - 1
            if best_text is None:
                raise ValueError("instructions, examples and email headers exceed context budget")
            tokens = render(best_text)
            truncated = True
        prefix_length = 0
        for a, b in zip(tokens, empty):
            if a != b:
                break
            prefix_length += 1
        return tokens, original, min(prefix_length, len(tokens)-1), truncated

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        if not self.fitted:
            raise RuntimeError("call fit with the training split before few-shot inference")
        if {e.id for e in emails} & {e.id for e in self.examples}:
            raise ValueError("evaluation contains a few-shot demonstration")
        return [self._predict_one(e) for e in emails]

    def _predict_one(self, email):
        start = time.perf_counter()
        try:
            tokens, original, prefix_length, truncated = self._tokens(email)
            identity = {"model": self.model, "revision": self.revision, "tokens": tokens,
                        "runtime": self.runtime, "prefix_length": prefix_length,
                        "example_ids": [e.id for e in self.examples],
                        "prompt_version": prompts.PROMPT_VERSION,
                        "scoring": "conditional-A-B-float32-shared-v1"}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = self.cache_dir / f"{key}.json" if self.cache_dir else None
            if path and path.exists():
                pred = Prediction(**json.loads(path.read_text()))
                pred.details = {**pred.details, "cache_hit": True}
                return pred
            a, b, extra = self._backend().score(tokens, prefix_length)
            if not math.isfinite(a) or not math.isfinite(b) or a > 0 or b > 0:
                raise ValueError("non-finite or positive answer log probabilities")
            top = max(a, b)
            score = math.exp(a - top) / (math.exp(a - top) + math.exp(b - top))
            pred = Prediction(score=score, latency_ms=(time.perf_counter()-start)*1000,
                details={**extra, "answer_logprobs": {"A_phishing": a, "B_legitimate": b},
                         "answer_probability_mass": math.exp(a)+math.exp(b),
                         "input_tokens": len(tokens), "original_input_tokens": original,
                         "truncated": truncated, "cache_hit": False,
                         "example_ids": [e.id for e in self.examples], "revision": self.revision})
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(asdict(pred), allow_nan=False))
                temp.replace(path)
            return pred
        except Exception as exc:
            return Prediction(score=None, latency_ms=(time.perf_counter()-start)*1000,
                              error=f"{type(exc).__name__}: {exc}")
