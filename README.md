# LLM & Multi-Tier Phishing Detection Benchmark

A research and benchmarking project comparing different kinds of AI models for phishing email detection across **accuracy**, **latency** and **cost**, and testing whether a tiered cascade of them beats any single model.

The full project plan is in [docs/PLAN.md](docs/PLAN.md).

---

## Models compared

| Cascade tier | Candidates | Runs on |
| :--- | :--- | :--- |
| 1 · Cheap baselines | Handwritten rules, TF-IDF + logistic regression | Local CPU |
| 2 · Encoders / System 1 | Laya (ModernBERT-large System 1 decision model), fine-tuned ModernBERT and DeBERTa-v3 | Local GPU (Apple MPS) |
| 3 · Small language model | Qwen ~4B, prompted | Local (MLX) |
| 4 · Frontier LLM | Claude Sonnet 5.5 and Haiku 5.5, GPT-6.1 Sol and GPT-6 Luna | API (Amazon Bedrock) |

Each model is also evaluated on its own. The cascade escalates only uncertain emails to the next tier; each tier's model is chosen on validation data. Target: keep ≥95% of the best single model's recall (at 1% false-positive rate) while sending ≤10% of emails to the top tier.

All models are scored on the same frozen splits of a combined public phishing dataset, through one evaluation harness. Results on it will be labelled preliminary (see the plan for why).

---

## Status

Phases 0 and 1 are done: the shared `Detector` interface, one evaluation harness with the plan's metrics, a Laya detector, and frozen splits of a combined public dataset (`data/splits/champa2023-v1`: 171,081 deduplicated emails from 9 corpora; see the plan's Dataset section). No benchmark results exist yet; Phases 2–5 (the models) are next.

The 7 sample emails in `data/benchmark_emails.json` are a smoke test only: Laya's signal weights were hand-picked on these same emails, so its results on them are not evidence of accuracy.

| Path | What it holds |
| :--- | :--- |
| `src/phishbench/types.py` | `Email`, `Prediction`, `TrainingCost` and the `Detector` interface |
| `src/phishbench/eval/` | Harness, metrics (recall at fixed false-positive rate, PR-AUC, calibration, bootstrap CIs), environment capture |
| `src/phishbench/detectors/` | One module per model; currently `laya.py` |
| `src/phishbench/data/` | Email loader, cleaning, deduplication, split building and verified split loading |
| `data/splits/` | Frozen split id lists and manifests (the emails themselves are rebuilt locally, see below) |
| `configs/models/` | One TOML config per model |
| `scripts/evaluate.py` | Runs one model and writes a results JSON to `results/` |

---

## Laya

**Laya** is an open-weights System 1 decision model built on a **ModernBERT-large** encoder (~421M parameters).

- **Non-autoregressive:** it does not generate text token by token, so there is no output to parse.
- Answers several questions about an email (urgency, credential requests, payment pressure, attachments, overall classification) as probabilities in a single forward pass.

POC hardware: MacBook Air (Apple M4, 10 CPU cores, 10 GPU cores, 16 GB unified memory), PyTorch on Apple Metal (`MPS`). First harness run on the 7 sample emails (laya 0.4.31, torch 2.15.0, all six questions, one email at a time, after warmup): p50 latency ~415 ms, p95 ~930 ms, ~2 emails/s; ~3.1 GB of MPS GPU memory held by the driver (~1.6 GB allocated) plus ~1 GB process RSS. Seven emails is far too few for stable numbers; Phase 1 data will give real ones.

---

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[laya]"
python scripts/evaluate.py --model configs/models/laya.toml --test data/benchmark_emails.json
```

To use the frozen splits, download the source data (~400 MB from Zenodo, checked against its MD5s) and rebuild the cleaned emails; the build re-creates exactly the committed splits or fails:

```bash
pip install -e ".[data]"
python scripts/download_data.py
python scripts/build_splits.py --verify
python scripts/evaluate.py --model configs/models/laya.toml --splits data/splits/champa2023-v1 --test test_llm --val val_llm
```

The harness and tests need only the standard library (Python 3.11+); the `laya` extra adds the model's dependencies.

```bash
python -m unittest
```

---

## Continuing the phases

Start with [docs/progress.md](docs/progress.md) (what's done and the known issues to carry forward), then the phase's row in the [roadmap](docs/PLAN.md#roadmap) (its work and the gate that must pass before the next phase).

### Workflow for a phase

1. Create a branch for the phase.
2. Add each model as a detector, with a config and tests (below).
3. Run it on the frozen splits and keep the results JSON in `results/`.
4. Check the phase's gate, then update `docs/progress.md`: status table, results, anything new under "Known issues".

### Adding a model

Each model is one module in `src/phishbench/detectors/` that implements the `Detector` interface from `src/phishbench/types.py`. The harness has no model-specific code, so nothing else needs to change.

```python
from typing import Sequence
from phishbench.types import Email, Prediction, TrainingCost

class MyDetector:
    def __init__(self, some_param: float = 1.0):
        self.name = "my-model"                 # used in the results file name
        import heavy_library                   # import heavy dependencies here, not at module top

    def fit(self, train: Sequence[Email], val: Sequence[Email]) -> TrainingCost:
        # Train on `train`; tune only on `val`. Zero-shot models just return TrainingCost().
        return TrainingCost(wall_clock_s=..., peak_memory_mb=..., hardware="M4 MPS",
                            cost_usd=0.0, n_train_emails=len(train))

    def predict(self, emails: Sequence[Email]) -> list[Prediction]:
        # Exactly one Prediction per email, in order.
        # score: 0..1, higher = more likely phishing.
        # On any failure (bad output, timeout, refusal): score=None, error="why". Never invent a score.
        return [Prediction(score=..., latency_ms=..., cost_usd=...) for e in emails]
```

Then:

- **Config:** add `configs/models/<name>.toml` with `class = "phishbench.detectors.<module>:<Class>"`, its `params`, `data_allowed`, and `[eval]` warmup and batch size. Use [`configs/models/laya.toml`](configs/models/laya.toml) as the template. Pin exact model versions or IDs here.
- **Dependencies:** add an optional extra in `pyproject.toml` (like `laya`), not a core dependency.
- **Tests:** unit tests with a fake backend so they run without the model (see `FakeRouter` in [`tests/test_laya.py`](tests/test_laya.py)), plus an integration test that skips itself when the dependency isn't installed.

### Running a model on the splits

```bash
# Models that learn (TF-IDF + LR, fine-tuned encoders): full splits
python scripts/evaluate.py --model configs/models/<name>.toml --splits data/splits/champa2023-v1 \
    --train train --val val --test test

# Every model, including Qwen and the frontier LLMs: the shared LLM subsample (the headline comparison)
python scripts/evaluate.py --model configs/models/<name>.toml --splits data/splits/champa2023-v1 \
    --train train --val val_llm --test test_llm
```

### Rules that keep results comparable

- **Never tune on `test` or `test_llm`.** Thresholds, hyperparameters, prompts and few-shot examples are chosen on `val` (or `val_llm` for the LLMs); few-shot examples come from `train`.
- **Never edit a frozen split set.** If the data has to change, build a new version (`--version champa2023-v2`) and rerun every model on it.
- **Don't drop failures.** Return them as errors; the harness counts them as wrong.
- **Same input for every model:** sender, subject, body and URLs as loaded. Each model may truncate to its own context limit.
- The full rules are in the plan's [Evaluation protocol](docs/PLAN.md#evaluation-protocol).

### Notes for the remaining phases

| Phase | Notes |
| :--- | :--- |
| 2 · Baselines | Handwritten rules must output a 0–1 score (a weighted count of triggered rules), not just yes/no. TF-IDF + LR needs scikit-learn as a new extra. The cross-source test isn't scripted yet: it retrains TF-IDF + LR (and one encoder in Phase 3) once per held-out source. |
| 3 · Encoders | Fit Laya's signal weights on `val`; the per-email signals are already saved in each prediction's `details`. Check whether laya supports fine-tuning. DeBERTa-v3 reads only 512 tokens. Fine-tuning large models may not fit a 16 GB M4. Laya warns that some of its confidences are uncalibrated. |
| 4 · SLM | Qwen ~4B via MLX, scored from the probability of the "phishing" answer token. `val_llm` gives only ~14 false positives at 1%; consider a larger validation subsample for Qwen, since it runs locally. |
| 5 · Frontier LLMs | Built: `detectors/claude.py`, `detectors/openai_llm.py` (shared base `detectors/llm_base.py`, prompt in `prompts.py`), 7 configs in `configs/models/`. All calls go through Amazon Bedrock with one key in `AWS_BEARER_TOKEN_BEDROCK`. Responses are cached in `data/cache/llm/`, so reruns are free; each config has a cost cap. Run order and budget: [docs/phase5-plan.md](docs/phase5-plan.md). |
| 6 · Cascade and report | Build the cascade as a `Detector` that wraps the tier detectors, so it goes through the same harness. Choose each tier's model and the hand-off thresholds on `val_llm`. |
