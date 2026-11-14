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
| 4 · Frontier LLM | Claude Sonnet and Haiku | API |

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
