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

Early stage. The repo currently holds a proof-of-concept Laya detector (`src/detector.py`), a 7-email sample set (`data/benchmark_emails.json`) and a POC script. No benchmark results exist yet.

The 7 sample emails are a smoke test only: the detector's weights and threshold were hand-picked on these same emails, so its results on them are not evidence of accuracy.

Next up is Phase 0 of the plan: a shared detector interface, the evaluation harness and fixes to the Laya detector.

---

## Laya

**Laya** is an open-weights System 1 decision model built on a **ModernBERT-large** encoder (~421M parameters).

- **Non-autoregressive:** it does not generate text token by token, so there is no output to parse.
- Answers several questions about an email (urgency, credential requests, payment pressure, attachments, overall classification) as probabilities in a single forward pass.

POC hardware: MacBook Air (Apple M4, 10 CPU cores, 10 GPU cores, 16 GB unified memory), PyTorch on Apple Metal (`MPS`). Early POC latency was ~55–160 ms per email; this has not yet been measured with the planned harness (warmup, p50/p95, GPU memory).

---

## Running the POC

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python poc_system1_test.py
```
