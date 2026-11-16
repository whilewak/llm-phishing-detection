# Progress

As of 2026-11-14. Phases 0 and 1 of [PLAN.md](PLAN.md) are done; no model has been benchmarked yet. Phases 0 and 1 are not committed yet (last commit: `bb00745`, the plan and README).

| Phase | Status | Gate |
| --- | --- | --- |
| Planning | Done | Plan agreed and written to `docs/PLAN.md` |
| 0 · Foundations | Done | Met: Laya ran end to end on the 7 sample emails and wrote a results file |
| 1 · Combined dataset and splits | Done | Met: splits frozen and hashed, dataset version recorded, rebuild verified |
| 2 · Baselines (rules, TF-IDF + LR) | Done | Met: rules and TF-IDF+LR benchmarked on test_llm, full test, and cross-source |
| 3 · Encoders (ModernBERT, DeBERTa-v3, Laya) | In progress | ModernBERT & Laya complete; DeBERTa-v3 training |
| 4 · SLM (Qwen ~4B) | Not started | |
| 5 · Frontier LLMs (Claude Sonnet, Haiku) | Not started | |
| 6 · Cascade and report | Not started | |

## Planning

- Reviewed the original proof of concept. Main problems: Laya's weights and threshold were tuned on the same 7 emails used to test it, the README's ~180 MB memory figure ignored GPU memory, and nothing could detect a malformed model response.
- Agreed the model lineup: handwritten rules, TF-IDF + logistic regression, fine-tuned ModernBERT and DeBERTa-v3, Laya, Qwen ~4B, Claude Sonnet and Haiku, and a four-tier cascade. Jev, GPT, Gemini, Claude Opus and a dozen other candidates were dropped.
- Agreed the evaluation rules: headline metric is recall at 1% false-positive rate on a shared ~2.5k LLM subsample; failed predictions always count as wrong; Claude gives a 0–100 score, Qwen uses answer-token probabilities; each model reads to its own context limit; cascade target is ≥95% of the best single model's recall with ≤10% of emails reaching the top tier.
- Decided to use one combined public dataset for now; building our own dataset is deferred (bottom of the plan).
- Rewrote the README to match the plan.

## Phase 0 · Foundations

Built the `phishbench` package (`src/phishbench/`), replacing the old `src/detector.py`, `poc_system1_test.py` and `requirements.txt`.

- **Interface** (`types.py`): `Email`, `Prediction`, `TrainingCost` and the `Detector` protocol every model implements.
- **Harness** (`eval/harness.py`): calls `fit`, picks the threshold on validation (0.5 if none), warms up, times prediction, and writes one results JSON per run with data fingerprints, training cost, memory, hardware and package versions.
- **Metrics** (`eval/metrics.py`): recall at 1% and 0.1% false-positive rate, PR-AUC, precision/recall/F1, precision re-weighted to a 1% phishing rate, calibration error, tie share, p50/p95 latency, bootstrap confidence intervals.
- **Laya detector** (`detectors/laya.py`): a missing or malformed Laya answer is now a reported error instead of a silent 0. Still uses the proof of concept's hand-picked weights until Phase 3.
- **CLI** (`scripts/evaluate.py`) driven by per-model TOML configs in `configs/models/`.
- Dependencies moved to `pyproject.toml`; the harness needs only the standard library.

First Laya run (7 sample emails, laya 0.4.31, torch 2.15.0, Apple M4):

| Measure | Result |
| --- | --- |
| Errors | 0 of 7 |
| Ranking | Every phishing email scored above every legitimate one (7 emails prove nothing) |
| Latency | p50 ~415 ms, p95 ~930 ms, ~2 emails/s |
| Memory | ~3.1 GB MPS driver memory (~1.6 GB allocated) + ~1 GB RSS |
| At default threshold 0.5 | Caught 1 of 4 phishing emails; Phase 3 fits the real threshold |

Laya also warned that its checkpoint has invalid calibration settings for some answers, so its confidences may be uncalibrated.

## Phase 1 · Combined dataset and splits

- **Dataset:** [Phishing Email Curated Datasets](https://zenodo.org/records/8339691) (Champa et al., Zenodo, CC-BY-4.0), 9 of 11 CSVs (~400 MB): CEAS-08, Nazario, Nigerian Fraud, SpamAssassin, TREC-05/06/07, Enron, Ling. Downloaded with `scripts/download_data.py`, MD5-checked, stored in `data/raw/` (git-ignored).
- **Labels:** source label 1 (spam or phishing) → phishing. Nazario and Nigerian Fraud are reported separately as "true phishing".
- **Cleaning:** strips header lines, inline message IDs and dates from bodies; caps bodies at 100k characters (77 truncated); dropped 876 TREC rows with blank labels and 2 mailbox artifacts.
- **Deduplication:** removed 5,555 exact and 31,225 near duplicates (MinHash, Jaccard ≥ 0.8), mostly Enron emails also in TREC-05, keeping the raw TREC-05 copy over the tokenised Enron one. Dropped 363 emails in duplicate groups with conflicting labels.
- **Splits** (`data/splits/champa2023-v1`, 70/15/15 stratified by source and label; only id lists and the manifest are committed):

| Split | Emails | Phishing | Legitimate |
| --- | ---: | ---: | ---: |
| Train | 119,755 | 53,598 | 66,157 |
| Validation | 25,660 | 11,484 | 14,176 |
| Test | 25,666 | 11,487 | 14,179 |
| Validation LLM subsample | 2,500 | 1,118 | 1,382 |
| Test LLM subsample | 2,500 | 1,118 | 1,382 |

- `scripts/build_splits.py` builds the splits and refuses to overwrite a frozen set; `--verify` rebuilds and checks against the committed splits (verified). Loading a split checks its ids and content against the manifest.
- The harness now reports recall and false-positive rate per source and for the "true phishing" group; the CLI loads frozen splits with `--splits`.

## Phase 2 · Baselines (Rules & TF-IDF + Logistic Regression)

Implemented and benchmarked both Phase 2 detectors on `test_llm` (2,500 emails), `test` (25,666 emails), and across all 9 held-out cross-source splits.

- **Handwritten Rules (`RulesDetector`)**:
  - Scans across urgency/coercion, credential harvesting, financial pressure/invoicing, suspicious URL patterns (IPs, shorteners, high-risk TLDs), and sender spoofing cues.
  - Results (`test_llm`): **5.5% Recall @ 1% FPR** (95% CI: 4.2%–7.1%), **PR-AUC 0.473**, p50 latency **0.4 ms**.
  - Results (`test` 25.6k emails): **3.6% Recall @ 1% FPR**, PR-AUC 0.472, p50 latency **0.3 ms**.
  - Cross-source performance: Drops to 0%–11% across held-out sources, demonstrating that keyword regexes cannot track template rotations.

- **TF-IDF + Logistic Regression (`TfidfLogisticRegressionDetector`)**:
  - Fitted on 119,755 training emails (50,000 unigram/bigram features).
  - Results (`test_llm`): **93.7% Recall @ 1% FPR** (95% CI: 92.3%–95.1%), **PR-AUC 0.996**, p50 latency **0.5 ms**.
  - Results (`test` 25.6k emails): **95.4% Recall @ 1% FPR**, PR-AUC 0.997, p50 latency **0.3 ms**.
  - Cross-source performance: Collapses to **33.8% on SpamAssassin** and **54.6% on CEAS-08**, exposing severe lexical shortcut learning / overfitting to corpus artifacts.

## Phase 3 · Encoders & Laya System 1

- **Laya System 1 (`LayaDetector`)**:
  - Fitted optimal multi-signal weights on validation using logistic regression.
  - Results (`test_llm`): **80.2% Recall @ 1% FPR** (95% CI: 66.9%–87.9%), **PR-AUC 0.9832**, p50 latency **1,477 ms**.
  - Robust across attack sources (80.0% recall on SpamAssassin, 0.998 PR-AUC).

- **ModernBERT-base (`TransformerEncoderDetector`)**:
  - Fine-tuned on balanced training split; handles native long context and rotary position embeddings.
  - Results (`test_llm`): **82.0% Recall @ 1% FPR** (95% CI: 74.5%–89.1%), **PR-AUC 0.9893** (highest among all semantic models), p50 latency **199.5 ms** (7.4× faster than Laya locally).
  - Robust semantic intent separation without vocabulary shortcuts.

- **DeBERTa-v3-base**:
  - In progress: Initial attempts suffered from FP16 gradient overflow; patched with explicit FP32 `.float()` casting and stabilized learning rate.

## Tests

51 unit tests, all passing, covering metrics, harness, rules, TF-IDF, Laya calibration, and encoder training/inference.

## Next

Complete DeBERTa-v3 evaluation, then proceed to Phase 4 (SLM / Qwen ~4B).
