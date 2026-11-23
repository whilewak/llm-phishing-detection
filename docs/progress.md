# Progress

As of 2026-11-16. Phases 0 and 1 of [PLAN.md](PLAN.md) are done and committed. Phase 4 is complete on `phase-4-qwen`: zero-shot and four-shot benchmarks are saved and verified. Phases 2, 3 and 5 remain unstarted.

| Phase | Status | Gate |
| --- | --- | --- |
| Planning | Done | Plan agreed and written to `docs/PLAN.md` |
| 0 · Foundations | Done | Met: Laya ran end to end on the 7 sample emails and wrote a results file |
| 1 · Combined dataset and splits | Done | Met: splits frozen and hashed, dataset version recorded, rebuild verified |
| 2 · Baselines (rules, TF-IDF + LR) | Not started | |
| 3 · Encoders (ModernBERT, DeBERTa-v3, Laya) | Not started | |
| 4 · SLM (Qwen ~4B) | Done | Met: zero/four-shot runs scored on frozen validation/test LLM splits; results retained |
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

## Phase 4 · SLM

- Added Qwen3.5-4B 4-bit MLX detector, pinned to revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, with zero-shot and four-shot configs.
- Shared task text, deterministic balanced train-only demonstrations, next-token scores `P(A | A or B)` (A = phishing), thinking disabled, float32 log-softmax, 8,192-token total cap, and persistent response caching. Raw answer log probabilities and mass, truncation, and example IDs are retained.
- Real zero-shot and few-shot integration checks pass on this machine (Apple M3, 16 GB); benchmarks must be labelled with this hardware rather than the POC's M4.
- Retained frozen `val_llm` for comparability after revisiting validation size. Its 1,382 negatives give about 14 false positives at 1%; threshold uncertainty remains a limitation.
- `scripts/evaluate_qwen.py` runs both variants through the shared harness, preserves original inference latencies on cache replay, and records MLX peak memory and cache use.
- **Gate met:** both variants scored on the unchanged 2,500-email `val_llm` and 2,500-email `test_llm`; test IDs and both content fingerprints checked against the frozen manifest. All 10,000 validation/test predictions completed without errors. No prompt or example selection was changed after seeing test results.
- Results: [zero-shot](../results/qwen3.5-4b-0shot_20261116T125505Z.json) and [four-shot](../results/qwen3.5-4b-4shot_20261116T154514Z.json). These are preliminary results on the combined dataset.

| Measure | Zero-shot | Four-shot |
| --- | ---: | ---: |
| Test recall at 1% FPR | 67.62% | 84.17% |
| 95% bootstrap CI | 63.97–77.98% | 79.96–87.57% |
| PR-AUC | 0.9724 | 0.9790 |
| Recall at validation-chosen threshold | 77.82% | 85.69% |
| FPR at validation-chosen threshold | 1.59% | 1.23% |
| ECE | 0.1120 | 0.0429 |
| Score tie share | 99.80% | 99.84% |
| Test inputs truncated | 9 / 2,500 (0.36%) | 11 / 2,500 (0.44%) |
| p50 / p95 latency | 1.146 / 6.904 s | 1.168 / 6.077 s |
| Throughput (batch size 1) | 0.453 emails/s | 0.483 emails/s |
| MLX peak memory | 3,324.6 MB | 3,387.3 MB |
| Training / marginal inference dollars | $0 / $0 | $0 / $0 |
| Test failures / response cache hits | 0 / 0 | 0 / 0 |

- Four-shot demonstrations (two per class, selected by the pinned seed): `spamassassin:5207`, `trec06:3416`, `trec07:14431`, `trec07:47666`. Their IDs are checked to belong to training and recorded in every prediction. The shared selector and task text are available for Phase 5 to reuse.
- Headline recall at 1% FPR describes the test ranking curve. It is separate from performance at the threshold chosen on validation; that threshold does not guarantee a 1% FPR on unseen test mail. Scores with ties move together, without interpolation.
- The shared harness's raw 0.1% FPR metric is retained but marked `reportable: false` for these subsample runs; the subsample is too small to support it.
- **Timing conditions:** Apple M3, 16 GB, with memory pressure observed during the zero-shot test. Idle-sleep prevention was attached near zero-shot prediction 4,714 and remained active for the rest of both runs; it ended with the benchmark process. Treat timings as preliminary local observations. GPU figures are MLX process lifetime peaks; both variants ran sequentially in one process. Neither result used response cache replay; prompt KV caching was enabled in both.

## Tests

52 tests discovered after the Phase 4 additions: 50 pass in the local environment; the Laya integration and opt-in Qwen integration tests skip by default. The Qwen integration test also passed separately with `PHISHBENCH_QWEN_INTEGRATION=1`, exercising real zero-shot and few-shot inference. Coverage includes metrics, the harness, loading, Laya responses, cleaning, deduplication, split verification, Qwen scoring, truncation, caching, and demonstration isolation.

## Known issues to carry forward

- **Few true-phishing emails in the test LLM subsample:** 57 (20 Nazario, 37 Nigerian), so that recall has a wide confidence interval for Qwen and Claude. The full test split has about 590.
- **Small validation LLM subsample:** about 14 false positives at 1%, so thresholds are noisy. Retained for Phase 4 with this uncertainty documented; revisit for Phase 5 and cascade threshold selection.
- **Possible source shortcuts:** sender addresses (e.g. `@ENRON.com`) and the tokenised text of Enron and Ling may let models recognise the source; the cross-source test is meant to catch this.
- **Qwen score ties and calibration:** nearly all test emails share a score with another email. Together with the small validation subsample, this limits fine threshold placement. Keep the raw log probabilities, tie-group metrics and ECE when building the cascade. Local timing measurements also reflect memory pressure and are not controlled hardware comparisons.
- **Laya calibration:** its checkpoint warning matters for the cascade's hand-offs.
- **Committed smoke-test result:** `results/laya-english_20261114T080908Z.json` (7 sample emails) is retained only as a smoke test.

## Next

Phase 2: handwritten rules and TF-IDF + logistic regression, scored on the frozen splits.
