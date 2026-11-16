# Progress

As of 2026-11-14. Phases 0 and 1 of [PLAN.md](PLAN.md) are done; no model has been benchmarked yet. Phases 0 and 1 are not committed yet (last commit: `bb00745`, the plan and README).

| Phase | Status | Gate |
| --- | --- | --- |
| Planning | Done | Plan agreed and written to `docs/PLAN.md` |
| 0 · Foundations | Done | Met: Laya ran end to end on the 7 sample emails and wrote a results file |
| 1 · Combined dataset and splits | Done | Met: splits frozen and hashed, dataset version recorded, rebuild verified |
| 2 · Baselines (rules, TF-IDF + LR) | Not started | |
| 3 · Encoders (ModernBERT, DeBERTa-v3, Laya) | Not started | |
| 4 · SLM (Qwen ~4B) | Not started | |
| 5 · Frontier LLMs (Claude Sonnet 5.5, Haiku 5.5) | Built and unit-tested; not run yet (needs an Anthropic API key) | |
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

## Phase 5 · Frontier LLMs (in progress)

Branch `phase-5-frontier-llms`. Plan and sources: [phase5-plan.md](phase5-plan.md).

- **Models:** Claude Sonnet 5.5 ($2/$10 per 1M tokens) and Claude Haiku 5.5 ($0.10/$0.50), through Anthropic's API. OpenAI and Gemini were dropped (no API credit).
- **Configs (4):** Sonnet and Haiku, each zero- and few-shot.
- **Shared prompt** (`src/phishbench/prompts.py`): one set of task instructions for every LLM (Qwen too), the email marked as untrusted data, bodies capped at 32,000 characters (~8k tokens).
- **Few-shot pool:** 8 train emails (4 phishing incl. 1 Nazario and 1 Nigerian Fraud, 4 legitimate), frozen in `configs/prompts/fewshot-champa2023-v1.json` by `scripts/build_fewshot.py`. Fixed rather than chosen on `val_llm`, which is too small to compare pools.
- **Detectors:** `llm_base.py` (concurrency, response cache in `data/cache/llm/`, cost accounting, per-run cost cap, JSON and logprob scoring; the logprob scorer is for Qwen) and `claude.py`. Refusals and unparseable output are errors; throttling and server errors are errors too but are not cached, so a rerun retries them. No refusal fallback, since it would answer with a different model.
- **Temperature:** current Claude models reject it, so Claude runs with thinking off at effort `low`. Reproducibility comes from the response cache.
- **Budget:** about $27 for all 4 configs; caps sum to $55.

To run (needs `ANTHROPIC_API_KEY`): smoke-test each config on `data/benchmark_emails.json`, check real token use on 50 `val_llm` emails, then run each config with `--val val_llm --test test_llm`.

## Tests

43 unit tests, all passing, including the real-Laya integration test. They cover metrics, the harness (with stub detectors), the loader, Laya response handling, cleaning, deduplication, and split building and verification.

## Known issues to carry forward

- **Few true-phishing emails in the test LLM subsample:** 57 (20 Nazario, 37 Nigerian), so that recall has a wide confidence interval for Qwen and Claude. The full test split has about 590.
- **Small validation LLM subsample:** about 14 false positives at 1%, so Qwen's and Claude's thresholds will be noisy. Revisit before Phase 4.
- **Possible source shortcuts:** sender addresses (e.g. `@ENRON.com`) and the tokenised text of Enron and Ling may let models recognise the source; the cross-source test is meant to catch this.
- **Laya calibration:** its checkpoint warning matters for the cascade's hand-offs.
- **Untracked smoke-test result:** `results/laya-english_20261114T080908Z.json` (7 sample emails) may not be worth committing.

## Next

Phase 2: handwritten rules and TF-IDF + logistic regression, scored on the frozen splits.
