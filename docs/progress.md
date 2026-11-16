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
| 5 · Frontier LLMs (Claude Sonnet 5.5, Haiku 5.5) | Done (PR #2, draft) | Met: all 4 configs scored on `test_llm`, thresholds from `val_llm` |
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

## Phase 5 · Frontier LLMs

Branch `phase-5-frontier-llms`. Plan and sources: [phase5-plan.md](phase5-plan.md).

- **Models:** Claude Sonnet 5.5 ($2/$10 per 1M tokens) and Claude Haiku 5.5 ($0.10/$0.50), through Anthropic's API. OpenAI and Gemini were dropped (no API credit).
- **Configs (4):** Sonnet and Haiku, each zero- and few-shot.
- **Shared prompt** (`src/phishbench/prompts.py`): one set of task instructions for every LLM (Qwen too), the email marked as untrusted data, bodies capped at 32,000 characters (~8k tokens).
- **Few-shot pool:** 8 train emails (4 phishing incl. 1 Nazario and 1 Nigerian Fraud, 4 legitimate), frozen in `configs/prompts/fewshot-champa2023-v1.json` by `scripts/build_fewshot.py`. Fixed rather than chosen on `val_llm`, which is too small to compare pools.
- **Detectors:** `llm_base.py` (concurrency, response cache in `data/cache/llm/`, cost accounting, per-run cost cap, JSON and logprob scoring; the logprob scorer is for Qwen) and `claude.py`. Refusals and unparseable output are errors; throttling and server errors are errors too but are not cached, so a rerun retries them. No refusal fallback, since it would answer with a different model.
- **Temperature:** current Claude models reject it, so Claude runs with thinking off at effort `low`. Reproducibility comes from the response cache.
- **Budget:** about $27 for all 4 configs; caps sum to $55.

### Results (2026-11-16, `test_llm`: 2,500 emails, 1,118 phishing; thresholds chosen on `val_llm`)

| Config | Recall at 1% FPR (95% CI) | PR-AUC | Errors | p50 / p95 latency | $ per 1M emails | Run cost |
| --- | --- | --- | --- | --- | --- | --- |
| Sonnet 5.5 zero-shot | **0.979** (0.968–0.986) | 0.998 | 0 | 1.7 s / 3.6 s | $1,638 | $8.31 |
| Sonnet 5.5 few-shot | 0.978 (0.967–0.990) | 0.998 | 0 | 1.6 s / 2.1 s | $1,988 | $10.19 |
| Opus 5.5 zero-shot | 0.976 (0.967–0.986) | 0.995 | 7 refusals | 1.9 s / 2.7 s | $3,277 | $16.57 |
| Haiku 5.5 few-shot | 0.898 (0.869–0.917) | 0.965 | 49 refusals | 1.0 s / 1.7 s | $120 | $0.61 |
| Haiku 5.5 zero-shot | 0.807 (0.738–0.864) | 0.943 | 89 refusals | 1.2 s / 1.9 s | $85 | $0.41 |

- Every API response is saved in `results/llm-response-cache-champa2023-v1.tar.gz` (3.3 MB, no email text); unpacked into `data/cache/`, it reproduces all five results with no API key and $0 spent.
- Total new API spend: about **$36.20** (Sonnet and Haiku $19.60 including smoke tests and a run interrupted to parallelise; Opus $16.60), against a $55 cap. Rerunning a finished config makes 0 API calls.
- **Sonnet** is the strongest single model so far and the current Tier 4 candidate. Few-shot doesn't help it.
- **Opus 5.5 adds nothing over Sonnet** (0.976 vs 0.979, overlapping CIs) at twice the cost. It refused 7 phishing emails (category `cyber`), all of them Nazario credential-phishing emails: 7 of the 20 Nazario emails in `test_llm`. Its "true phishing" recall is therefore 0.88, against 1.0 for Sonnet. At effort `low` it wrote no hidden reasoning (~23 output tokens per email). "True phishing" recall (Nazario + Nigerian, 57 emails) is 1.0 for both Sonnet configs.
- **Haiku is held back by refusals, not only by accuracy.** 88 of the zero-shot errors and 48 of the few-shot ones are phishing emails that Haiku's safety classifier declined (category `bio`, likely pharmacy spam), and each counts as a miss. Few-shot halves the refusals and lifts recall by 9 points.
- Calibration error (ECE) is 0.03–0.04 for all four. Precision at the chosen threshold is about 0.99 for all four. 0.28% of emails hit the 32,000-character input cap.

To rerun (needs `ANTHROPIC_API_KEY`; cached responses are reused for free): see "Running it" in [phase5-plan.md](phase5-plan.md).

## Tests

51 unit tests, all passing, covering metrics, harness, rules, TF-IDF, Laya calibration, and encoder training/inference.

## Known issues to carry forward

- **Few true-phishing emails in the test LLM subsample:** 57 (20 Nazario, 37 Nigerian), so that recall has a wide confidence interval for Qwen and Claude. The full test split has about 590.
- **Small validation LLM subsample:** about 14 false positives at 1%, so Qwen's and Claude's thresholds will be noisy. Revisit before Phase 4.
- **Possible source shortcuts:** sender addresses (e.g. `@ENRON.com`) and the tokenised text of Enron and Ling may let models recognise the source; the cross-source test is meant to catch this.
- **Laya calibration:** its checkpoint warning matters for the cascade's hand-offs.
- **Safety refusals on phishing emails:** Haiku 5.5 declines 3.5–8% of phishing emails in `test_llm` (category `bio`), and Opus 5.5 declines 7 of 20 Nazario credential-phishing emails (category `cyber`); Sonnet 5.5 declined none. They count as misses, so Haiku's recall partly measures refusals. Not fixable by prompting without changing the shared prompt; noted for the cascade (a refusal could escalate to the next tier instead).
- **0–100 scores tie heavily:** nearly every LLM score shares its value with another email (scores cluster at a few values like 5 and 95), so recall at 1% FPR moves in steps. Qwen's logprob scores in Phase 4 will not have this problem.
- **Untracked smoke-test result:** `results/laya-english_20261114T080908Z.json` (7 sample emails) may not be worth committing.

## Next

Complete DeBERTa-v3 evaluation, then proceed to Phase 4 (SLM / Qwen ~4B).
