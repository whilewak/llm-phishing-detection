# Progress

As of 2026-11-23. Phases 0–5 are complete and benchmarked on the frozen splits. Phase 6 (cascade and report) is in progress.

| Phase | Status | Gate |
| --- | --- | --- |
| Planning | Done | Plan agreed and written to `docs/PLAN.md` |
| 0 · Foundations | Done | Met: Laya ran end to end on the 7 sample emails and wrote a results file |
| 1 · Combined dataset and splits | Done | Met: splits frozen and hashed, dataset version recorded, rebuild verified |
| 2 · Baselines (rules, TF-IDF + LR) | Done | Met: rules and TF-IDF+LR benchmarked on test_llm, full test, and cross-source |
| 3 · Encoders (ModernBERT, DeBERTa-v3, Laya) | Done | Met: all 3 encoder models benchmarked on test_llm, Laya calibrated on validation |
| 4 · SLM (Qwen3.5-4B) | Done | Met: corrected zero-shot and eight-shot scored on frozen validation/test LLM splits; earlier zero-shot/four-shot results retained |
| 5 · Frontier LLMs (Claude Sonnet 5.5, Haiku 5.5) | Done (PR #2, draft) | Met: all 4 configs scored on `test_llm`, thresholds from `val_llm` |
| 6 · Cascade and report | In progress (branch `phase-6-cascade`) | Cascade built, tuned and scored on `test_llm` and the synthetic set; Laya not scored, Qwen not yet in the cascade; report not yet reviewed |

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

- **DeBERTa-v3-base (`TransformerEncoderDetector`)**:
  - Fine-tuned with explicit FP32 `.float()` casting to avoid disentangled attention gradient overflow on MPS/CPU.
  - Results (`test_llm`): **88.2% Recall @ 1% FPR** (95% CI: 84.3%–91.4%), **PR-AUC 0.987**, **0 errors**, p50 latency **328.3 ms**.
  - Achieved the highest recall at 1% FPR among all local dense encoders, balancing strong accuracy with 328 ms local throughput.

## Phase 4 · SLM

- Added Qwen3.5-4B 4-bit MLX detector, pinned to revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`; its initial configs used zero and four shots.
- The initial runs used Qwen's own task text and deterministic balanced train-only demonstrations. Next-token scores `P(A | A or B)` (A = phishing), thinking disabled, float32 log-softmax, 8,192-token total cap, and persistent response caching remain. Raw answer log probabilities and mass, truncation, and example IDs are retained.
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

- Four-shot demonstrations (two per class, selected by the pinned seed): `spamassassin:5207`, `trec06:3416`, `trec07:14431`, `trec07:47666`. Their IDs are checked to belong to training and recorded in every prediction. Qwen and Claude use different task text and few-shot pools; their comparison includes this prompt difference.
- **Corrected comparison completed:** Qwen's current zero-shot and eight-shot configs use Claude's task wording, `<email>` rendering, and the same frozen eight training examples from `configs/prompts/fewshot-champa2023-v1.json`. Qwen retains A/B next-token scoring. The earlier zero-shot/four-shot results above remain unchanged and are not comparable as a controlled shot-count test.
- Corrected results: [zero-shot](../results/qwen3.5-4b-0shot-shared-v1_20261123T113259Z.json) and [eight-shot](../results/qwen3.5-4b-8shot-shared-v1_20261123T141151Z.json). Both scored 2,500 validation and 2,500 test emails with zero errors and zero cache hits. Zero-shot recall at 1% FPR is **71.11%** (95% CI 65.10–77.07%), PR-AUC **0.9711**, p50 **1.119 s**. Eight-shot recall is **70.75%** (95% CI 61.12–75.18%), PR-AUC **0.9740**, p50 **1.097 s**. Eight shots slightly improve PR-AUC but not recall at 1% FPR. Both record $0 API cost; local electricity and hardware are excluded. The result metadata marks the Git worktree as dirty because these runs preceded the code commit.
- Headline recall at 1% FPR describes the test ranking curve. It is separate from performance at the threshold chosen on validation; that threshold does not guarantee a 1% FPR on unseen test mail. Scores with ties move together, without interpolation.
- The shared harness's raw 0.1% FPR metric is retained but marked `reportable: false` for these subsample runs; the subsample is too small to support it.
- **Timing conditions:** Apple M3, 16 GB, with memory pressure observed during the zero-shot test. Idle-sleep prevention was attached near zero-shot prediction 4,714 and remained active for the rest of both runs; it ended with the benchmark process. Treat timings as preliminary local observations. GPU figures are MLX process lifetime peaks; both variants ran sequentially in one process. Neither result used response cache replay; prompt KV caching was enabled in both.

## Phase 5 · Frontier LLMs

Branch `phase-5-frontier-llms`. Plan and sources: [phase5-plan.md](phase5-plan.md).

- **Models:** Claude Sonnet 5.5 ($2/$10 per 1M tokens) and Claude Haiku 5.5 ($0.10/$0.50), through Anthropic's API. OpenAI and Gemini were dropped (no API credit).
- **Configs (4):** Sonnet and Haiku, each zero- and few-shot.
- **Shared prompt** (`src/phishbench/prompts.py`): Claude task instructions and answer formatting; the corrected Qwen run uses its task wording and email rendering with A/B answers. The prompt marks email as untrusted data and caps bodies at 32,000 characters (~8k tokens).
- **Few-shot pool:** 8 train emails (4 phishing incl. 1 Nazario and 1 Nigerian Fraud, 4 legitimate), frozen in `configs/prompts/fewshot-champa2023-v1.json` by `scripts/build_fewshot.py`. Fixed rather than chosen on `val_llm`, which is too small to compare pools.
- **Detectors:** `llm_base.py` (concurrency, response cache in `data/cache/llm/`, cost accounting, per-run cost cap, JSON and optional logprob scoring; Qwen uses its own MLX token scorer) and `claude.py`. Refusals and unparseable output are errors; throttling and server errors are errors too but are not cached, so a rerun retries them. No refusal fallback, since it would answer with a different model.
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

## Phase 6 · Cascade and report

Branch `phase-6-cascade`. Status as of 2026-11-23: the cascade is tuned and scored on `test_llm` and on the synthetic set with rules, TF-IDF, both encoders and Claude. Laya is not scored and Qwen is not yet in the cascade (see "Still to do").

### What was built

- **Saved scores** (`scripts/dump_scores.py` → `results/scores/<model>__<split>.json`): every model's per-email score, latency and cost on `val_llm`, `test_llm` and the synthetic set. Claude scores came from the response cache ($0 new spend). The cascade is tuned and evaluated from these files, so no model is re-run or re-paid for.
- **Cascade detector** (`detectors/cascade.py`): each tier but the last has a `low` and a `high` hand-off threshold; at or below `low` it answers legitimate, at or above `high` phishing, otherwise (or on an error or refusal) the email escalates. Latency and cost are summed over the tiers visited. `detectors/replay.py` replays saved scores so the cascade goes through the normal harness.
- **Tuning** (`scripts/tune_cascade.py`, logic in `cascade_tuning.py`), on `val_llm` only: per tier, the candidate with the best validation recall at 1% FPR (the cheapest one if several are within one point); then a grid search over hand-off thresholds for the most recall with the cascade's FPR ≤ 1% and ≤ 10% of emails reaching the top tier. Writes `configs/models/cascade.toml` and `results/cascade_tuning.json`.
- **Report** (`scripts/make_report.py` → `results/report/`): summary table, accuracy-vs-cost and accuracy-vs-latency plots, and the cascade's recall-vs-escalation curve.
- **Published encoder weights:** `configs/models/deberta-hf.toml` and `modernbert-hf.toml` load the Phase 3 models from Hugging Face (`iceninja3/phishing-deberta-v3-base`, `iceninja3/phishing-modernbert-base`) at pinned revisions (checksums verified).

### Result on `test_llm` (thresholds fixed on `val_llm`)

Tiers chosen on validation recall at 1% FPR: TF-IDF + LR (0.935, over rules 0.037) → DeBERTa-v3 (0.890, over ModernBERT 0.856) → Claude Sonnet 5.5 zero-shot (0.978; few-shot's 0.980 is within a point and costs more). No Tier 3: Qwen has no saved validation scores. Laya was not scored (see "Still to do").

| | Recall | FPR | Emails reaching Sonnet | $ per 1M emails | Mean latency |
| --- | --- | --- | --- | --- | --- |
| Cascade: TF-IDF → DeBERTa → Sonnet | **0.986** | 1.16% | 8.5% | $149 | 228 ms |
| Two tiers: TF-IDF → Sonnet | 0.990 | 0.94% | 9.9% | $150 | 253 ms |
| Two tiers: DeBERTa → Sonnet | 0.971 | 1.30% | 8.5% | $172 | 315 ms |
| Sonnet 5.5 zero-shot alone | 0.979 | 0.94% | 100% | $1,638 | 2,255 ms |
| TF-IDF + LR alone | 0.933 | 0.80% | — | $0 | <1 ms |

- **The plan's target is met on this dataset:** the three-tier cascade keeps 100.6% of Sonnet's recall with 8.5% of emails escalated, at about 9% of Sonnet's cost. TF-IDF answers 80.4% of emails, DeBERTa 11.1%.
- Its false-positive rate lands at 1.16% on test (16 false positives against a budget of 13), having been 0.94% on validation.
- **The encoder tier adds nothing measurable here:** the two-tier TF-IDF → Sonnet cascade does as well or better at the same cost.
- Through the harness (`results/cascade_*.json`): recall at 1% FPR 0.984 (95% CI 0.897–0.992), PR-AUC 0.996.
- The published encoder weights reproduce Phase 3 exactly on this machine: DeBERTa-v3 0.882 (0.843–0.914), ModernBERT 0.820 (0.745–0.891). Latency here is 110 ms and 28 ms per email (batch 16, M4 MPS), against the 328 ms and 200 ms reported in Phase 3 on another machine.
- With the top tier capped at 3% of emails the cascade still reaches 0.983 recall ($43 per 1M); at 1% it reaches 0.970 (`results/report/cascade_curve.png`).
- Full table: `results/report/summary.md`.

### Synthetic test set (`data/splits/synthetic-v1`)

300 emails written by models that are not being benchmarked: 150 by ChatGPT and 150 by Grok, each 75 phishing and 75 legitimate. Prompts were pasted into the chat apps by hand.

- **Design:** `scripts/build_synthetic_prompts.py` draws 300 specs from a fixed grid (6 phishing subtypes, 25 emails each; 11 legitimate categories, 6 of them deliberately tricky such as genuine password resets and urgent requests from a manager; tone, length, writing quality, organisation type and, for phishing, how detectable). Specs, prompts and raw replies are in `data/synthetic/`.
- **Safety and shortcuts:** every organisation is fictional and every domain ends in `.example`, in both classes, so the domain convention cannot give the label away.
- **Checks** (`scripts/build_synthetic.py`): all 300 present, none flagged for outside domains, placeholders, in-email disclaimers or short bodies. Labels come from the spec. `source` is the writer, so results break down per writer.
- **Edits to the raw text:** ChatGPT addressed five emails to the project member's real name; those were replaced with fictional names. Nothing else was edited by hand.
- **Not hand-checked by a person.** The LLM label audit and the replacements it led to are described below.
- Grok wrote shorter than asked (median 139 words for the 220–350 band; ChatGPT 271).

#### Label audit and the 9 replaced emails

The first build of the set is kept, frozen, as `data/splits/synthetic-old`. On 2026-11-23 six Claude subagents each read 100 of its emails, so every email was read twice in different groupings. They saw the spec and the email, never a detector's score. Verdicts with reasons and quoted evidence: `data/synthetic/review.json`; the 15 doubted emails in full: `data/synthetic/review_flagged.md`.

| Outcome on `synthetic-old` | Emails |
| --- | ---: |
| Both reviewers: fits its label | 278 |
| Label doubted by both reviewers | 9 |
| Label doubted by one reviewer | 6 |
| Label fine, quality note only (off-category, muddled payer/payee roles) | 7 |

- All 15 doubted emails were labelled phishing: "malicious attachment" (9) and CEO fraud (6) emails with nothing fraudulent in the text. No legitimate email was doubted.
- **Rule, fixed before any rescoring** (`data/synthetic/replacements.json`): an email is replaced only if both reviewers doubted its label. The same writer rewrites it to the same spec plus one rule, that the fraud must be identifiable from the text. That gave 9 replacements (3 ChatGPT, 6 Grok); the 6 doubted by one reviewer stay.
- The replacements were reviewed the same way, mixed in with untouched emails (`data/synthetic/review_replacements.json`). Two of Grok's were doubted by both reviewers again and were rewritten once more with a stricter prompt; both reviewers passed the rewrites. Final state of the 9: 6 passed by both reviewers, 3 (ChatGPT's, where the only tell is a suffixed sender domain) passed by one and called ambiguous by the other.
- `synthetic-v1` is the 291 unchanged emails plus those 9. Prompts for both rounds: `data/synthetic/prompts/replacements/`.
- **Limits:** the reviewers are Claude models, the same family as two of the detectors, and the set was changed after first results were seen. Both sets are reported below so the effect is visible; it is small for every model except the frontier ones, whose recall rises by about 3 points.

#### Results on `synthetic-v1` (2026-11-24, thresholds unchanged from `val_llm`)

Full tables and plots: `results/report/summary_synthetic.md` and `summary_synthetic-old.md`, `pareto_*_synthetic*.png`.

| Model | Recall at 1% FPR (95% CI) | Recall / FPR at validation threshold | PR-AUC | $ per 1M emails | Recall at 1% FPR on `synthetic-old` |
| --- | --- | --- | --- | ---: | ---: |
| Sonnet 5.5 few-shot | 0.980 (0.926–1.000) | 1.000 / 10.67% | 0.999 | $1,804 | 0.953 |
| Sonnet 5.5 zero-shot | 0.980 (0.934–1.000) | 1.000 / 11.33% | 0.999 | $1,001 | 0.953 |
| Haiku 5.5 zero-shot | 0.960 (0.779–0.993) | 0.960 / 0.67% | 0.998 | $52 | 0.940 |
| Haiku 5.5 few-shot | 0.940 (0.897–1.000) | 0.993 / 5.33% | 0.997 | $102 | 0.913 |
| TF-IDF + LR | 0.247 (0.041–0.345) | 0.373 / 4.00% | 0.806 | $0 | 0.240 |
| Cascade (TF-IDF → DeBERTa → Sonnet) | 0.247 (0.041–0.345) | 0.660 / 8.00% | 0.820 | $306 | 0.240 |
| Rules | 0.113 (0.022–0.269) | 0.340 / 8.67% | 0.703 | $0 | 0.113 |
| ModernBERT | 0.073 (0.000–0.221) | 0.233 / 4.00% | 0.745 | $0 | 0.073 |
| DeBERTa-v3 | 0.033 (0.007–0.106) | 0.073 / 2.67% | 0.661 | $0 | 0.033 |

Cascades with the `val_llm` thresholds (`transfer_to_synthetic` in `results/cascade_tuning.json`):

| | Recall | FPR | Emails reaching Sonnet | Recall on `synthetic-old` |
| --- | --- | --- | --- | --- |
| TF-IDF → DeBERTa → Sonnet | 0.660 | 8.0% | 32.0% | 0.667 |
| TF-IDF → Sonnet | 0.833 | 12.7% | 41.3% | 0.840 |
| DeBERTa → Sonnet | 0.413 | 2.7% | 24.7% | 0.407 |
| Sonnet 5.5 zero-shot alone | 1.000 | 11.3% | 100% | 0.987 |

- **The cascade does not transfer.** It catches 66% of phishing here against 100% for Sonnet alone. Every tier trained on the old corpora clears phishing as legitimate before Sonnet sees it, and adding the encoder tier makes it worse, not better.
- **Every trained model collapses:** TF-IDF + LR from 0.937 on `test_llm` to 0.247, DeBERTa-v3 from 0.882 to 0.033, ModernBERT from 0.820 to 0.073. The fine-tuned encoders do worse than the hand-written rules (0.113). They learned the old corpora, not phishing. Replacing the 9 emails changes none of this.
- **The frontier models hold up on ranking** (PR-AUC 0.997–0.999, no refusals; Haiku had 49–89 refusals on `test_llm`), but **their `val_llm` thresholds are too low for this set**: Sonnet zero-shot catches every phishing email and flags 17 of 150 legitimate ones, all from the tricky categories (mostly genuine security notices and genuine invoices). In the audit, both reviewers passed 16 of those 17 as legitimate.
- **Haiku 5.5 zero-shot is the cheap model that survives:** 0.960 recall at 0.67% FPR at its validation threshold, for $52 per million emails.
- On the 9 replacements, Sonnet scores 0.62–0.95 and Haiku 0.62–0.92; TF-IDF scores 8 of them below 0.34 and DeBERTa 8 of them below 0.08.
- With 150 legitimate emails, 1% FPR allows one false positive, so recall at 1% FPR is coarse on this set.
- API spend on the synthetic sets: about $0.92 in total ($0.89 for `synthetic-old`, $0.03 for the replacements). Opus was not run.

### Still to do

- **Laya as a Tier 2 candidate.** Scoring on `val_llm` and `synthetic-v1` started 2026-11-24 on this machine; results not in yet. On `test_llm` in Phase 3 it was behind DeBERTa-v3 (0.802 vs 0.882) and about 4.5 times slower.
- **Qwen (Phase 4) as Tier 3.** Merged from `main` on 2026-11-24. Its `test_llm` scores are imported from the committed results files (`scripts/import_result_scores.py`) and appear in `results/report/summary.md`: zero-shot 0.711, eight-shot 0.708 recall at 1% FPR. It is not in the cascade yet: tuning needs its per-email `val_llm` scores, and the results files hold only test predictions (the response cache `data/cache/qwen-shared-v1` is git-ignored). It also has no scores on `synthetic-v1`.
- A person reading the 9 replacement emails and the 6 phishing emails one reviewer doubted. Paired significance tests (McNemar) between the cascade and Sonnet. A cheap first tier that survives the synthetic set (for example Haiku zero-shot, 0.933 recall at 0.67% FPR there) is the obvious next cascade to try.

### Reproduce

```bash
S=data/splits/champa2023-v1
python scripts/dump_scores.py --model configs/models/tfidf_lr.toml --splits $S --train train --val val_llm --score val_llm test_llm
python scripts/dump_scores.py --model configs/models/claude-sonnet-5-5-zeroshot.toml --splits $S --val val_llm --score val_llm test_llm   # from the cache
python scripts/tune_cascade.py
python scripts/evaluate.py --model configs/models/cascade.toml --splits $S --val val_llm --test test_llm
python scripts/make_report.py
# synthetic set (for the first build, use data/splits/synthetic-old and the split name synthetic-old)
python scripts/dump_scores.py --model configs/models/<name>.toml --splits $S --score-splits data/splits/synthetic-v1 --score synthetic
python scripts/evaluate.py --model configs/models/cascade.toml --splits $S --val val_llm --test-splits data/splits/synthetic-v1 --test synthetic
python scripts/make_report.py --splits data/splits/synthetic-v1 --split synthetic
```

## Tests

86 unit tests pass with one optional integration skipped (Qwen GPU), run as named modules with `HF_HUB_OFFLINE=1`: metrics, harness, rules, TF-IDF, Laya calibration, encoders, LLM detectors, Qwen, cascade, replay and threshold tuning. The Laya integration test downloads the model and was not run.

## Known issues to carry forward

- **Few true-phishing emails in the test LLM subsample:** 57 (20 Nazario, 37 Nigerian), so that recall has a wide confidence interval for Qwen and Claude. The full test split has about 590.
- **Small validation LLM subsample:** about 14 false positives at 1%, so Qwen's and Claude's thresholds will be noisy. Retained for Phase 4; threshold uncertainty is documented in its results.
- **Possible source shortcuts:** sender addresses (e.g. `@ENRON.com`) and the tokenised text of Enron and Ling may let models recognise the source; the cross-source test is meant to catch this.
- **Laya calibration:** its checkpoint warning matters for the cascade's hand-offs.
- **Safety refusals on phishing emails:** Haiku 5.5 declines 3.5–8% of phishing emails in `test_llm` (category `bio`), and Opus 5.5 declines 7 of 20 Nazario credential-phishing emails (category `cyber`); Sonnet 5.5 declined none. They count as misses, so Haiku's recall partly measures refusals. Not fixable by prompting without changing the shared prompt; noted for the cascade (a refusal could escalate to the next tier instead).
- **0–100 scores tie heavily:** nearly every LLM score shares its value with another email (scores cluster at a few values like 5 and 95), so recall at 1% FPR moves in steps. Qwen also has a high tie share (about 99.8%) in its saved results.
- **Smoke-test result:** `results/laya-english_20261114T080908Z.json` (7 sample emails) is committed and retained only as a smoke test.

## Next

Finish Phase 6 ("Still to do" above), then the presentation.
