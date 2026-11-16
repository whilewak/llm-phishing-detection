# LLMs for Phishing Detection: Midterm Report

**Team 16B** · Ronit Barman, Vishal Dandamudi, Kyle Pak, Aneesh Ratnala, Gavin Sonntag
ECE M117 / CS M138, Fall 2026 · Draft of 2026-11-16

> Placeholders are marked **TODO (name)**. Everything not marked TODO comes from the repo (`docs/PLAN.md`, `docs/progress.md`, `results/`).

## 1. Problem area

Phishing email remains one of the most common ways into an organisation: one convincing message can lead to stolen credentials, a fraudulent payment or malware. Filters must catch as much phishing as possible while flagging almost no legitimate mail, because users quickly start ignoring a filter with too many false alarms.

Large language models (LLMs) are now strong phishing detectors, but they are slow (about 2 s per email) and costly (over $1,600 per million emails for the models we tested). Cheaper detectors, such as hand-written rules, classic machine learning and small "System 1" classifiers, run in milliseconds at almost no cost but are less accurate. A real mail system processes millions of emails a day, so the trade-off between **accuracy, latency and cost** decides which detector is practical.

> **TODO (all):** 2–3 citations on LLM phishing detection (e.g. Aziz et al. 2026; Liu et al. 2026 from the intro slides), with full references in section 7.

## 2. Focused topic and proposed solution

**Question:** how much accuracy does each step up in model cost buy, and can a combination of cheap and expensive models match the best model at a fraction of its cost?

**Approach:** benchmark five families of detectors on the same emails, with the same harness and the same metrics, ordered from cheapest to most expensive:

| Tier | Detectors | Runs on |
| --- | --- | --- |
| 1 · Baselines | Hand-written rules; TF-IDF + logistic regression | Laptop CPU |
| 2 · Encoders / System 1 | Laya (ModernBERT-large System 1 model); fine-tuned ModernBERT and DeBERTa-v3 | Laptop GPU (Apple M4) |
| 3 · Small LLM | Qwen ~4B, prompted | Laptop (MLX) |
| 4 · Frontier LLM | Claude Sonnet 5.5, Haiku 5.5, Opus 5.5 | Anthropic API |

**Proposed solution (Phase 6): a cascade.** Every email first goes to the cheapest tier; only emails it is unsure about are escalated to the next tier. Most emails stop early, and only the hardest reach the frontier LLM. Each tier's model and hand-off threshold are chosen on validation data.

**Threat model (brief):** the attacker sends phishing email (credential theft, advance-fee fraud, malicious links) and wants it delivered. The defender sees only the email's sender, subject, body and URLs; no header-authentication results (SPF/DKIM) or reputation lookups. Email text is untrusted: the LLM prompt marks it as data to classify, not instructions to follow.

## 3. Goals and timeline

**Goals**

1. **Detection:** measure recall at a 1% false-positive rate (share of phishing caught while flagging only 1% of legitimate mail) for every detector.
2. **Cost:** measure latency, memory, dollars per million emails and training cost.
3. **System 1 vs. LLM:** test whether fast System 1 / encoder models come close to frontier LLMs.
4. **Cascade:** keep ≥95% of the best single model's recall while sending ≤10% of emails to the top tier.

**Timeline**

| Phase | Work | Owner | Status |
| --- | --- | --- | --- |
| 0 · Foundations | Detector interface, evaluation harness, metrics, tests | Kyle | Done |
| 1 · Dataset + splits | Combined public dataset, cleaning, dedup, frozen splits | Kyle | Done |
| 2 · Baselines | Rules, TF-IDF + LR | Vishal | **TODO (Vishal):** status |
| 3 · Encoders | Laya, ModernBERT, DeBERTa-v3 | Vishal | **TODO (Vishal):** status |
| 4 · Small LLM | Qwen ~4B, zero- and few-shot | Ronit | **TODO (Ronit):** status |
| 5 · Frontier LLMs | Claude Sonnet / Haiku / Opus 5.5 | Aneesh, Gavin | Done |
| 6 · Cascade + report | Tier choice, hand-off thresholds, accuracy–cost charts, final report | All | After midterm |

Phases 2–5 depend only on Phase 1 and run in parallel. Target: Phases 0–5 by the midterm, Phase 6 and the final report by the final presentation.

## 4. Experimental setup (Phases 0–1)

> **TODO (Kyle):** check and expand this section.

**Dataset.** Phishing Email Curated Datasets (Champa et al., Zenodo record 8339691, CC-BY-4.0), 9 corpora: CEAS-08, Nazario, Nigerian Fraud, SpamAssassin, TREC-05/06/07, Enron, Ling. The source label "spam or phishing" is mapped to phishing. Nazario and Nigerian Fraud are phishing/fraud by construction and are also reported separately as "true phishing".

**Cleaning.** Header lines, message IDs and dates are stripped from bodies so no model can recognise the source corpus. Removed 5,555 exact and 31,225 near duplicates (MinHash, Jaccard ≥ 0.8), and 363 emails whose duplicates had conflicting labels. **171,081 emails** remain.

**Splits** (70/15/15, stratified by source and label, frozen and hashed):

| Split | Emails | Phishing | Legitimate |
| --- | ---: | ---: | ---: |
| Train | 119,755 | 53,598 | 66,157 |
| Validation | 25,660 | 11,484 | 14,176 |
| Test | 25,666 | 11,487 | 14,179 |
| Validation LLM subsample | 2,500 | 1,118 | 1,382 |
| Test LLM subsample | 2,500 | 1,118 | 1,382 |

The LLM subsamples keep API cost bounded; **every** model is scored on the test LLM subsample, so the headline table compares all models on the same 2,500 emails.

**Evaluation rules.** Every model implements one `Detector` interface and returns a 0–1 phishing score. Thresholds, prompts and hyperparameters are chosen on validation only, never on test. Unparseable outputs, timeouts and refusals count as errors, and an error is always wrong (a miss if phishing, a false alarm if legitimate). Every metric has a 95% bootstrap confidence interval.

## 5. Preliminary results

### 5.1 All models (test LLM subsample: 2,500 emails, 1,118 phishing)

| Phase | Model | Recall @ 1% FPR | Latency (p50) | $ / 1M emails |
| --- | --- | --- | --- | --- |
| 2 | Rules | **TODO (Vishal)** | | |
| 2 | TF-IDF + LR | **TODO (Vishal)** | | |
| 3 | Laya (System 1) | **TODO (Vishal)** | | |
| 3 | ModernBERT / DeBERTa-v3 | **TODO (Vishal)** | | |
| 4 | Qwen ~4B | **TODO (Ronit)** | | |
| 5 | Claude Sonnet 5.5 (zero-shot) | **0.979** | 1.7 s | $1,638 |
| 5 | Claude Opus 5.5 (zero-shot) | 0.976 | 1.9 s | $3,277 |
| 5 | Claude Haiku 5.5 (few-shot) | 0.898 | 1.0 s | $120 |

### 5.2 Phase 2–4 results

> **TODO (Vishal):** Phase 2 (rules, TF-IDF + LR) and Phase 3 (Laya, ModernBERT, DeBERTa-v3): setup, results table, observations; full-test-split and cross-source results if available.
>
> **TODO (Ronit):** Phase 4 (Qwen ~4B): setup, zero- vs few-shot, results table, observations.

### 5.3 Phase 5: frontier LLMs (Aneesh, Gavin)

**Setup.** Claude Sonnet 5.5, Haiku 5.5 and Opus 5.5 through Anthropic's API. Each email goes into one shared prompt (also to be used for Qwen) that marks the email as untrusted data; the model returns a 0–100 risk score in structured output, divided by 100. Thinking is off where the model allows it (Opus cannot turn it off), at effort `low`. Few-shot runs add 8 fixed examples from the training split (4 phishing, 4 legitimate). Inputs are capped at 32,000 characters (~8k tokens); 0.28% of emails hit the cap. Every API response is cached, so all results reproduce with $0 and no API key.

| Config | Recall @ 1% FPR (95% CI) | PR-AUC | Precision* | Errors | p50 / p95 latency | $ / 1M emails | Run cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sonnet 5.5 zero-shot | **0.979** (0.968–0.986) | 0.998 | 0.988 | 0 | 1.7 / 3.6 s | $1,638 | $8.31 |
| Sonnet 5.5 few-shot | 0.978 (0.967–0.990) | 0.998 | 0.988 | 0 | 1.6 / 2.1 s | $1,988 | $10.19 |
| Opus 5.5 zero-shot | 0.976 (0.967–0.986) | 0.995 | 0.991 | 7 refusals | 1.9 / 2.7 s | $3,277 | $16.57 |
| Haiku 5.5 few-shot | 0.898 (0.869–0.917) | 0.965 | 0.989 | 49 refusals | 1.0 / 1.7 s | $120 | $0.61 |
| Haiku 5.5 zero-shot | 0.807 (0.738–0.864) | 0.943 | 0.989 | 89 refusals | 1.2 / 1.9 s | $85 | $0.41 |

\*Precision at the threshold chosen on validation. Total Phase 5 API spend: about $36.

**Findings**

- **Sonnet 5.5 is the strongest model so far** and the current top-tier candidate for the cascade. It caught all 57 "true phishing" emails (Nazario + Nigerian Fraud).
- **Opus 5.5 adds nothing over Sonnet** (overlapping confidence intervals) at twice the cost. **Few-shot examples do not help Sonnet.**
- **Safety refusals are a measurable failure mode.** Claude's safety classifiers sometimes decline to answer, and a refusal counts as a miss. Haiku refused 88 phishing emails zero-shot (48 few-shot), mostly flagged as `bio`, likely pharmacy spam; Opus refused 7 of the 20 Nazario credential-phishing emails (`cyber`), so its "true phishing" recall falls to 0.88. Sonnet refused none. Haiku's lower recall therefore partly measures its safety filter, not its judgement. Few-shot halves Haiku's refusals and raises its recall by 9 points.
- **Base rate matters.** Precision is about 0.99 on the test sample, where 45% of emails are phishing. Re-weighted to a realistic 1% phishing rate, it drops to about 0.51–0.58, so roughly every other flag would be a false alarm. This is why the 1%-FPR operating point, not accuracy, is the headline metric.
- **Scores are coarse.** Claude's 0–100 scores cluster on a few values (e.g. 5 and 95), so almost every score is tied with another and recall at a fixed FPR moves in steps. Calibration error is low (ECE 0.03–0.04).

## 6. Next steps, risks and backup plans

**Next (Phase 6):** choose each tier's model and the hand-off thresholds on validation, build the cascade as one more `Detector` so it runs through the same harness, and plot accuracy against latency and cost.

- A refused email could be escalated to the next tier instead of counted as a miss; this rule must be fixed on validation before it is applied to test.
- **TODO (Vishal / Kyle):** further Phase 6 details.

| Risk | Effect | Mitigation / backup |
| --- | --- | --- |
| Public corpora are old and likely in LLM training data | LLM scores look too good, results may not transfer to modern phishing | Label results preliminary; build a newer or synthetic test set if results look suspicious |
| Dataset labels spam as phishing | Partly measures spam detection | Report "true phishing" recall (Nazario, Nigerian Fraud) separately |
| Models learn the source corpus, not phishing | Inflated scores | Headers stripped, dedup; cross-source test (train without one corpus) |
| Safety refusals | Missed phishing in the top tier | Track refusals; escalate them in the cascade |
| Small validation LLM subsample (~14 false positives at 1%) | Noisy LLM thresholds | Larger validation sample for Qwen, which runs locally for free |
| **TODO (Vishal / Ronit):** risks from Phases 2–4 (e.g. fine-tuning memory on a 16 GB M4) | | |

## 7. References

> **TODO (all):** full references, including the dataset (Champa et al., Zenodo 8339691) and the papers cited in the intro slides.

## Contributions (so far)

- **Kyle:** Phases 0–1 (harness, dataset, splits), project plan.
- **Vishal:** **TODO**
- **Ronit:** **TODO**
- **Gavin:** Phase 5 implementation and runs (Claude detectors, prompt, response cache, results).
- **Aneesh:** Phase 5 review and verification, midterm slides and report draft.
