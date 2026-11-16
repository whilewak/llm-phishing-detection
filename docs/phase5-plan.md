# Phase 5 · Frontier LLMs: plan

Final version, 2026-11-15. Claude only, through Anthropic's API.

## Context

Phase 5 of [PLAN.md](PLAN.md) adds frontier LLMs to the benchmark as its expensive, strongest point of comparison. They are used three ways: the comparison point for goal 3 (System 1 / encoders vs LLM), the Tier 4 candidates for the Phase 6 cascade, and the reference for the cascade target (≥95% of the best single model's recall). Phases 0–1 are done (harness, metrics, frozen `champa2023-v1` splits with 2,500-email `val_llm` / `test_llm`). Phase 5 doesn't depend on Phases 2–4.

## Decisions

| Decision | Why |
| --- | --- |
| **Claude only**: Sonnet 5.5 and Haiku 5.5 | No OpenAI API credit. Gemini isn't on Amazon Bedrock, which was the first plan for one bill. |
| **Anthropic's API** (`ANTHROPIC_API_KEY`) is the default backend; Amazon Bedrock is a config option (`backend = "bedrock"`) | Bedrock refused Claude for the AWS account ("not available for this account"), even in the console. |
| **Both models zero- and few-shot** (4 configs) | With one provider, the few-shot runs are cheap enough to keep for both. |
| **Opus 5.5 deferred** | ~$31 more for zero-shot; to be decided after these results. |
| **No temperature**; thinking off, effort `low` | Current Claude models reject `temperature`. Thinking off keeps cost and latency low and fits a cascade tier. Reproducibility comes from the response cache. |
| **0–100 risk score** in structured output, divided by 100 | Claude returns no token probabilities. Tie share is reported. |
| **No refusal fallback** | A fallback would answer with a different model. Refusals count as errors. |
| **Input capped at 32,000 characters** (~8k tokens) per email | Bounds the cost of the longest emails; the truncated share is reported. Same cap for every LLM, Qwen included. |
| **Budget $55** in total, enforced per run | User's limit. Expected spend ~$27. |

## Models

| Model | API id | $/1M input / cache write / cache read / output | Thinking | Effort |
| --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | 2.00 / 2.50 / 0.10 / 10.00 | `{"type": "between_tools"}` (off; `disabled` is a 400 on Sonnet 5.5) | `low` |
| Claude Haiku 5.5 | `claude-haiku-5-5` | 0.10 / 0.125 / 0.01 / 0.50 (prompts ≤100k tokens) | `{"type": "disabled"}` | `low` |

Account rate limits (read from response headers): 13,000 requests/min, 10M input and 2M output tokens/min, on both models.

## Configs (`configs/models/`)

| Config | Prompting | Cap |
| --- | --- | --- |
| `claude-sonnet-5-5-zeroshot` | zero-shot | $20 |
| `claude-sonnet-5-5-fewshot` | 8 examples | $24.90 |
| `claude-haiku-5-5-zeroshot` | zero-shot | $5 |
| `claude-haiku-5-5-fewshot` | 8 examples | $5 |

Caps sum to $54.90, so the four can run in parallel without exceeding $55. Each config pins the model, prices, thinking, effort, `max_output_tokens = 256`, `concurrency = 16`, `max_input_chars = 32000`, `data_allowed`, and `[eval] warmup = 1, batch_size = 16`.

## Design

The harness (`src/phishbench/eval/harness.py`) is unchanged: it chooses the threshold on `val_llm`, counts errors as wrong, and reports tie share, calibration, latency, cost per million and per-source results.

- **Shared prompt** (`src/phishbench/prompts.py`): one set of task instructions for every LLM (Qwen in Phase 4 too), the email inside `<email>` tags marked as untrusted data, answer formats for a 0–100 score or a single answer token, and `PROMPT_VERSION = "v1"`. Wording may change only from `val_llm` outputs.
- **Few-shot pool** (`configs/prompts/fewshot-champa2023-v1.json`, built by `scripts/build_fewshot.py`): 8 train emails, labels alternating. Phishing: Nazario, Nigerian Fraud, TREC-07, TREC-06. Legitimate: Enron, Ling, SpamAssassin, TREC-06. Stored with its content hash and the train split's id hash. Fixed rather than chosen on `val_llm`, which is too small to compare pools.
- **Shared LLM base** (`src/phishbench/detectors/llm_base.py`): parallel requests; a response cache in `data/cache/llm/<backend>/` (git-ignored), so reruns and interrupted runs resume for free; cost from token usage × config prices; a cost cap checked before each request; JSON scoring, plus the logprob scoring Qwen will use. Refusals, `max_tokens` and unparseable answers are cached errors. Throttling, server and network errors left after 5 SDK retries are errors that aren't cached, so a rerun retries them. Bad requests (4xx) stop the run.
- **Claude detector** (`src/phishbench/detectors/claude.py`): `client.messages.parse(...)` with the Pydantic `RiskAssessment` output, the system prompt marked for prompt caching, and thinking and effort from the config.
- **Runners:** `scripts/evaluate.py --max-cost N` lowers a config's cap and records new API spend in the results file (`api_spend`). `scripts/run_suite.py --budget N` runs configs one after another under one total budget.
- **Tests:** `tests/test_prompts.py` and `tests/test_llm.py` use fake clients and need no key. An integration test runs the 7 sample emails on Haiku when `ANTHROPIC_API_KEY` is set.

## Budget

5,000 calls per config (2,500 `val_llm` for the threshold + 2,500 `test_llm`), assuming ~900 email tokens, ~612 instruction tokens, ~6k cached few-shot tokens and ~22 output tokens per call:

| Config | Expected |
| --- | --- |
| Sonnet zero-shot | ~$11 |
| Sonnet few-shot | ~$14 |
| Haiku zero-shot | ~$0.60 |
| Haiku few-shot | ~$0.90 |
| **Total** | **~$27** (hard limit $55) |

Smoke tests on the 7 sample emails cost $0.015.

## Running it

1. `python3 -m venv .venv && .venv/bin/pip install -e '.[data,llm]'`
2. `python scripts/download_data.py && python scripts/build_splits.py --verify && python scripts/build_fewshot.py`
3. `ANTHROPIC_API_KEY` in the environment.
4. All four in parallel, caps summing to $54.90:
   ```bash
   for c in haiku-5-5-zeroshot:5 haiku-5-5-fewshot:5 sonnet-5-5-zeroshot:20 sonnet-5-5-fewshot:24.9; do
     python scripts/evaluate.py --model configs/models/claude-${c%%:*}.toml --splits data/splits/champa2023-v1 \
       --train train --val val_llm --test test_llm --max-cost ${c##*:} &
   done; wait
   ```
   Or one after another under one budget: `python scripts/run_suite.py --budget 55 configs/models/claude-*.toml`.

## Verification

- `python -m unittest discover tests` passes without a key.
- Each results file has errors and refusals reported, `api_spend.new_usd` under its cap, and a tie share.
- Rerunning a finished config makes 0 API calls and gives identical scores.
- Gate (PLAN.md): all 4 configs scored on `test_llm`, thresholds chosen on `val_llm` only.

## Appendix: providers considered (2026-11-15)

- **OpenAI** GPT-6.1 Sol ($2/$10) and GPT-6 Luna ($0.10/$0.50) were price-matched to Sonnet and Haiku, with Luna also scored from logprobs (allowed only at reasoning effort `none`). Dropped: no API credit. GPT-6 Astra ($10/$50) was excluded on price.
- **Google Gemini** 4 Argon and 3.5 Flash-Lite: recent Gemini models ignore `temperature` and logprobs are deprecated on Gemini 3.x. Dropped: not on Amazon Bedrock (only open-weight Gemma is).
- **Amazon Bedrock**: Claude, GPT-6.x and Gemma are available through it, at global-routing prices equal to list prices but with no batch discount. Dropped as the default because the account was refused Claude.
- **Claude Opus 5.5** ($4/$20, thinking always on): ~$31 zero-shot, ~$37 few-shot. Deferred.

Sources: Anthropic model and pricing reference (claude-api skill, 2026-11-03); OpenAI docs (pricing, models, latest-model guide); Google AI Studio docs (models, pricing, thinking, generate-content reference) and Google Cloud's inference reference; AWS Bedrock model cards for Claude Sonnet 5.5, Haiku 5.5, GPT-6.1 Sol, GPT-6 Luna and the Google model list.
