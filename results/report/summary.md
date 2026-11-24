# Results on `champa2023-v1/test_llm` (2500 emails, 1118 phishing)

| Model | Recall at 1% FPR (95% CI) | Recall / FPR at validation threshold | PR-AUC | Errors | p50 / mean latency | $ per 1M emails |
| --- | --- | --- | --- | ---: | --- | ---: |
| Cascade | 0.984 (0.897–0.992) | 0.986 / 1.16% | 0.996 | 0 | 0.1 / 228 ms | $149 |
| Sonnet 5.5 zero-shot | 0.979 (0.968–0.986) | 0.979 / 0.94% | 0.998 | 0 | 1,728.9 / 2,255 ms | $1,638 |
| Sonnet 5.5 few-shot | 0.978 (0.967–0.990) | 0.977 / 0.94% | 0.998 | 0 | 1,639.0 / 1,640 ms | $1,988 |
| Opus 5.5 zero-shot | 0.976 (0.967–0.986) | 0.976 / 0.72% | 0.995 | 7 | 1,864.6 / 1,934 ms | $3,277 |
| TF-IDF + LR | 0.937 (0.883–0.965) | 0.933 / 0.80% | 0.996 | 0 | 0.1 / 0 ms | $0 |
| Haiku 5.5 few-shot | 0.898 (0.869–0.917) | 0.898 / 0.80% | 0.965 | 49 | 1,001.0 / 1,083 ms | $120 |
| DeBERTa-v3 | 0.882 (0.843–0.914) | 0.903 / 1.52% | 0.987 | 0 | 109.7 / 109 ms | $0 |
| ModernBERT | 0.820 (0.745–0.891) | 0.848 / 1.23% | 0.989 | 0 | 27.9 / 28 ms | $0 |
| Haiku 5.5 zero-shot | 0.807 (0.738–0.864) | 0.807 / 0.72% | 0.943 | 89 | 1,180.5 / 1,226 ms | $85 |
| Qwen3.5-4B zero-shot | 0.711 (0.651–0.771) | 0.711 / 0.72% | 0.971 | 0 | 1,119.1 / 2,013 ms | $0 |
| Qwen3.5-4B 8-shot | 0.708 (0.611–0.752) | 0.738 / 1.52% | 0.974 | 0 | 1,096.7 / 1,885 ms | $0 |
| Rules | 0.055 (0.021–0.067) | 0.027 / 0.51% | 0.473 | 0 | 0.2 / 0 ms | $0 |

Cascade: share of emails answered by each tier: tier 1 80.4%, tier 2 11.1%, tier 3 8.5%.

"Recall at 1% FPR" picks the best threshold on these emails (how every model is compared); "at validation threshold" applies the threshold fixed on the validation split, which is what a deployment would get. API and cascade latencies are the recorded per-email times.
