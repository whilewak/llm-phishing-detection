# Results on `synthetic-v1/synthetic` (300 emails, 150 phishing)

| Model | Recall at 1% FPR (95% CI) | Recall / FPR at validation threshold | PR-AUC | Errors | p50 / mean latency | $ per 1M emails |
| --- | --- | --- | --- | ---: | --- | ---: |
| Sonnet 5.5 few-shot | 0.980 (0.926–1.000) | 1.000 / 10.67% | 0.999 | 0 | 1,917.6 / 2,069 ms | $1,804 |
| Sonnet 5.5 zero-shot | 0.980 (0.934–1.000) | 1.000 / 11.33% | 0.999 | 0 | 1,650.9 / 2,197 ms | $1,001 |
| Haiku 5.5 zero-shot | 0.960 (0.779–0.993) | 0.960 / 0.67% | 0.998 | 0 | 1,046.6 / 1,251 ms | $52 |
| Haiku 5.5 few-shot | 0.940 (0.897–1.000) | 0.993 / 5.33% | 0.997 | 0 | 1,492.3 / 1,637 ms | $102 |
| Qwen3.5-4B 8-shot | 0.680 (0.604–0.768) | 0.667 / 0.67% | 0.951 | 0 | 378.3 / 422 ms | $0 |
| Qwen3.5-4B zero-shot | 0.600 (0.511–0.707) | 0.620 / 2.00% | 0.955 | 0 | 367.8 / 406 ms | $0 |
| Cascade | 0.427 (0.190–0.537) | 0.660 / 8.00% | 0.853 | 0 | 25.7 / 496 ms | $185 |
| TF-IDF + LR | 0.247 (0.041–0.345) | 0.373 / 4.00% | 0.806 | 0 | 0.1 / 0 ms | $0 |
| Laya | 0.153 (0.083–0.335) | 0.120 / 0.67% | 0.856 | 0 | 210.7 / 235 ms | $0 |
| Rules | 0.113 (0.022–0.269) | 0.340 / 8.67% | 0.703 | 0 | 0.2 / 0 ms | $0 |
| ModernBERT | 0.073 (0.000–0.221) | 0.233 / 4.00% | 0.745 | 0 | 19.6 / 18 ms | $0 |
| DeBERTa-v3 | 0.033 (0.007–0.106) | 0.073 / 2.67% | 0.661 | 0 | 55.7 / 47 ms | $0 |

Cascade: share of emails answered by each tier: tier 1 42.0%, tier 2 26.0%, tier 3 12.7%, tier 4 19.3%.

"Recall at 1% FPR" picks the best threshold on these emails (how every model is compared); "at validation threshold" applies the threshold fixed on the validation split, which is what a deployment would get. API and cascade latencies are the recorded per-email times.
