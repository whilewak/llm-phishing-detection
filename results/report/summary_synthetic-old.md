# Results on `synthetic-old/synthetic-old` (300 emails, 150 phishing)

| Model | Recall at 1% FPR (95% CI) | Recall / FPR at validation threshold | PR-AUC | Errors | p50 / mean latency | $ per 1M emails |
| --- | --- | --- | --- | ---: | --- | ---: |
| Sonnet 5.5 few-shot | 0.953 (0.903–0.986) | 0.987 / 10.67% | 0.994 | 0 | 1,926.6 / 2,098 ms | $1,802 |
| Sonnet 5.5 zero-shot | 0.953 (0.899–0.987) | 0.987 / 11.33% | 0.995 | 0 | 1,670.6 / 2,026 ms | $999 |
| Haiku 5.5 zero-shot | 0.940 (0.759–0.974) | 0.933 / 0.67% | 0.992 | 0 | 1,049.6 / 1,247 ms | $52 |
| Haiku 5.5 few-shot | 0.913 (0.864–0.986) | 0.980 / 5.33% | 0.993 | 0 | 1,509.8 / 1,661 ms | $100 |
| TF-IDF + LR | 0.240 (0.041–0.331) | 0.367 / 4.00% | 0.807 | 0 | 0.1 / 0 ms | $0 |
| Cascade | 0.240 (0.041–0.331) | 0.667 / 8.00% | 0.822 | 0 | 27.5 / 624 ms | $310 |
| Rules | 0.113 (0.022–0.262) | 0.333 / 8.67% | 0.696 | 0 | 0.2 / 0 ms | $0 |
| ModernBERT | 0.073 (0.000–0.213) | 0.227 / 4.00% | 0.741 | 0 | 19.9 / 18 ms | $0 |
| DeBERTa-v3 | 0.033 (0.007–0.106) | 0.073 / 2.67% | 0.661 | 0 | 62.3 / 53 ms | $0 |

Cascade: share of emails answered by each tier: tier 1 41.3%, tier 2 26.0%, tier 3 32.7%.

"Recall at 1% FPR" picks the best threshold on these emails (how every model is compared); "at validation threshold" applies the threshold fixed on the validation split, which is what a deployment would get. API and cascade latencies are the recorded per-email times.
