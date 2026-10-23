# LLM & Multi-Tier Phishing Detection Benchmark

A research and benchmarking project comparing different paradigms of AI models for real-time phishing and social engineering detection across **Accuracy**, **Latency**, and **Cost**.

---

## 🏗️ 4-Tier Model Taxonomy

```
+---------------------------------------------------------------------------------------------------------+
|                                    Inbound Email / URL Telemetry Stream                                  |
+---------------------------------------------------------------------------------------------------------+
                                                     │
                                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────┐
     │  Tier 1: Traditional ML (Random Forest / XGBoost / SVM)                       │
     │  • Domain age, URL entropy, lexical counts, SPF/DKIM flags                    │
     │  • Latency: < 5 ms | Cost: $0.00                                              │
     └───────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────┐
     │  Tier 2: System 1 Decision Model (Laya / Jev Architecture)                    │
     │  • Open-Source ModernBERT-large (421M params), Non-autoregressive             │
     │  • Calibrated multi-signal probabilities in a single forward pass             │
     │  • Latency: ~50 - 160 ms (on Apple Silicon M4 GPU) | Cost: $0.00 (Local)       │
     └───────────────────────────────────────────────────────────────────────────────┘
                                                     │ (Ambiguous / High Risk)
                                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────┐
     │  Tier 3: Small Language Models (SLMs - Llama-3.2 3B, Phi-3.5 Mini, Gemma 2)   │
     │  • Semantic parsing, token classification, local reasoning                    │
     │  • Latency: ~300 - 800 ms | Cost: $0.00 (Local MLX / Ollama)                  │
     └───────────────────────────────────────────────────────────────────────────────┘
                                                     │ (Novel / Zero-Day Threats)
                                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────┐
     │  Tier 4: Frontier LLMs & Multimodal Reasoners (GPT-4o, Claude 3.5, Gemini 1.5)│
     │  • Deep brand intent analysis, DOM reconstruction, visual logo inspection     │
     │  • Latency: ~1,500 - 3,500 ms | Cost: ~$0.005 - $0.02 / query                │
     └───────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Proof-of-Concept: System 1 (Laya) on Apple M4

### What is Laya?
**Laya** is the open-source, open-weights (Apache 2.0) alternative to closed System 1 decision models like TypeSafe AI's **Jev**. 
- Built on a **ModernBERT-large** bidirectional encoder backbone (~421M parameters).
- **Non-autoregressive:** Does *not* generate token-by-token text, avoiding parsing errors and hallucinations.
- Returns calibrated probabilities and choices in parallel in a single forward pass.

### Hardware Profile Tested
- **Hardware:** MacBook Air (Apple M4, 10 CPU cores, 10 GPU cores, 16 GB Unified RAM)
- **PyTorch Acceleration:** Apple Metal Performance Shaders (`MPS`)
- **Memory Footprint:** ~180 MB RSS (< 2% of total RAM)
- **Latency (Single-Signal):** ~55 - 160 ms
- **Throughput:** ~6 - 18 inferences / sec

---

## 🚀 Running the POC

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the System 1 Phishing Proof-of-Concept
python poc_system1_test.py
```

### Verified Multi-Signal Defense Results
| Test Case | Type | Detected Signals | Ground Truth | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Microsoft 365 Expiry** | Credential Harvesting | P(urgency)=0.72, P(cred)=0.70 | PHISHING | **PASS** |
| **GitHub Monthly Invoice** | Legitimate Transaction | P(urgency)=0.05, P(cred)=0.10 | BENIGN | **PASS** |
| **CEO Gift Card Favor** | Business Email Compromise (BEC) | P(urgency)=0.65, P(fin)=0.60 | PHISHING | **PASS** |
| **Internal IT Maintenance** | Admin Announcement | P(phish)=0.00, P(cred)=0.04 | BENIGN | **PASS** |
| **Overdue Invoice Document** | Malicious Macro / Attachment | P(attachment)=0.79, P(urgency)=0.24 | PHISHING | **PASS** |

---

## 📋 Next Phases in the Comparison Study
1. **Traditional ML:** Train and benchmark Random Forest / XGBoost on PhishTank and Enron email feature vectors.
2. **SLMs:** Benchmark quantized Llama-3.2-3B or Phi-3.5 on Apple Silicon using MLX / Ollama.
3. **Cloud LLMs:** Measure prompt latency, token cost, and F1 score with frontier APIs (Gemini / Claude / OpenAI).
4. **Comparative Analysis:** Generate unified Pareto curves comparing Accuracy vs. Latency vs. Cost.
