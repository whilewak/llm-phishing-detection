"""
Core System 1 Decision Model Detector for Phishing & Social Engineering.
Utilizes Laya (ModernBERT-large decision head) for fast on-device inference.
"""

import time
from typing import Dict, Any, List, Optional
import torch
from laya import Router


class LayaPhishingDetector:
    """
    Discriminative System 1 Phishing Detector.
    Extracts calibrated probabilities across orthogonal social engineering signals
    in a single parallel forward pass.
    """

    DEFAULT_QUESTIONS = {
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing, scam, or social engineering attempt to steal money, passwords, or data?",
            "criteria": {
                "true": "phishing, scam, or social engineering fraud",
                "false": "a legitimate business or personal email"
            },
        },
        "urgency_pressure": {
            "type": "noul",
            "instructions": "Does this email create artificial time pressure, panic, or threats of account suspension?",
            "criteria": {
                "true": "severe urgency or account termination threat",
                "false": "routine timing or no deadline pressure"
            },
        },
        "credential_harvest": {
            "type": "noul",
            "instructions": "Does this email ask the recipient to verify login credentials, enter passwords, or access an auth portal?",
            "criteria": {
                "true": "requests login credentials, passwords, or MFA tokens",
                "false": "no login or credential request"
            },
        },
        "financial_pressure": {
            "type": "noul",
            "instructions": "Does this email ask to buy gift cards, execute wire transfers, or divert invoice payments?",
            "criteria": {
                "true": "requests direct money transfer or gift cards",
                "false": "no financial payment diversion or gift card request"
            },
        },
        "suspicious_attachment": {
            "type": "noul",
            "instructions": "Does this email urge opening an attachment, enabling macros, or reviewing an attached invoice?",
            "criteria": {
                "true": "prompts to open attachment or enable content",
                "false": "no attachment or safe standard file"
            },
        },
        "classification": {
            "type": "choice",
            "instructions": "What is the security classification of the email?",
            "criteria": {
                "benign": "legitimate, safe business or personal email",
                "promotional": "unsolicited marketing, newsletter, or sales offer",
                "phishing": "fraudulent lure, credential theft, impersonation, or scam"
            }
        }
    }

    def __init__(
        self,
        default_model: str = "english",
        device: Optional[str] = None,
        risk_threshold: float = 1.05
    ):
        """
        Initialize the detector, select compute device (MPS/CUDA/CPU),
        and preload the specified model checkpoint.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.default_model = default_model
        self.risk_threshold = risk_threshold
        
        # Initialize Router
        self.router = Router(default=default_model, device=self.device)
        self.router.preload([default_model])

    def predict(
        self,
        email: Dict[str, str],
        custom_questions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an email payload against the System 1 questions.
        
        Args:
            email: Dict containing 'from', 'subject', and 'body'
            custom_questions: Optional override for the question suite
            
        Returns:
            Dict containing predicted verdict, composite risk score,
            individual signal probabilities, and inference latency.
        """
        questions = custom_questions or self.DEFAULT_QUESTIONS
        state = {
            "from": email.get("from", ""),
            "subject": email.get("subject", ""),
            "body": email.get("body", "")
        }

        t_start = time.perf_counter()
        raw_pred = self.router.predict(state, questions)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        answers = raw_pred.get("answers", {})

        p_phish = answers.get("is_phishing", {}).get("noul", 0.0)
        p_urg = answers.get("urgency_pressure", {}).get("noul", 0.0)
        p_cred = answers.get("credential_harvest", {}).get("noul", 0.0)
        p_fin = answers.get("financial_pressure", {}).get("noul", 0.0)
        p_att = answers.get("suspicious_attachment", {}).get("noul", 0.0)
        
        class_ans = answers.get("classification", {})
        chosen_class = class_ans.get("choice", "unknown")
        class_probs = class_ans.get("probabilities", {})

        # Multi-signal composite threat scoring formula
        composite_score = (
            p_phish
            + 0.6 * p_urg
            + 0.8 * p_cred
            + 0.8 * p_fin
            + 0.7 * p_att
        )

        is_phishing = (composite_score >= self.risk_threshold) or (chosen_class == "phishing")
        verdict = "PHISHING" if is_phishing else "BENIGN"

        return {
            "verdict": verdict,
            "composite_risk_score": round(composite_score, 4),
            "risk_threshold": self.risk_threshold,
            "latency_ms": round(elapsed_ms, 2),
            "signals": {
                "p_is_phishing": round(p_phish, 4),
                "p_urgency_pressure": round(p_urg, 4),
                "p_credential_harvest": round(p_cred, 4),
                "p_financial_pressure": round(p_fin, 4),
                "p_suspicious_attachment": round(p_att, 4),
                "classification_head": chosen_class,
                "classification_probabilities": class_probs
            },
            "raw_answers": answers
        }

    def evaluate_batch(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a dataset of emails and compute summary metrics.
        """
        tp = fp = tn = fn = 0
        latencies = []
        detailed_results = []

        for email in emails:
            res = self.predict(email)
            ground_truth = email.get("ground_truth", "UNKNOWN").upper()
            pred = res["verdict"].upper()
            latencies.append(res["latency_ms"])

            if ground_truth != "UNKNOWN":
                if ground_truth == "PHISHING" and pred == "PHISHING":
                    tp += 1
                elif ground_truth == "BENIGN" and pred == "PHISHING":
                    fp += 1
                elif ground_truth == "BENIGN" and pred == "BENIGN":
                    tn += 1
                elif ground_truth == "PHISHING" and pred == "BENIGN":
                    fn += 1

            detailed_results.append({
                "id": email.get("id"),
                "ground_truth": ground_truth,
                "predicted": pred,
                "correct": (ground_truth == pred) if ground_truth != "UNKNOWN" else None,
                "score": res["composite_risk_score"],
                "latency_ms": res["latency_ms"]
            })

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        latencies.sort()
        return {
            "total_samples": total,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            },
            "latency": {
                "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p50_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
                "p90_ms": round(latencies[int(len(latencies) * 0.9)], 2) if latencies else 0.0
            },
            "results": detailed_results
        }
