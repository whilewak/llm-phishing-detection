import unittest
from unittest.mock import MagicMock, patch
import torch

from phishbench.detectors.encoder import TransformerEncoderDetector
from phishbench.types import Email


class TransformerEncoderTest(unittest.TestCase):
    def test_mock_predict_flow(self):
        det = TransformerEncoderDetector(model_name_or_path="fake/model", name="test-encoder")
        
        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102], [101, 103]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
        mock_model.return_value = mock_output
        
        det.tokenizer = mock_tokenizer
        det.model = mock_model
        det._is_loaded = True

        emails = [
            Email(id="1", label="legitimate", sender="", subject="Hello", body="Test message", urls=()),
            Email(id="2", label="phishing", sender="", subject="Alert", body="Urgent login", urls=()),
        ]
        preds = det.predict(emails)
        self.assertEqual(len(preds), 2)
        self.assertFalse(preds[0].failed)
        self.assertFalse(preds[1].failed)
        self.assertLess(preds[0].score, 0.5)
        self.assertGreater(preds[1].score, 0.5)

    def test_error_handling_in_predict(self):
        det = TransformerEncoderDetector(model_name_or_path="fake/model", name="test-encoder")
        mock_tokenizer = MagicMock(side_effect=RuntimeError("Tokenization failed"))
        det.tokenizer = mock_tokenizer
        det.model = MagicMock()
        det._is_loaded = True

        emails = [Email(id="1", label=None, sender="", subject="Hi", body="", urls=())]
        preds = det.predict(emails)
        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0].failed)
        self.assertIn("Tokenization failed", preds[0].error)


if __name__ == "__main__":
    unittest.main()

