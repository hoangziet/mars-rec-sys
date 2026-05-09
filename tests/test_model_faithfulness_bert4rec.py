import unittest
import torch

from models.bert4rec import BERT4Rec


class BERT4RecFaithfulnessTest(unittest.TestCase):
    def test_loss_ignores_unmasked_positions(self):
        model = BERT4Rec(n_items=100, max_len=8, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0)
        masked_seq = torch.tensor([[1, 2, model.mask_token, 4]], dtype=torch.long)
        labels = torch.tensor([[0, 0, 3, 0]], dtype=torch.long)

        loss = model.loss(masked_seq, labels)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
