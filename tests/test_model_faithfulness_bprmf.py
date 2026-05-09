import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.bprmf import BPRMF


class BPRMFFaithfulnessTest(unittest.TestCase):
    def test_loss_decreases_when_pos_score_exceeds_neg_score(self):
        model = BPRMF(n_users=10, n_items=20, emb_dim=8)
        user = torch.tensor([1], dtype=torch.long)
        pos = torch.tensor([2], dtype=torch.long)
        neg = torch.tensor([3], dtype=torch.long)

        with torch.no_grad():
            model.user_emb.weight[user] = 1.0
            model.item_emb.weight[pos] = 1.0
            model.item_emb.weight[neg] = -1.0

        loss = model.loss(user, pos, neg)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(float(loss.item()), 0.7)


if __name__ == "__main__":
    unittest.main()
