import unittest

import torch

from models.gsasrec import GSASRec
from models.sasrec import SASRec


class SASFamilyFaithfulnessTest(unittest.TestCase):
    def test_sasrec_forward_shape_matches_full_item_vocab(self):
        model = SASRec(n_items=50, emb_dim=16, num_layers=1, num_heads=2, dropout=0.0, max_len=10)
        seq = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            logits = model(seq)

        self.assertEqual(tuple(logits.shape), (1, 51))

    def test_gsasrec_confidence_weight_changes_loss(self):
        model = GSASRec(n_items=50, emb_dim=16, num_layers=1, num_heads=2, dropout=0.0, max_len=10)
        seq = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        target = torch.tensor([5], dtype=torch.long)
        low_confidence = torch.tensor([0.2], dtype=torch.float32)
        high_confidence = torch.tensor([1.0], dtype=torch.float32)

        loss_low = model.loss(seq, target, low_confidence)
        loss_high = model.loss(seq, target, high_confidence)

        self.assertNotEqual(float(loss_low.item()), float(loss_high.item()))

    def test_sas_family_uses_last_valid_absolute_timestep_for_left_padded_mask(self):
        for model_cls in (SASRec, GSASRec):
            with self.subTest(model=model_cls.__name__):
                model = model_cls(
                    n_items=5,
                    emb_dim=4,
                    num_layers=0,
                    num_heads=2,
                    dropout=0.0,
                    max_len=4,
                )
                model.final_ln = torch.nn.Identity()
                model.output = torch.nn.Linear(4, 6, bias=False)

                with torch.no_grad():
                    model.item_embedding.weight.zero_()
                    model.pos_embedding.weight.zero_()
                    model.pos_embedding.weight[1, 0] = 1.0
                    model.pos_embedding.weight[3, 0] = 3.0
                    model.output.weight.zero_()
                    model.output.weight[1, 0] = 1.0

                seq = torch.tensor([[0, 0, 4, 5]], dtype=torch.long)
                mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
                logits = model(seq, mask=mask)

                self.assertEqual(float(logits[0, 1].item()), 3.0)


if __name__ == "__main__":
    unittest.main()
