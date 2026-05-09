import unittest

import torch

from models.gru4rec import GRU4Rec, _compact_left_padded_batch


class GRU4RecMaskingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GRU4Rec(n_items=50, emb_dim=8, hidden_dim=8, num_layers=1, dropout=0.0)
        self.model.eval()

    def test_compaction_preserves_valid_tokens_for_multi_row_batch(self):
        padded_seq = torch.tensor([
            [0, 0, 4, 5, 6],
            [0, 7, 8, 9, 10],
        ], dtype=torch.long)
        padded_mask = torch.tensor([
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ], dtype=torch.bool)

        compact_seq, lengths = _compact_left_padded_batch(padded_seq, padded_mask)

        self.assertTrue(torch.equal(lengths, torch.tensor([3, 4], dtype=torch.long)))
        self.assertTrue(torch.equal(
            compact_seq,
            torch.tensor([
                [4, 5, 6, 0, 0],
                [7, 8, 9, 10, 0],
            ], dtype=torch.long),
        ))

    def test_forward_ignores_left_padding_when_mask_provided(self):
        padded_seq = torch.tensor([[0, 0, 4, 5, 6]], dtype=torch.long)
        padded_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)

        compact_seq = torch.tensor([[4, 5, 6]], dtype=torch.long)
        compact_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

        with torch.no_grad():
            padded_logits = self.model(padded_seq, mask=padded_mask)
            compact_logits = self.model(compact_seq, mask=compact_mask)

        self.assertTrue(torch.allclose(padded_logits, compact_logits, atol=1e-6, rtol=1e-6))

    def test_forward_matches_unmasked_path_when_no_padding_present(self):
        input_seq = torch.tensor([[4, 5, 6]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

        with torch.no_grad():
            masked_logits = self.model(input_seq, mask=mask)
            unmasked_logits = self.model(input_seq)

        self.assertTrue(torch.allclose(masked_logits, unmasked_logits, atol=1e-6, rtol=1e-6))

    def test_forward_handles_multi_row_left_padded_batch(self):
        padded_seq = torch.tensor([
            [0, 0, 4, 5, 6],
            [0, 7, 8, 9, 10],
        ], dtype=torch.long)
        padded_mask = torch.tensor([
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ], dtype=torch.bool)

        with torch.no_grad():
            padded_logits = self.model(padded_seq, mask=padded_mask)
            row0_logits = self.model(torch.tensor([[4, 5, 6]], dtype=torch.long))
            row1_logits = self.model(torch.tensor([[7, 8, 9, 10]], dtype=torch.long))
            expected_logits = torch.cat([row0_logits, row1_logits], dim=0)

        self.assertTrue(torch.allclose(padded_logits, expected_logits, atol=1e-6, rtol=1e-6))

    def test_forward_rejects_non_contiguous_mask(self):
        input_seq = torch.tensor([[0, 4, 0, 5, 6]], dtype=torch.long)
        bad_mask = torch.tensor([[0, 1, 0, 1, 1]], dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, r"0\*1\*"):
            self.model(input_seq, mask=bad_mask)

    def test_forward_rejects_all_padding_rows(self):
        input_seq = torch.tensor([[0, 0, 0]], dtype=torch.long)
        empty_mask = torch.tensor([[0, 0, 0]], dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "all-padding row"):
            self.model(input_seq, mask=empty_mask)


if __name__ == "__main__":
    unittest.main()
