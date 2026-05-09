import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import evaluate_bprmf


class SingleEvalDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "user": torch.tensor(1, dtype=torch.long),
            "candidates": torch.tensor([2, 3], dtype=torch.long),
        }


class DummyBPRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(2, 1)
        self.item_embedding = torch.nn.Embedding(4, 1)
        with torch.no_grad():
            self.user_embedding.weight.zero_()
            self.item_embedding.weight.zero_()
            self.user_embedding.weight[1, 0] = 1.0
            self.item_embedding.weight[2, 0] = 2.0  # positive item, should rank first
            self.item_embedding.weight[3, 0] = 1.0  # negative item


class EvaluateBPRMFTest(unittest.TestCase):
    def test_positive_candidate_ranks_above_negative(self):
        model = DummyBPRModel()
        loader = DataLoader(SingleEvalDataset(), batch_size=1, shuffle=False)

        results = evaluate_bprmf(model, loader, device="cpu", k_list=(1, 2))

        self.assertEqual(results["HR@1"], 1.0)
        self.assertEqual(results["HR@2"], 1.0)


if __name__ == "__main__":
    unittest.main()
