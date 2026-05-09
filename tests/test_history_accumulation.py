import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataloader import BPRDataset, EvalDataset


class HistoryAccumulationTest(unittest.TestCase):
    def test_bpr_dataset_accumulates_user_items_across_rows(self):
        df = pd.DataFrame([
            {"user_idx": 1, "item_sequence": "[1, 2, 3]"},
            {"user_idx": 1, "item_sequence": "[4, 5]"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            df.to_csv(csv_path, index=False)

            ds = BPRDataset(csv_path, n_items=20)

        self.assertEqual(ds.user_items[1], {1, 2, 3, 4, 5})

    def test_eval_dataset_accumulates_seen_items_across_rows(self):
        df = pd.DataFrame([
            {"user_idx": 1, "train_seq": "[1, 2]", "target": 3},
            {"user_idx": 1, "train_seq": "[4]", "target": 5},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "val.csv"
            df.to_csv(csv_path, index=False)

            ds = EvalDataset(csv_path, n_items=50, max_len=5, num_neg=5, neg_mode="random")

        self.assertEqual(ds.user_history[1], {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
