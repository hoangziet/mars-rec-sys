import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataloader import MaskedSequenceDataset


class BERT4RecMaskingPolicyTest(unittest.TestCase):
    def test_train_sample_guarantees_at_least_one_supervised_target(self):
        df = pd.DataFrame([
            {"item_sequence": "[11, 22, 33]"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            df.to_csv(csv_path, index=False)

            ds = MaskedSequenceDataset(
                csv_path=csv_path,
                n_items=100,
                max_len=5,
                mask_prob=0.0,
                is_train=True,
            )

            sample = ds[0]
            self.assertGreater(sample["labels"].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
