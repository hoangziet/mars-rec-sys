import ast
import unittest
import pandas as pd

from dataloader import TrainSequenceDataset

class ProcessedConfidenceContractTest(unittest.TestCase):
    def test_train_export_has_confidence_and_not_all_ones(self):
        train_df = pd.read_csv("data/processed/train.csv")
        self.assertIn("confidence", train_df.columns)
        self.assertGreater(train_df["confidence"].nunique(), 1)

    def test_train_confidence_matches_interactions_for_target_item(self):
        train_df = pd.read_csv("data/processed/train.csv")
        interactions_df = pd.read_csv("data/processed/interactions.csv")
        confidence_lookup = interactions_df.set_index(["user_idx", "item_idx"])["confidence"]

        sample_size = min(25, len(train_df))
        self.assertGreater(sample_size, 0)
        for row in train_df.head(sample_size).itertuples(index=False):
            history = ast.literal_eval(row.item_sequence)
            self.assertEqual(row.seq_len, len(history))
            self.assertIn((row.user_idx, row.target), confidence_lookup.index)
            self.assertEqual(row.confidence, confidence_lookup.loc[(row.user_idx, row.target)])

    def test_train_sequence_dataset_raises_when_confidence_missing_for_gsasrec(self):
        with self.assertRaisesRegex(
            ValueError,
            "TrainSequenceDataset requires 'confidence' column when use_confidence=True",
        ):
            TrainSequenceDataset(
                "tests/fixtures/train_without_confidence.csv",
                use_confidence=True,
            )

if __name__ == "__main__":
    unittest.main()
