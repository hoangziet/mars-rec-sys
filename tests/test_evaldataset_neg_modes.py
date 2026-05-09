import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataloader import EvalDataset, get_eval_loader


class EvalDatasetNegativeModesTest(unittest.TestCase):
    def _write_eval_csv(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write("user_idx,train_seq,target\n")
        for user_idx, train_seq, target in rows:
            tmp.write(f'{user_idx},"{train_seq}",{target}\n')
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_target_is_first_candidate(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        dataset = EvalDataset(csv_path, n_items=8, num_neg=3)

        sample = dataset[0]

        self.assertEqual(sample["candidates"][0].item(), sample["target"].item())

    def test_candidates_are_unique(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        dataset = EvalDataset(csv_path, n_items=8, num_neg=3)

        candidates = dataset[0]["candidates"].tolist()

        self.assertEqual(len(candidates), len(set(candidates)))

    def test_unknown_neg_mode_raises_value_error(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])

        with self.assertRaisesRegex(ValueError, "Unknown neg_mode"):
            EvalDataset(csv_path, n_items=8, num_neg=3, neg_mode="bad")

    def test_negatives_are_unseen_and_candidate_length_is_exact(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        dataset = EvalDataset(csv_path, n_items=8, num_neg=3)

        candidates = dataset[0]["candidates"].tolist()
        negatives = candidates[1:]

        self.assertEqual(len(candidates), 4)
        self.assertTrue(set(negatives).isdisjoint({1, 2, 3}))

    def test_popularity_mode_uses_item_popularity_weights(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        item_popularity = np.zeros(9)
        item_popularity[8] = 1.0
        dataset = EvalDataset(
            csv_path,
            n_items=8,
            num_neg=1,
            neg_mode="popularity",
            item_popularity=item_popularity,
        )

        candidates = dataset[0]["candidates"].tolist()

        self.assertEqual(candidates, [3, 8])

    def test_popularity_mode_requires_item_popularity(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])

        with self.assertRaisesRegex(ValueError, "item_popularity"):
            EvalDataset(csv_path, n_items=8, num_neg=1, neg_mode="popularity")

    def test_mixed_mode_uses_half_popularity_and_extra_random(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        item_popularity = np.zeros(9)
        item_popularity[8] = 1.0
        dataset = EvalDataset(
            csv_path,
            n_items=8,
            num_neg=3,
            neg_mode="mixed",
            item_popularity=item_popularity,
        )

        candidates = dataset[0]["candidates"].tolist()
        negatives = candidates[1:]

        self.assertEqual(len(negatives), 3)
        self.assertIn(8, negatives)
        self.assertTrue(set(negatives).isdisjoint({1, 2, 3}))
        self.assertEqual(len(negatives), len(set(negatives)))

    def test_fails_fast_when_not_enough_unseen_items(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        dataset = EvalDataset(csv_path, n_items=4, num_neg=2)

        with self.assertRaisesRegex(ValueError, "Cannot sample"):
            dataset[0]
    def test_get_eval_loader_threads_non_default_neg_mode(self):
        csv_path = self._write_eval_csv([(1, "[1, 2]", 3)])
        item_popularity = np.zeros(9)
        item_popularity[8] = 1.0

        loader = get_eval_loader(
            csv_path,
            {"n_items": 8},
            batch_size=1,
            num_neg=1,
            neg_mode="popularity",
            item_popularity=item_popularity,
        )

        self.assertEqual(loader.dataset.neg_mode, "popularity")
        self.assertEqual(loader.dataset[0]["candidates"].tolist(), [3, 8])


if __name__ == "__main__":
    unittest.main()
