import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from pipeline.loaders import MaskedSequenceDataset


def test_masked_sequence_dataset_emits_engagement_and_watch_inputs(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["11 12 13 14"],
            "engagement_sequence": ["0.9 0.2 0.7 1.0"],
        }
    ).to_csv(csv_path, index=False)

    dataset = MaskedSequenceDataset(
        str(csv_path),
        n_items=20,
        max_len=4,
        mask_prob=1.0,
        is_train=True,
        watch_num_bins=5,
    )

    sample = dataset[0]
    assert set(sample) >= {"input_seq", "labels", "engagement", "watch_input_ids"}
    assert sample["engagement"].tolist() == pytest.approx([0.9, 0.2, 0.7, 1.0])
    assert sample["watch_input_ids"].shape[0] == 4
    masked_positions = sample["labels"] != 0
    assert masked_positions.any()
    # masked positions should all have the same special watch_id (WATCH_MASK_ID = 1)
    assert sample["watch_input_ids"][masked_positions].unique().numel() == 1
