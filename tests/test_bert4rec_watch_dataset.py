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


def test_watch_input_ids_reflect_engagement_bins_for_unmasked_positions(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["11 12 13 14"],
            "engagement_sequence": ["0.0 0.3 0.6 1.0"],
        }
    ).to_csv(csv_path, index=False)

    dataset = MaskedSequenceDataset(
        str(csv_path),
        n_items=20,
        max_len=4,
        mask_prob=0.0,
        is_train=True,
        force_last_item_mask=True,
        watch_num_bins=5,
    )

    sample = dataset[0]
    watch_ids = sample["watch_input_ids"].tolist()
    from pipeline.watch_features import WATCH_MASK_ID
    assert watch_ids[-1] == WATCH_MASK_ID  # last is masked
    assert watch_ids[0] == 2   # 0.0 -> bin 0 -> ID 2+0=2
    assert watch_ids[1] == 3   # 0.3 -> bin 1 -> ID 2+1=3
    assert watch_ids[2] == 5   # 0.6 -> bin 3 -> ID 2+3=5
