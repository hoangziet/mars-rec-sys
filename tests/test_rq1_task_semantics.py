import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from data.preprocess import split_leave_one_out
from pipeline.loaders import FullSortEvalDataset


def test_split_rejects_duplicate_courses():
    user_sequences = pd.DataFrame(
        {
            "user_idx": [1],
            "user_id": ["u1"],
            "item_seq_idx": [[10, 20, 30, 10, 40]],
            "engagement_seq": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "watch_signal_seq": [[True, True, True, True, True]],
        }
    )

    with pytest.raises(ValueError, match="next-distinct-course semantics"):
        split_leave_one_out(user_sequences)


def test_split_keeps_unique_leave_one_out_order():
    user_sequences = pd.DataFrame(
        {
            "user_idx": [1],
            "user_id": ["u1"],
            "item_seq_idx": [[10, 20, 30, 40, 50]],
            "engagement_seq": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "watch_signal_seq": [[True, True, True, True, True]],
        }
    )

    train_df, val_df, test_df = split_leave_one_out(user_sequences)

    assert train_df.loc[0, "item_sequence"] == [10, 20, 30]
    assert val_df.loc[0, "item_sequence"] == [10, 20, 30]
    assert val_df.loc[0, "target_item"] == 40
    assert test_df.loc[0, "item_sequence"] == [10, 20, 30, 40]
    assert test_df.loc[0, "target_item"] == 50


def test_full_sort_rejects_target_in_history(tmp_path: Path):
    csv_path = tmp_path / "eval.csv"
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["1 2 3"],
            "target_item": [2],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError, match="target item appears in history"):
        FullSortEvalDataset(str(csv_path), n_items=5, max_len=5)
