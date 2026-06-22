import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import split_leave_one_out


# ---------------------------------------------------------------------------
# split_leave_one_out: duplicate detection
# ---------------------------------------------------------------------------


def test_split_leave_one_out_rejects_duplicate_sequence():
    """If a user's sequence has duplicates, split_leave_one_out must raise."""
    row = pd.Series({
        "user_idx": 1,
        "user_id": "u1",
        "item_seq_idx": [1, 2, 3, 2, 4],  # 2 appears twice
        "engagement_seq": [0.5, 0.5, 0.5, 0.5, 0.5],
        "watch_signal_seq": [True, True, True, True, True],
    })
    user_sequences = pd.DataFrame([row])
    with pytest.raises(ValueError, match="duplicate items"):
        split_leave_one_out(user_sequences)


def test_split_leave_one_out_accepts_distinct_sequence():
    """A sequence with all-distinct items is accepted."""
    row = pd.Series({
        "user_idx": 1,
        "user_id": "u1",
        "item_seq_idx": [10, 20, 30, 40, 50],
        "engagement_seq": [0.5, 0.5, 0.5, 0.5, 0.5],
        "watch_signal_seq": [True, True, True, True, True],
    })
    user_sequences = pd.DataFrame([row])
    train_df, val_df, test_df = split_leave_one_out(user_sequences)
    assert len(train_df) == 1
    assert len(val_df) == 1
    assert len(test_df) == 1
    # Targets
    assert val_df.iloc[0]["target_item"] == 40  # second-to-last
    assert test_df.iloc[0]["target_item"] == 50  # last
    # Train history for this user
    assert train_df.iloc[0]["item_sequence"] == [10, 20, 30]


# ---------------------------------------------------------------------------
# Dedup behavior
# ---------------------------------------------------------------------------


def test_drop_duplicates_keeps_first_interaction_per_course():
    """When (user, item) appears multiple times, keep the FIRST created_at only.

    Simulate the dedup step directly (without running full preprocess).
    """
    implicit = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1"],
        "item_id": ["a", "a", "b", "c"],
        "created_at": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
        ]),
    })
    implicit_sorted = implicit.sort_values("created_at", kind="stable")
    implicit_dedup = implicit_sorted.drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    # (u1, a) appears twice; first keeps at 2024-01-01
    assert len(implicit_dedup) == 3
    a_row = implicit_dedup[implicit_dedup["item_id"] == "a"].iloc[0]
    assert a_row["created_at"] == pd.Timestamp("2024-01-01")
    # All items preserved (a kept, b kept, c kept)
    assert set(implicit_dedup["item_id"]) == {"a", "b", "c"}


def test_repeated_interaction_removed_count_is_nonzero():
    """When duplicates exist, repeat_events_removed should be > 0."""
    implicit = pd.DataFrame({
        "user_id": ["u1"] * 5,
        "item_id": ["a", "a", "b", "b", "c"],
        "created_at": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"
        ]),
    })
    implicit_sorted = implicit.sort_values("created_at", kind="stable")
    implicit_dedup = implicit_sorted.drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    removed = len(implicit) - len(implicit_dedup)
    assert removed == 2  # one duplicate a, one duplicate b


# ---------------------------------------------------------------------------
# Evaluation: target unmask removed
# ---------------------------------------------------------------------------


def test_full_sort_eval_does_not_unmask_target(tmp_path):
    """Under next-distinct-course, the target is naturally not in history.

    Even if a user sequence has a target in the prior history (which violates
    the dedup invariant), the dataset must NOT unmask it. This catches any
    accidental re-introduction of the unmask line.
    """
    from pipeline.loaders import FullSortEvalDataset

    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(
        "user_idx,item_sequence,target_item,sequence_length\n"
        "1,1 2 3,4,3\n"
    )
    dataset = FullSortEvalDataset(str(csv_path), n_items=10, max_len=10)

    sample = dataset[0]
    history_mask = sample["history_mask"]

    # Items 1, 2, 3 are in history and should be masked
    assert history_mask[1].item() is True
    assert history_mask[2].item() is True
    assert history_mask[3].item() is True
    # Item 4 is the target and should NOT be unmasked (the unmask line is removed)
    assert history_mask[4].item() is False
    # Item 5+ are not in history and should be rankable
    assert history_mask[5].item() is False


def test_full_sort_eval_target_unmasked_only_by_dedup_property(tmp_path):
    """The target is not in history by the time the dataset sees it
    (preprocessing dedup). This test verifies the natural property: if
    target were in history, the dataset would NOT unmask it.
    """
    from pipeline.loaders import FullSortEvalDataset

    csv_path = tmp_path / "test_data.csv"
    # Synthesize a row where target IS in history (this should not happen
    # under correct preprocessing, but tests the data structure's behavior)
    csv_path.write_text(
        "user_idx,item_sequence,target_item,sequence_length\n"
        "1,1 2 3 4 5,3,5\n"  # target=3 is in history
    )
    dataset = FullSortEvalDataset(str(csv_path), n_items=10, max_len=10)
    sample = dataset[0]
    history_mask = sample["history_mask"]

    # Item 3 is in history AND is the target — it's masked (no special handling)
    assert history_mask[3].item() is True
