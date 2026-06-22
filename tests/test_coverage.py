import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import compute_per_split_coverage, split_leave_one_out


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_user_sequences():
    """Build a small user_sequences DataFrame with known watch signals.

    User 1: seq=[10, 20, 30, 40, 50], watch=[T, F, T, T, F]
        -> train history = [10, 20, 30], val target = 40 (T), test target = 50 (F)
    User 2: seq=[100, 200, 300, 400, 500], watch=[T, T, F, T, T]
        -> train history = [100, 200, 300], val target = 400 (T), test target = 500 (T)
    """
    return pd.DataFrame([
        {
            "user_idx": 1,
            "user_id": "u1",
            "item_seq_idx": [10, 20, 30, 40, 50],
            "engagement_seq": [0.5, 0.0, 0.7, 0.6, 0.0],
            "watch_signal_seq": [True, False, True, True, False],
        },
        {
            "user_idx": 2,
            "user_id": "u2",
            "item_seq_idx": [100, 200, 300, 400, 500],
            "engagement_seq": [0.5, 0.6, 0.0, 0.7, 0.8],
            "watch_signal_seq": [True, True, False, True, True],
        },
    ])


def _make_splits(user_sequences):
    return split_leave_one_out(user_sequences)


# ---------------------------------------------------------------------------
# Target signal: not history's last element
# ---------------------------------------------------------------------------


def test_target_with_watch_reads_target_column():
    """Coverage must use target_has_watch_signal column, not the last history element."""
    user_sequences = _make_user_sequences()
    train_df, val_df, test_df = _make_splits(user_sequences)

    # Sanity: val/test rows now have target_has_watch_signal
    assert "target_has_watch_signal" in val_df.columns
    assert "target_has_watch_signal" in test_df.columns

    cov = compute_per_split_coverage(train_df, val_df, test_df, user_sequences)

    # User 1: val_target=40 (T), test_target=50 (F) -> 1/2 with watch
    # User 2: val_target=400 (T), test_target=500 (T) -> 2/2 with watch
    assert cov["val_target_with_watch"] == 2
    assert cov["val_target_total"] == 2
    assert cov["test_target_with_watch"] == 1
    assert cov["test_target_total"] == 2


def test_target_with_watch_old_csv_fallback():
    """Old CSVs (no target_has_watch_signal column) should still work via target_engagement fallback."""
    user_sequences = _make_user_sequences()
    train_df, val_df, test_df = _make_splits(user_sequences)
    # Remove the new column to simulate old CSV
    val_df = val_df.drop(columns=["target_has_watch_signal"])
    test_df = test_df.drop(columns=["target_has_watch_signal"])

    cov = compute_per_split_coverage(train_df, val_df, test_df, user_sequences)
    # User 1: val_target_engagement=0.6 (>0), test_target_engagement=0.0 (NOT >0) -> 1/2
    # User 2: val=0.7 (>0), test=0.8 (>0) -> 2/2
    assert cov["val_target_with_watch"] == 2
    assert cov["test_target_with_watch"] == 1


def test_target_with_watch_history_signal_ignored():
    """The bug: previously, history signal of last item was reported as target signal.

    Test that we DO NOT mistake history's last element for target.
    """
    # seq=[1, 2, 3, 4, 5], watch=[T, F, T, F, F]
    # train history = [1, 2, 3], val target = 4 (F), test target = 5 (F)
    # val watch_signal_sequence = [T, F, T] (item 5's watch excluded)
    # OLD code: val_target_with_watch = watch_signal_sequence[-1] = T (WRONG, this is item 3's watch)
    # NEW code: val_target_with_watch = target_has_watch_signal = F (item 4's watch)
    user_sequences = pd.DataFrame([{
        "user_idx": 1,
        "user_id": "u1",
        "item_seq_idx": [1, 2, 3, 4, 5],
        "engagement_seq": [0.5, 0.0, 0.7, 0.0, 0.0],
        "watch_signal_seq": [True, False, True, False, False],
    }])
    train_df, val_df, test_df = _make_splits(user_sequences)
    cov = compute_per_split_coverage(train_df, val_df, test_df, user_sequences)
    # val target is item 4 (watch=F); NEW code reports val_target_with_watch=0
    assert cov["val_target_with_watch"] == 0
    assert cov["val_target_total"] == 1


# ---------------------------------------------------------------------------
# Train sample coverage
# ---------------------------------------------------------------------------


def test_train_sample_coverage_counts_per_sample_targets():
    """Train sample coverage uses the watch signal of each sample's target,
    not the per-user aggregate."""
    user_sequences = _make_user_sequences()
    train_df, val_df, test_df = _make_splits(user_sequences)
    cov = compute_per_split_coverage(train_df, val_df, test_df, user_sequences)

    # User 1: watch=[T, F, T, T, F]; train_history=[10,20,30] (3 items);
    #   training samples target positions [1, 2] of seq (i.e., item 20 (F), item 30 (T))
    #   = 1/2 with watch
    # User 2: watch=[T, T, F, T, T]; train_history=[100,200,300];
    #   training samples target positions [1, 2] of seq (i.e., item 200 (T), item 300 (F))
    #   = 1/2 with watch
    # Total: 2 samples with watch out of 4 total = 50%
    assert cov["train_sample_total"] == 4
    assert cov["train_sample_with_watch"] == 2
    assert cov["train_sample_with_watch_pct"] == 50.0


def test_train_sample_coverage_handles_empty():
    """Empty user_sequences -> 0 coverage, no error."""
    user_sequences = pd.DataFrame(columns=[
        "user_idx", "user_id", "item_seq_idx", "engagement_seq", "watch_signal_seq"
    ])
    train_df = pd.DataFrame(columns=[
        "user_idx", "item_sequence", "engagement_sequence",
        "watch_signal_sequence", "sequence_length"
    ])
    val_df = pd.DataFrame(columns=[
        "user_idx", "item_sequence", "engagement_sequence",
        "watch_signal_sequence", "sequence_length",
        "target_item", "target_engagement", "target_has_watch_signal"
    ])
    test_df = val_df.copy()
    cov = compute_per_split_coverage(train_df, val_df, test_df, user_sequences)
    assert cov["train_sample_total"] == 0
    assert cov["train_sample_with_watch"] == 0
    assert cov["train_sample_with_watch_pct"] == 0.0
