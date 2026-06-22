import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_subgroup


def test_derive_subgroups_reads_watch_signal_sequence_column(tmp_path):
    """The CSV column is named `watch_signal_sequence` (space-joined 0/1),
    not `has_watch_signal`. Subgroup lookup must read the real column.

    Three cases:
        User 1: at least one positive signal -> has_watch=True
        User 2: all-zero watch_signal_sequence (no events recorded) -> has_watch=False
        User 3: short history but positive signal -> has_watch=True
    """
    data_dir = tmp_path / "data"
    splits = data_dir / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_idx": [1, 2, 3],
            "item_sequence": ["10 20", "30 40 50", "60 70"],
            "engagement_sequence": ["0.0 0.5", "0.0 0.0", "0.4 0.0"],
            "watch_signal_sequence": ["0 1", "0 0", "1 0"],
            "sequence_length": [2, 3, 2],
        }
    ).to_csv(splits / "train_sequences.csv", index=False)
    item_features = data_dir / "item_features"
    item_features.mkdir()
    pd.DataFrame(
        {
            "item_idx": [10, 20, 30, 40, 50, 60, 70],
            "item_id": ["a", "b", "c", "d", "e", "f", "g"],
            "title": ["t"] * 7,
            "description": ["d"] * 7,
            "text": ["t [SEP] d"] * 7,
            "language": ["fr"] * 7,
            "difficulty": ["beginner"] * 7,
            "theme": ["x"] * 7,
            "software": ["y"] * 7,
            "job": ["z"] * 7,
            "type": ["course"] * 7,
            "duration": [10] * 7,
        }
    ).to_csv(item_features / "item_metadata.csv", index=False)

    subgroups = rq4_subgroup._derive_subgroups(data_dir)

    # User 1: at least one positive signal in watch_signal_sequence
    assert subgroups["has_watch"][1] is True
    # User 2: all zeros — no watch event recorded for any interaction
    # (old fallback `any(e > 0)` would have said False here, but new code is stricter)
    assert subgroups["has_watch"][2] is False
    # User 3: at least one positive signal
    assert subgroups["has_watch"][3] is True


def test_derive_subgroups_no_watch_signal_column_distinguishes_real_missing(tmp_path):
    """If the column is missing entirely, all users have has_watch=False."""
    data_dir = tmp_path / "data"
    splits = data_dir / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_idx": [1, 2],
            "item_sequence": ["10 20", "30 40"],
            "engagement_sequence": ["0.0 0.5", "0.0 0.0"],
            "sequence_length": [2, 2],
        }
    ).to_csv(splits / "train_sequences.csv", index=False)
    item_features = data_dir / "item_features"
    item_features.mkdir()
    pd.DataFrame(
        {
            "item_idx": [10, 20, 30, 40],
            "item_id": ["a", "b", "c", "d"],
            "title": ["t"] * 4,
            "description": ["d"] * 4,
            "text": ["t [SEP] d"] * 4,
            "language": [""] * 4,
            "difficulty": [""] * 4,
            "theme": [""] * 4,
            "software": [""] * 4,
            "job": [""] * 4,
            "type": [""] * 4,
            "duration": [0] * 4,
        }
    ).to_csv(item_features / "item_metadata.csv", index=False)

    subgroups = rq4_subgroup._derive_subgroups(data_dir)

    assert subgroups["has_watch"][1] is False
    assert subgroups["has_watch"][2] is False
