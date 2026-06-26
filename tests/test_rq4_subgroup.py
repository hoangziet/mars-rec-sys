import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_subgroup


def test_derive_subgroups_reads_engagement_sequence_column(tmp_path):
    """Engagement strength buckets: high (>=0.66), mid (>=0.33), low (<0.33).

    Three cases:
        User 1: mean engagement 0.75 -> high_engagement
        User 2: mean engagement 0.0 -> low_engagement
        User 3: mean engagement 0.5 -> mid_engagement
    """
    data_dir = tmp_path / "data"
    splits = data_dir / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_idx": [1, 2, 3],
            "item_sequence": ["10 20", "30 40 50", "60 70"],
            "engagement_sequence": ["0.8 0.7", "0.0 0.0", "0.4 0.6"],
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

    # User 1: mean engagement (0.8+0.7)/2 = 0.75 >= 0.66 -> high_engagement
    assert subgroups["mean_engagement_bucket"][1] == "high_engagement"
    # User 2: mean engagement 0.0 < 0.33 -> low_engagement
    assert subgroups["mean_engagement_bucket"][2] == "low_engagement"
    # User 3: mean engagement (0.4+0.6)/2 = 0.5 >= 0.33 -> mid_engagement
    assert subgroups["mean_engagement_bucket"][3] == "mid_engagement"


def test_derive_subgroups_no_engagement_column_marks_unavailable(tmp_path):
    """If engagement_sequence column is missing, all users get 'unavailable'."""
    data_dir = tmp_path / "data"
    splits = data_dir / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_idx": [1, 2],
            "item_sequence": ["10 20", "30 40"],
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

    assert subgroups["mean_engagement_bucket"][1] == "unavailable"
    assert subgroups["mean_engagement_bucket"][2] == "unavailable"
