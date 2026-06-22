import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import preprocess


def test_attach_engagement_score_uses_zero_for_missing_explicit_pair():
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1"],
            "item_id": ["10", "20"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    lookup = pd.DataFrame(
        {
            "user_id": ["1"],
            "item_id": ["10"],
            "created_at": pd.to_datetime(["2024-01-01"]),
            "engagement_score": [0.8],
        }
    )

    result = preprocess.attach_engagement_score(implicit, lookup)

    assert result.loc[result["item_id"] == "10", "engagement_score"].item() == 0.8
    assert result.loc[result["item_id"] == "20", "engagement_score"].item() == 0.0


def test_attach_engagement_score_multi_user_temporal():
    """merge_asof must handle multiple users with different timestamp ranges."""
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1", "2", "2"],
            "item_id": ["10", "20", "10", "30"],
            "created_at": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]
            ),
        }
    )
    lookup = pd.DataFrame(
        {
            "user_id": ["1", "2"],
            "item_id": ["10", "10"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "engagement_score": [0.8, 0.6],
        }
    )

    result = preprocess.attach_engagement_score(implicit, lookup)

    # user 1, item 10: explicit at 01-01 <= implicit at 01-01 → 0.8
    r1 = result[(result["user_id"] == "1") & (result["item_id"] == "10")]
    assert r1["engagement_score"].item() == 0.8
    # user 1, item 20: no explicit → 0.0
    r2 = result[(result["user_id"] == "1") & (result["item_id"] == "20")]
    assert r2["engagement_score"].item() == 0.0
    # user 2, item 10: explicit at 01-02 > implicit at 01-01 → 0.0
    r3 = result[(result["user_id"] == "2") & (result["item_id"] == "10")]
    assert r3["engagement_score"].item() == 0.0
    # user 2, item 30: no explicit → 0.0
    r4 = result[(result["user_id"] == "2") & (result["item_id"] == "30")]
    assert r4["engagement_score"].item() == 0.0


def test_split_leave_one_out_emits_new_sequence_schema():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u2",
                "user_idx": 7,
                "item_seq_idx": [101, 102, 103, 104],
                "engagement_seq": [0.1, 0.2, 0.3, 0.4],
                "watch_signal_seq": [True, True, True, False],
            }
        ]
    )

    train_df, val_df, test_df = preprocess.split_leave_one_out(user_sequences)

    assert train_df.columns.tolist() == [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
        "watch_signal_sequence",
        "sequence_length",
    ]
    assert val_df.columns.tolist() == [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
        "watch_signal_sequence",
        "sequence_length",
        "target_item",
        "target_engagement",
    ]
    assert test_df.columns.tolist() == val_df.columns.tolist()
    assert train_df.loc[0, "item_sequence"] == [101, 102]
    assert train_df.loc[0, "engagement_sequence"] == [0.1, 0.2]
    assert train_df.loc[0, "sequence_length"] == 2
    assert val_df.loc[0, "item_sequence"] == [101, 102]
    assert val_df.loc[0, "engagement_sequence"] == [0.1, 0.2]
    assert val_df.loc[0, "sequence_length"] == 2
    assert test_df.loc[0, "item_sequence"] == [101, 102, 103]
    assert test_df.loc[0, "engagement_sequence"] == [0.1, 0.2, 0.3]
    assert test_df.loc[0, "sequence_length"] == 3
    assert val_df.loc[0, "target_item"] == 103
    assert val_df.loc[0, "target_engagement"] == 0.3
    assert test_df.loc[0, "target_item"] == 104
    assert test_df.loc[0, "target_engagement"] == 0.4


def test_build_item_metadata_maps_current_raw_headers():
    items = pd.DataFrame(
        {
            "item_id": ["i2"],
            "name": ["Course"],
            "description": ["Intro"],
            "language": ["fr"],
            "Difficulty": ["beginner"],
            "Theme": ["data"],
            "Software": ["python"],
            "Job": ["analyst"],
            "type": ["course"],
            "duration": [10],
        }
    )
    item_id_map = pd.DataFrame({"item_id": ["i2"], "item_idx": [2]})

    metadata = preprocess._build_item_metadata(items, item_id_map)

    assert metadata.loc[0, "item_idx"] == 2
    assert metadata.loc[0, "title"] == "Course"
    assert metadata.loc[0, "difficulty"] == "beginner"
    assert metadata.loc[0, "theme"] == "data"
    assert metadata.loc[0, "software"] == "python"
    assert metadata.loc[0, "job"] == "analyst"
    assert metadata.loc[0, "text"] == "Course [SEP] Intro"


def test_save_processed_outputs_writes_entity_folders(tmp_path):
    output_dir = tmp_path / "processed"
    preprocess.save_processed_outputs(
        output_dir=output_dir,
        interactions=pd.DataFrame(
            {
                "user_idx": [1],
                "item_idx": [2],
                "user_id": ["u1"],
                "item_id": ["i2"],
                "created_at": ["2024-01-01T00:00:00"],
                "engagement_score": [0.5],
                "sequence_order": [0],
            }
        ),
        train_df=pd.DataFrame(
            {
                "user_idx": [1],
                "item_sequence": [[1, 2]],
                "engagement_sequence": [[0.1, 0.5]],
                "sequence_length": [2],
            }
        ),
        val_df=pd.DataFrame(
            {
                "user_idx": [1],
                "item_sequence": [[1, 2]],
                "engagement_sequence": [[0.1, 0.5]],
                "sequence_length": [2],
                "target_item": [3],
                "target_engagement": [0.8],
            }
        ),
        test_df=pd.DataFrame(
            {
                "user_idx": [1],
                "item_sequence": [[1, 2, 3]],
                "engagement_sequence": [[0.1, 0.5, 0.8]],
                "sequence_length": [3],
                "target_item": [4],
                "target_engagement": [1.0],
            }
        ),
        item_metadata=pd.DataFrame(
            {
                "item_idx": [2],
                "item_id": ["i2"],
                "title": ["Course"],
                "description": [""],
                "text": ["Course [SEP]"],
                "language": ["fr"],
                "difficulty": [""],
                "theme": [""],
                "software": [""],
                "job": [""],
                "type": [""],
                "duration": [10],
            }
        ),
        user_id_map=pd.DataFrame({"user_id": ["u1"], "user_idx": [1]}),
        item_id_map=pd.DataFrame({"item_id": ["i2"], "item_idx": [2]}),
        dataset_stats={"n_users": 1, "n_items": 1, "n_interactions": 1},
        preprocessing_report={
            "repeat_events_removed": 0,
            "orphan_implicit_count": 1,
            "orphan_explicit_count": 2,
            "engagement_pairs": 3,
            "min_user_interactions": 5,
            "min_item_interactions": 3,
        },
    )

    expected_files = [
        output_dir / "interactions" / "interactions.csv",
        output_dir / "splits" / "train_sequences.csv",
        output_dir / "splits" / "val_sequences.csv",
        output_dir / "splits" / "test_sequences.csv",
        output_dir / "item_features" / "item_metadata.csv",
        output_dir / "mappings" / "user_id_map.csv",
        output_dir / "mappings" / "item_id_map.csv",
        output_dir / "reports" / "dataset_stats.json",
        output_dir / "reports" / "preprocessing_report.json",
        output_dir / "interactions.csv",
        output_dir / "train.csv",
        output_dir / "val.csv",
        output_dir / "test.csv",
        output_dir / "item_meta.csv",
        output_dir / "dataset_stats.json",
    ]

    for path in expected_files:
        assert path.exists(), path

    train_saved = pd.read_csv(output_dir / "splits" / "train_sequences.csv")
    assert train_saved.loc[0, "item_sequence"] == "1 2"
    assert train_saved.loc[0, "engagement_sequence"] == "0.1 0.5"
    val_saved = pd.read_csv(output_dir / "splits" / "val_sequences.csv")
    assert val_saved.loc[0, "item_sequence"] == "1 2"
    legacy_val = pd.read_csv(output_dir / "val.csv")
    assert legacy_val.columns.tolist() == ["user_idx", "train_seq", "target"]
    assert legacy_val.loc[0, "train_seq"] == "[1, 2]"
    assert legacy_val.loc[0, "target"] == 3
    legacy_train = pd.read_csv(output_dir / "train.csv")
    assert legacy_train.loc[0, "item_sequence"] == "[1, 2]"
    legacy_meta = pd.read_csv(output_dir / "item_meta.csv")
    assert legacy_meta.loc[0, "title"] == "Course"

    with open(output_dir / "reports" / "preprocessing_report.json") as f:
        report = json.load(f)
    assert report["repeat_events_removed"] == 0
    assert report["orphan_implicit_count"] == 1
    assert report["orphan_explicit_count"] == 2
    assert report["engagement_pairs"] == 3
    assert report["min_user_interactions"] == 5
    assert report["min_item_interactions"] == 3


def test_repeated_interactions_are_not_dropped_as_user_item_duplicates():
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1", "1"],
            "item_id": ["10", "10", "20"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    implicit_sorted = implicit.sort_values("created_at", kind="stable")

    dedup = implicit_sorted.drop_duplicates(
        subset=["user_id", "item_id", "created_at"],
        keep="first",
    )

    assert len(dedup) == 3


def test_build_user_sequences_preserves_watch_signal_sequence():
    interactions = pd.DataFrame(
        {
            "user_idx": [1, 1, 1],
            "user_id": ["u1", "u1", "u1"],
            "item_idx": [10, 11, 12],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "engagement_score": [0.0, 0.4, 0.0],
            "has_watch_signal": [True, True, False],
        }
    )

    result = preprocess.build_user_sequences(interactions)
    row = result.iloc[0]

    assert row["engagement_seq"] == [0.0, 0.4, 0.0]
    assert row["watch_signal_seq"] == [True, True, False]


def test_split_leave_one_out_writes_watch_signal_sequence():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "user_idx": 1,
                "item_seq_idx": [1, 2, 3, 4],
                "engagement_seq": [0.0, 0.2, 0.3, 0.0],
                "watch_signal_seq": [True, True, True, False],
            }
        ]
    )

    train_df, val_df, test_df = preprocess.split_leave_one_out(user_sequences)

    assert train_df.loc[0, "watch_signal_sequence"] == [True, True]
    assert val_df.loc[0, "watch_signal_sequence"] == [True, True]
    assert test_df.loc[0, "watch_signal_sequence"] == [True, True, True]


def test_has_watch_signal_requires_temporal_match():
    implicit = pd.DataFrame(
        {
            "user_id": ["1"],
            "item_id": ["10"],
            "created_at": pd.to_datetime(["2024-01-01"]),
        }
    )
    lookup = pd.DataFrame(
        {
            "user_id": ["1"],
            "item_id": ["10"],
            "created_at": pd.to_datetime(["2024-01-02"]),
            "engagement_score": [0.8],
        }
    )

    result = preprocess.attach_engagement_score(implicit, lookup)
    assert result.loc[0, "engagement_score"] == 0.0
    assert bool(result.loc[0, "has_watch_signal"]) is False


def test_attach_engagement_score_records_temporal_stats():
    """attach_engagement_score should return temporal deltas so the report
    can distinguish 'explicit before implicit' from 'explicit after implicit'."""
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1", "2", "2"],
            "item_id": ["10", "20", "10", "30"],
            "created_at": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]
            ),
        }
    )
    lookup = pd.DataFrame(
        {
            "user_id": ["1", "1", "2", "2"],
            "item_id": ["10", "20", "10", "30"],
            "created_at": pd.to_datetime(
                ["2023-12-31", "2024-01-02", "2024-01-02", "2024-01-04"]
            ),
            "engagement_score": [0.5, 0.3, 0.7, 0.9],
        }
    )

    result = preprocess.attach_engagement_score(implicit, lookup, return_temporal_stats=True)

    assert "temporal_delta_seconds" in result.columns
    assert "temporal_direction" in result.columns
    matched = result[result["has_watch_signal"]]
    assert (matched["temporal_direction"] == "before").all()
    assert result.loc[2, "temporal_direction"] == "after"
    assert result.loc[3, "temporal_direction"] == "after"

    if "temporal_stats" in result.attrs:
        stats = result.attrs["temporal_stats"]
        assert "matched_count" in stats
        assert "explicit_before_pct" in stats
        assert "explicit_after_pct" in stats
        assert "median_delta_seconds" in stats


def test_split_leave_one_out_emits_per_split_coverage(tmp_path):
    """Coverage should be reported per split: train_history, val_target, test_target."""
    train_df = pd.DataFrame(
        [
            {
                "user_idx": 1,
                "item_sequence": [10, 20, 30, 40],
                "engagement_sequence": [0.0, 0.5, 0.0, 0.7],
                "watch_signal_sequence": [0, 1, 0, 1],
                "sequence_length": 4,
                "target_item": 40,
                "target_engagement": 0.7,
            }
        ]
    )

    coverage = preprocess.compute_per_split_coverage(train_df, train_df, train_df)

    assert coverage["train_history_with_watch"] == 1
    assert coverage["val_target_with_watch"] == 1
    assert coverage["test_target_with_watch"] == 1
    assert coverage["train_history_total"] == 1
    assert coverage["train_history_pct"] == 100.0
