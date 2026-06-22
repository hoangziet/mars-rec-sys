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


def test_split_leave_one_out_emits_new_sequence_schema():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u2",
                "user_idx": 7,
                "item_seq_idx": [101, 102, 103, 104],
                "engagement_seq": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    )

    train_df, val_df, test_df = preprocess.split_leave_one_out(user_sequences)

    assert train_df.columns.tolist() == [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
        "sequence_length",
    ]
    assert val_df.columns.tolist() == [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
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
