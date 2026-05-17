import pytest
import pandas as pd

from data.preprocess import (
    attach_confidence,
    build_confidence_lookup,
    build_dataset_stats,
    build_user_sequences,
    filter_benchmark_safe_user_sequences,
    load_explicit_ratings,
    split_leave_one_out,
    summarize_sequence_lengths,
)


def test_load_explicit_ratings_clips_watch_percentage_and_rating(tmp_path):
    csv_path = tmp_path / "explicit_ratings.csv"
    csv_path.write_text(
        '"user_id","item_id","watch_percentage","created_at","rating"\n'
        '"1","10","125","2024-01-01 00:00:00","11"\n'
        '"2","20","40","2024-01-02 00:00:00","4"\n'
    )

    explicit = load_explicit_ratings(csv_path)

    assert explicit.loc[0, "watch_percentage"] == 100
    assert explicit.loc[0, "rating"] == 10
    assert explicit.loc[1, "watch_percentage"] == 40
    assert explicit.loc[1, "rating"] == 4


def test_load_explicit_ratings_tolerates_extra_quote_in_rating(tmp_path):
    csv_path = tmp_path / "explicit_ratings.csv"
    csv_path.write_text(
        '"user_id","item_id","watch_percentage","created_at","rating"\n'
        '"277","62096","6","2022-09-29 16:26:14","1""\n'
    )

    explicit = load_explicit_ratings(csv_path)

    assert len(explicit) == 1
    assert explicit.loc[0, "rating"] == 1
    assert explicit.loc[0, "watch_percentage"] == 6


def test_build_confidence_lookup_uses_one_plus_alpha_watch_ratio():
    explicit = pd.DataFrame(
        {
            "user_id": ["1", "1", "2"],
            "item_id": ["10", "20", "30"],
            "watch_percentage": [100, 50, 0],
        }
    )

    lookup = build_confidence_lookup(explicit, alpha=0.5)

    assert lookup.loc[
        (lookup["user_id"] == "1") & (lookup["item_id"] == "10"),
        "confidence",
    ].item() == 1.5
    assert lookup.loc[
        (lookup["user_id"] == "1") & (lookup["item_id"] == "20"),
        "confidence",
    ].item() == 1.25
    assert lookup.loc[
        (lookup["user_id"] == "2") & (lookup["item_id"] == "30"),
        "confidence",
    ].item() == 1.0


def test_build_confidence_lookup_keeps_max_confidence_per_user_item():
    explicit = pd.DataFrame(
        {
            "user_id": ["1", "1"],
            "item_id": ["10", "10"],
            "watch_percentage": [20, 100],
        }
    )

    lookup = build_confidence_lookup(explicit, alpha=0.5)

    assert len(lookup) == 1
    assert lookup.loc[0, "confidence"] == 1.5


def test_attach_confidence_uses_baseline_for_missing_explicit_pair():
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1"],
            "item_id": ["10", "20"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    confidence_lookup = pd.DataFrame(
        {
            "user_id": ["1"],
            "item_id": ["10"],
            "confidence": [1.5],
        }
    )

    result = attach_confidence(implicit, confidence_lookup, baseline=1.0)

    assert result.loc[result["item_id"] == "10", "confidence"].item() == 1.5
    assert result.loc[result["item_id"] == "20", "confidence"].item() == 1.0


def test_build_user_sequences_keeps_items_and_confidence_aligned_by_time():
    implicit = pd.DataFrame(
        {
            "user_id": ["1", "1", "1"],
            "item_id": ["30", "10", "20"],
            "created_at": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"]
            ),
            "confidence": [1.3, 1.1, 1.2],
        }
    )

    sequences = build_user_sequences(implicit)

    assert sequences.loc[0, "item_sequence"] == ["10", "20", "30"]
    assert sequences.loc[0, "confidence_sequence"] == [1.1, 1.2, 1.3]
    assert sequences.loc[0, "seq_len"] == 3


def test_filter_benchmark_safe_user_sequences_requires_min_original_len_four():
    implicit = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "created_at": pd.Timestamp("2024-01-01"), "confidence": 1.0},
            {"user_id": "u1", "item_id": "i2", "created_at": pd.Timestamp("2024-01-02"), "confidence": 1.0},
            {"user_id": "u1", "item_id": "i3", "created_at": pd.Timestamp("2024-01-03"), "confidence": 1.0},
            {"user_id": "u2", "item_id": "i1", "created_at": pd.Timestamp("2024-01-01"), "confidence": 1.0},
            {"user_id": "u2", "item_id": "i2", "created_at": pd.Timestamp("2024-01-02"), "confidence": 1.0},
            {"user_id": "u2", "item_id": "i3", "created_at": pd.Timestamp("2024-01-03"), "confidence": 1.0},
            {"user_id": "u2", "item_id": "i4", "created_at": pd.Timestamp("2024-01-04"), "confidence": 1.0},
        ]
    )

    user_sequences = build_user_sequences(implicit)
    eligible = filter_benchmark_safe_user_sequences(user_sequences)

    assert eligible["user_id"].tolist() == ["u2"]
    assert eligible.loc[0, "item_sequence"] == ["i1", "i2", "i3", "i4"]
    assert eligible.loc[0, "confidence_sequence"] == [1.0, 1.0, 1.0, 1.0]


def test_split_leave_one_out_preserves_train_history_and_confidence_alignment():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u2",
                "item_seq_idx": [101, 102, 103, 104],
                "confidence_seq": [1.0, 1.2, 1.5, 1.0],
                "user_idx": 7,
            }
        ]
    )

    train_df, val_df, test_df = split_leave_one_out(user_sequences)

    assert train_df.loc[0, "user_idx"] == 7
    assert train_df.loc[0, "item_sequence"] == [101, 102]
    assert train_df.loc[0, "seq_len"] == 2
    assert train_df.loc[0, "target"] == 103
    assert train_df.loc[0, "confidence"] == 1.5
    assert train_df.loc[0, "confidence_sequence"] == [1.0, 1.2]
    assert val_df.loc[0, "train_seq"] == [101, 102]
    assert val_df.loc[0, "target"] == 103
    assert test_df.loc[0, "train_seq"] == [101, 102, 103]
    assert test_df.loc[0, "target"] == 104


def test_empty_benchmark_safe_user_set_has_robust_summary_and_stats():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_sequence": ["i1", "i2", "i3"],
                "confidence_sequence": [1.0, 1.0, 1.0],
                "seq_len": 3,
            }
        ]
    )

    eligible = filter_benchmark_safe_user_sequences(user_sequences)

    assert eligible.empty
    assert summarize_sequence_lengths(eligible) == "Seq len - min: n/a, max: n/a, mean: n/a"

    stats = build_dataset_stats(
        n_users=0,
        n_items=0,
        n_interactions=0,
        min_seq_len=4,
        min_item_freq=3,
    )

    assert stats["min_seq_len"] == 4
    assert stats["sparsity"] == 0.0


def test_filter_benchmark_safe_user_sequences_uses_actual_sequence_length_not_stale_seq_len():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_sequence": ["i1", "i2", "i3"],
                "confidence_sequence": [1.0, 1.0, 1.0],
                "seq_len": 99,
            },
            {
                "user_id": "u2",
                "item_sequence": ["i1", "i2", "i3", "i4"],
                "confidence_sequence": [1.0, 1.1, 1.2, 1.3],
                "seq_len": 1,
            },
        ]
    )

    eligible = filter_benchmark_safe_user_sequences(user_sequences)

    assert eligible["user_id"].tolist() == ["u2"]
    assert eligible.loc[0, "seq_len"] == 4


def test_split_leave_one_out_raises_on_item_and_confidence_length_mismatch():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u2",
                "item_seq_idx": [101, 102, 103, 104],
                "confidence_seq": [1.0, 1.2, 1.5],
                "user_idx": 7,
            }
        ]
    )

    with pytest.raises(ValueError, match="confidence sequence length mismatch"):
        split_leave_one_out(user_sequences)


def test_split_leave_one_out_raises_on_sequence_shorter_than_benchmark_safe_minimum():
    user_sequences = pd.DataFrame(
        [
            {
                "user_id": "u2",
                "item_seq_idx": [101, 102, 103],
                "confidence_seq": [1.0, 1.2, 1.5],
                "user_idx": 7,
            }
        ]
    )

    with pytest.raises(ValueError, match="benchmark-safe minimum"):
        split_leave_one_out(user_sequences)


def test_split_leave_one_out_returns_empty_dataframes_with_expected_schema():
    train_df, val_df, test_df = split_leave_one_out(pd.DataFrame())

    assert train_df.empty
    assert val_df.empty
    assert test_df.empty
    assert train_df.columns.tolist() == [
        "user_idx",
        "item_sequence",
        "seq_len",
        "target",
        "confidence",
        "confidence_sequence",
    ]
    assert val_df.columns.tolist() == ["user_idx", "train_seq", "target"]
    assert test_df.columns.tolist() == ["user_idx", "train_seq", "target"]
