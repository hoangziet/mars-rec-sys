import pandas as pd

from data.preprocess import (
    attach_confidence,
    build_confidence_lookup,
    build_user_sequences,
    load_explicit_ratings,
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


def test_train_confidence_sequence_should_match_train_history_convention():
    item_seq = [101, 102, 103, 104]
    confidence_seq = [1.0, 1.2, 1.5, 1.0]

    train_history = item_seq[:-2]
    validation_target = item_seq[-2]
    train_confidence_history = confidence_seq[:-2]
    validation_confidence = confidence_seq[-2]

    assert train_history == [101, 102]
    assert validation_target == 103
    assert train_confidence_history == [1.0, 1.2]
    assert validation_confidence == 1.5
