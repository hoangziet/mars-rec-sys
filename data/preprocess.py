"""
data/preprocess.py
==================
Preprocess raw interaction data into the canonical processed dataset layout.

Usage:
    uv run python data/preprocess.py
"""

import csv
import json
from pathlib import Path

import pandas as pd


LEAVE_ONE_OUT_HOLDOUTS = 2
MIN_TRAIN_HISTORY_LEN_FOR_NEURAL_MODELS = 2
MIN_BENCHMARK_SAFE_SEQUENCE_LEN = (
    MIN_TRAIN_HISTORY_LEN_FOR_NEURAL_MODELS + LEAVE_ONE_OUT_HOLDOUTS
)
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 3


def load_explicit_ratings(path: Path) -> pd.DataFrame:
    """Load explicit ratings and derive normalized engagement scores."""
    explicit = pd.read_csv(
        path,
        quoting=csv.QUOTE_NONE,
        dtype=str,
    )
    explicit.columns = [col.strip('"') for col in explicit.columns]

    for col in explicit.columns:
        explicit[col] = explicit[col].str.strip().str.strip('"')

    numeric_cols = ["user_id", "item_id", "watch_percentage", "rating"]
    for col in numeric_cols:
        if col in explicit.columns:
            explicit[col] = pd.to_numeric(explicit[col], errors="coerce")

    explicit["created_at"] = pd.to_datetime(explicit["created_at"], errors="coerce")
    explicit = explicit.dropna(
        subset=["user_id", "item_id", "watch_percentage", "created_at"]
    ).copy()

    explicit["user_id"] = explicit["user_id"].astype(int).astype(str)
    explicit["item_id"] = explicit["item_id"].astype(int).astype(str)
    explicit["watch_percentage"] = explicit["watch_percentage"].clip(lower=0, upper=100)
    explicit["engagement_score"] = explicit["watch_percentage"] / 100.0

    return explicit[
        ["user_id", "item_id", "watch_percentage", "engagement_score", "created_at"]
    ].reset_index(drop=True)


def build_engagement_lookup(explicit: pd.DataFrame) -> pd.DataFrame:
    """Build engagement lookup with temporal alignment.

    Returns one row per explicit watch event with (user_id, item_id, created_at, engagement_score).
    Used by attach_engagement_score for temporal join.
    """
    explicit = explicit.copy()
    if "engagement_score" not in explicit.columns:
        explicit["engagement_score"] = (
            explicit["watch_percentage"].astype(float).clip(lower=0, upper=100) / 100.0
        )
    return explicit[["user_id", "item_id", "created_at", "engagement_score"]].copy()


def attach_engagement_score(
    interactions: pd.DataFrame,
    engagement_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach engagement score using temporal alignment.

    For each implicit interaction, finds the most recent explicit watch event
    for the same (user_id, item_id) with timestamp <= interaction timestamp.
    If no such event exists, engagement_score = 0.0.
    Also adds has_watch_signal (True if any explicit watch exists for this pair).
    """
    if engagement_lookup.empty:
        result = interactions.copy()
        result["engagement_score"] = 0.0
        result["has_watch_signal"] = False
        return result

    # Sort both by created_at for merge_asof (must be globally sorted)
    interactions_sorted = interactions.assign(
        _original_order=range(len(interactions))
    ).sort_values("created_at", kind="stable").reset_index(drop=True)

    engagement_sorted = engagement_lookup.dropna(
        subset=["created_at"]
    ).sort_values("created_at", kind="stable").reset_index(drop=True)

    if engagement_sorted.empty:
        result = interactions_sorted.drop(columns=["_original_order"])
        result["engagement_score"] = 0.0
        result["has_watch_signal"] = False
        return result.sort_values("_original_order").drop(columns=["_original_order"]) if "_original_order" in result.columns else result

    # merge_asof: for each interaction row, find the latest engagement event
    # with same (user_id, item_id) and engagement.created_at <= interaction.created_at
    merged = pd.merge_asof(
        interactions_sorted,
        engagement_sorted.rename(columns={"created_at": "engagement_at", "engagement_score": "engagement_score_raw"}),
        left_on="created_at",
        right_on="engagement_at",
        by=["user_id", "item_id"],
        direction="backward",
    )

    merged["engagement_score"] = merged["engagement_score_raw"].fillna(0.0).clip(lower=0.0, upper=1.0)

    # has_watch_signal: True if temporal match found (engagement_at is not NaT)
    merged["has_watch_signal"] = merged["engagement_at"].notna()

    # Restore original order
    merged = merged.sort_values("_original_order", kind="stable")
    merged = merged.drop(columns=["engagement_at", "engagement_score_raw", "_original_order"], errors="ignore")

    return merged


def build_user_sequences(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame(
            columns=["user_idx", "user_id", "item_seq_idx", "engagement_seq", "has_watch_signal"]
        )

    rows = []
    ordered = interactions.sort_values(["user_idx", "created_at", "item_idx"], kind="stable")
    for user_idx, group in ordered.groupby("user_idx", sort=True):
        has_watch = bool(group["has_watch_signal"].any()) if "has_watch_signal" in group.columns else False
        rows.append(
            {
                "user_idx": int(user_idx),
                "user_id": group["user_id"].iloc[0],
                "item_seq_idx": group["item_idx"].tolist(),
                "engagement_seq": group["engagement_score"].astype(float).tolist(),
                "has_watch_signal": has_watch,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["user_idx", "user_id", "item_seq_idx", "engagement_seq", "has_watch_signal"],
    )


def _get_sequence_length(row: pd.Series) -> int:
    if isinstance(row.get("item_seq_idx"), list):
        return len(row["item_seq_idx"])
    if "sequence_length" in row and pd.notna(row["sequence_length"]):
        return int(row["sequence_length"])
    return int(row["seq_len"])


def _get_engagement_sequence_length(row: pd.Series) -> int | None:
    if isinstance(row.get("engagement_seq"), list):
        return len(row["engagement_seq"])
    return None


def _validate_row_sequence_alignment(row: pd.Series) -> None:
    seq_len = _get_sequence_length(row)
    engagement_len = _get_engagement_sequence_length(row)
    if engagement_len is not None and seq_len != engagement_len:
        user_id = row.get("user_id", "<unknown>")
        raise ValueError(
            "engagement sequence length mismatch "
            f"for user_id={user_id}: items={seq_len}, engagement={engagement_len}"
        )


def filter_benchmark_safe_user_sequences(
    user_sequences: pd.DataFrame,
    min_seq_len: int = MIN_BENCHMARK_SAFE_SEQUENCE_LEN,
) -> pd.DataFrame:
    if user_sequences.empty:
        return user_sequences.reset_index(drop=True)

    sequence_lengths = user_sequences.apply(_get_sequence_length, axis=1)
    engagement_lengths = user_sequences.apply(_get_engagement_sequence_length, axis=1)
    mismatched = engagement_lengths.notna() & (sequence_lengths != engagement_lengths)
    if mismatched.any():
        _validate_row_sequence_alignment(user_sequences.loc[mismatched].iloc[0])

    filtered = user_sequences[sequence_lengths >= min_seq_len].copy()
    filtered["sequence_length"] = sequence_lengths[sequence_lengths >= min_seq_len].astype(int)
    return filtered.reset_index(drop=True)


def apply_iterative_k_core_filter(
    interactions: pd.DataFrame,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> pd.DataFrame:
    filtered = interactions.copy()
    while True:
        before_len = len(filtered)

        user_counts = filtered.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["user_id"].isin(valid_users)].copy()

        item_counts = filtered.groupby("item_id").size()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered["item_id"].isin(valid_items)].copy()

        if len(filtered) == before_len:
            return filtered.reset_index(drop=True)


def format_user_retention(after: int, before: int) -> str:
    if before == 0:
        return f"Users remaining: {after:,} / {before:,} (n/a)"
    return f"Users remaining: {after:,} / {before:,} ({after / before:.1%})"


def summarize_sequence_lengths(user_sequences: pd.DataFrame) -> str:
    if user_sequences.empty:
        return "Seq len - min: n/a, max: n/a, mean: n/a"

    lengths = user_sequences.apply(_get_sequence_length, axis=1)
    return (
        "Seq len - min: "
        f"{lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}"
    )


def split_leave_one_out(
    user_sequences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_columns = [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
        "sequence_length",
    ]
    eval_columns = [
        "user_idx",
        "item_sequence",
        "engagement_sequence",
        "sequence_length",
        "target_item",
        "target_engagement",
    ]

    if user_sequences.empty:
        return (
            pd.DataFrame(columns=train_columns),
            pd.DataFrame(columns=eval_columns),
            pd.DataFrame(columns=eval_columns),
        )

    train_rows, val_rows, test_rows = [], [], []
    for _, row in user_sequences.iterrows():
        _validate_row_sequence_alignment(row)
        seq = row["item_seq_idx"]
        engagement = row["engagement_seq"]
        n = len(seq)

        if n < MIN_BENCHMARK_SAFE_SEQUENCE_LEN:
            user_id = row.get("user_id", "<unknown>")
            raise ValueError(
                "sequence shorter than benchmark-safe minimum "
                f"for user_id={user_id}: length={n}, "
                f"minimum={MIN_BENCHMARK_SAFE_SEQUENCE_LEN}"
            )

        train_history = seq[:-2]
        train_engagement = engagement[:-2]
        val_target = seq[-2]
        val_target_engagement = engagement[-2]
        test_history = seq[:-1]
        test_engagement = engagement[:-1]
        test_target = seq[-1]
        test_target_engagement = engagement[-1]

        train_rows.append(
            {
                "user_idx": row["user_idx"],
                "item_sequence": train_history,
                "engagement_sequence": train_engagement,
                "sequence_length": len(train_history),
            }
        )
        val_rows.append(
            {
                "user_idx": row["user_idx"],
                "item_sequence": train_history,
                "engagement_sequence": train_engagement,
                "sequence_length": len(train_history),
                "target_item": val_target,
                "target_engagement": val_target_engagement,
            }
        )
        test_rows.append(
            {
                "user_idx": row["user_idx"],
                "item_sequence": test_history,
                "engagement_sequence": test_engagement,
                "sequence_length": len(test_history),
                "target_item": test_target,
                "target_engagement": test_target_engagement,
            }
        )

    return (
        pd.DataFrame(train_rows, columns=train_columns),
        pd.DataFrame(val_rows, columns=eval_columns),
        pd.DataFrame(test_rows, columns=eval_columns),
    )


def serialize_sequence(values: list[int] | list[float]) -> str:
    return " ".join(str(v) for v in values)


def serialize_legacy_sequence(values: list[int] | list[float]) -> str:
    return str(list(values))


def save_processed_outputs(
    *,
    output_dir: Path,
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_metadata: pd.DataFrame,
    user_id_map: pd.DataFrame,
    item_id_map: pd.DataFrame,
    dataset_stats: dict,
    preprocessing_report: dict,
) -> None:
    interactions_dir = output_dir / "interactions"
    splits_dir = output_dir / "splits"
    item_features_dir = output_dir / "item_features"
    mappings_dir = output_dir / "mappings"
    reports_dir = output_dir / "reports"

    for path in (
        interactions_dir,
        splits_dir,
        item_features_dir,
        mappings_dir,
        reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    def serialize_split_frame(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in ("item_sequence", "engagement_sequence"):
            if column in result.columns:
                result[column] = result[column].apply(
                    lambda values: serialize_sequence(values)
                    if isinstance(values, list)
                    else values
                )
        return result

    def serialize_legacy_split_frame(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in ("item_sequence", "engagement_sequence"):
            if column in result.columns:
                result[column] = result[column].apply(
                    lambda values: serialize_legacy_sequence(values)
                    if isinstance(values, list)
                    else values
                )
        return result

    serialized_train = serialize_split_frame(train_df)
    serialized_val = serialize_split_frame(val_df)
    serialized_test = serialize_split_frame(test_df)
    legacy_train = serialize_legacy_split_frame(train_df)
    legacy_val = serialize_legacy_split_frame(val_df)
    legacy_test = serialize_legacy_split_frame(test_df)

    interactions.to_csv(interactions_dir / "interactions.csv", index=False)
    serialized_train.to_csv(
        splits_dir / "train_sequences.csv",
        index=False,
    )
    serialized_val.to_csv(
        splits_dir / "val_sequences.csv",
        index=False,
    )
    serialized_test.to_csv(
        splits_dir / "test_sequences.csv",
        index=False,
    )
    item_metadata.to_csv(item_features_dir / "item_metadata.csv", index=False)
    user_id_map.to_csv(mappings_dir / "user_id_map.csv", index=False)
    item_id_map.to_csv(mappings_dir / "item_id_map.csv", index=False)

    with open(reports_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    with open(reports_dir / "preprocessing_report.json", "w") as f:
        json.dump(preprocessing_report, f, indent=2)

    # Transitional compatibility outputs for existing downstream readers.
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    legacy_train.to_csv(output_dir / "train.csv", index=False)
    legacy_val.rename(
        columns={"item_sequence": "train_seq", "target_item": "target"}
    )[["user_idx", "train_seq", "target"]].to_csv(output_dir / "val.csv", index=False)
    legacy_test.rename(
        columns={"item_sequence": "train_seq", "target_item": "target"}
    )[["user_idx", "train_seq", "target"]].to_csv(output_dir / "test.csv", index=False)
    item_metadata.to_csv(output_dir / "item_meta.csv", index=False)
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)


def build_dataset_stats(
    n_users: int,
    n_items: int,
    n_interactions: int,
    min_seq_len: int,
    min_item_freq: int,
) -> dict:
    total_slots = n_users * n_items
    sparsity = 0.0 if total_slots == 0 else round(1 - n_interactions / total_slots, 6)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "min_seq_len": int(min_seq_len),
        "min_item_freq": int(min_item_freq),
        "sparsity": sparsity,
        "pad_token": 0,
        "max_item_idx": n_items,
    }


def _build_item_metadata(items: pd.DataFrame, item_id_map: pd.DataFrame) -> pd.DataFrame:
    metadata = items.copy()
    metadata = metadata[metadata["item_id"].isin(item_id_map["item_id"])].copy()
    metadata["item_idx"] = metadata["item_id"].map(
        dict(zip(item_id_map["item_id"], item_id_map["item_idx"]))
    )

    def map_string_column(target: str, candidates: list[str]) -> None:
        source = next((column for column in candidates if column in metadata.columns), None)
        if source is None:
            metadata[target] = ""
        else:
            metadata[target] = metadata[source]
        metadata[target] = metadata[target].fillna("").astype(str)

    map_string_column("title", ["title", "name", "Name"])
    map_string_column("description", ["description", "Description"])
    map_string_column("language", ["language", "Language"])
    map_string_column("difficulty", ["difficulty", "Difficulty"])
    map_string_column("theme", ["theme", "Theme"])
    map_string_column("software", ["software", "Software"])
    map_string_column("job", ["job", "Job"])
    map_string_column("type", ["type", "Type"])

    duration_source = next(
        (column for column in ["duration", "Duration"] if column in metadata.columns),
        None,
    )
    if duration_source is None:
        metadata["duration"] = ""
    else:
        metadata["duration"] = metadata[duration_source]
    metadata["duration"] = metadata["duration"].fillna("")
    metadata["text"] = metadata["title"] + " [SEP] " + metadata["description"]

    ordered_columns = [
        "item_idx",
        "item_id",
        "title",
        "description",
        "text",
        "language",
        "difficulty",
        "theme",
        "software",
        "job",
        "type",
        "duration",
    ]
    return metadata[ordered_columns].sort_values("item_idx", kind="stable").reset_index(
        drop=True
    )


def build_preprocessing_report(
    *,
    orphan_implicit_count: int,
    orphan_explicit_count: int,
    engagement_pairs: int,
    min_user_interactions: int,
    min_item_interactions: int,
    repeat_events_removed: int,
    eligible_user_count: int,
    filtered_item_count: int,
    n_interactions_total: int = 0,
    n_with_watch_signal: int = 0,
    n_observed_zero: int = 0,
    n_missing_watch: int = 0,
    n_positive_watch: int = 0,
    coverage_pct: float = 0.0,
) -> dict:
    return {
        "orphan_implicit_count": orphan_implicit_count,
        "orphan_explicit_count": orphan_explicit_count,
        "engagement_pairs": engagement_pairs,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "repeat_events_removed": repeat_events_removed,
        "eligible_user_count": eligible_user_count,
        "filtered_item_count": filtered_item_count,
        "engagement_coverage": {
            "total_interactions": n_interactions_total,
            "with_watch_signal": n_with_watch_signal,
            "observed_zero_watch": n_observed_zero,
            "missing_watch": n_missing_watch,
            "positive_watch": n_positive_watch,
            "coverage_pct": round(coverage_pct, 2),
        },
    }


def count_orphan_implicit_rows(
    implicit: pd.DataFrame,
    catalog_item_ids: set[str],
) -> int:
    return int((~implicit["item_id"].isin(catalog_item_ids)).sum())


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    min_seq_len = MIN_BENCHMARK_SAFE_SEQUENCE_LEN
    min_user_interactions = MIN_USER_INTERACTIONS
    min_item_interactions = MIN_ITEM_INTERACTIONS

    implicit = pd.read_csv(
        raw_dir / "implicit_ratings.csv",
        parse_dates=["created_at"],
        dtype={"user_id": str, "item_id": str},
    )
    items = pd.read_csv(raw_dir / "items.csv", dtype={"item_id": str})
    explicit = load_explicit_ratings(raw_dir / "explicit_ratings.csv")
    engagement_lookup = build_engagement_lookup(explicit)
    engagement_pairs = len(engagement_lookup)
    catalog_item_ids = set(items["item_id"].astype(str))

    implicit_sorted = implicit.sort_values("created_at", kind="stable")
    implicit_dedup = implicit_sorted.drop_duplicates(
        subset=["user_id", "item_id"],
        keep="first",
    ).copy()
    repeat_events_removed = len(implicit) - len(implicit_dedup)
    orphan_implicit_count = count_orphan_implicit_rows(implicit, catalog_item_ids)
    orphan_explicit_count = int((~explicit["item_id"].isin(catalog_item_ids)).sum())

    interactions = implicit_dedup[implicit_dedup["item_id"].isin(catalog_item_ids)].copy()
    interactions = attach_engagement_score(interactions, engagement_lookup)

    # Coverage stats
    n_interactions_total = len(interactions)
    has_watch = interactions["has_watch_signal"].astype(bool)
    n_with_watch_signal = int(has_watch.sum())
    n_observed_zero = int((has_watch & (interactions["engagement_score"] == 0.0)).sum())
    n_missing_watch = int((~has_watch).sum())
    n_positive_watch = int((has_watch & (interactions["engagement_score"] > 0.0)).sum())
    coverage_pct = n_with_watch_signal / n_interactions_total * 100 if n_interactions_total > 0 else 0.0

    interactions = apply_iterative_k_core_filter(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    all_users = sorted(interactions["user_id"].unique())
    all_items = sorted(interactions["item_id"].unique())
    user_id_map = pd.DataFrame(
        {"user_id": all_users, "user_idx": range(1, len(all_users) + 1)}
    )
    item_id_map = pd.DataFrame(
        {"item_id": all_items, "item_idx": range(1, len(all_items) + 1)}
    )
    user_idx_lookup = dict(zip(user_id_map["user_id"], user_id_map["user_idx"]))
    item_idx_lookup = dict(zip(item_id_map["item_id"], item_id_map["item_idx"]))

    interactions["user_idx"] = interactions["user_id"].map(user_idx_lookup)
    interactions["item_idx"] = interactions["item_id"].map(item_idx_lookup)
    interactions = interactions.dropna(subset=["user_idx", "item_idx"]).copy()
    interactions["user_idx"] = interactions["user_idx"].astype(int)
    interactions["item_idx"] = interactions["item_idx"].astype(int)
    interactions = interactions.sort_values(
        ["user_idx", "created_at", "item_idx"],
        kind="stable",
    ).reset_index(drop=True)

    user_sequences = build_user_sequences(interactions)
    eligible_sequences = filter_benchmark_safe_user_sequences(user_sequences, min_seq_len)
    eligible_users = set(eligible_sequences["user_id"])
    interactions = interactions[interactions["user_id"].isin(eligible_users)].copy()

    all_users = sorted(interactions["user_id"].unique())
    all_items = sorted(interactions["item_id"].unique())
    user_id_map = pd.DataFrame(
        {"user_id": all_users, "user_idx": range(1, len(all_users) + 1)}
    )
    item_id_map = pd.DataFrame(
        {"item_id": all_items, "item_idx": range(1, len(all_items) + 1)}
    )
    user_idx_lookup = dict(zip(user_id_map["user_id"], user_id_map["user_idx"]))
    item_idx_lookup = dict(zip(item_id_map["item_id"], item_id_map["item_idx"]))
    interactions["user_idx"] = interactions["user_id"].map(user_idx_lookup).astype(int)
    interactions["item_idx"] = interactions["item_id"].map(item_idx_lookup).astype(int)
    interactions = interactions.sort_values(
        ["user_idx", "created_at", "item_idx"],
        kind="stable",
    ).reset_index(drop=True)
    interactions["sequence_order"] = interactions.groupby("user_idx").cumcount()
    interactions = interactions[
        [
            "user_idx",
            "item_idx",
            "user_id",
            "item_id",
            "created_at",
            "engagement_score",
            "has_watch_signal",
            "sequence_order",
        ]
    ]

    mapped_user_sequences = build_user_sequences(interactions)
    train_df, val_df, test_df = split_leave_one_out(mapped_user_sequences)

    item_metadata = _build_item_metadata(items, item_id_map)
    dataset_stats = build_dataset_stats(
        n_users=len(user_id_map),
        n_items=len(item_id_map),
        n_interactions=len(interactions),
        min_seq_len=min_seq_len,
        min_item_freq=min_item_interactions,
    )
    preprocessing_report = build_preprocessing_report(
        orphan_implicit_count=orphan_implicit_count,
        orphan_explicit_count=orphan_explicit_count,
        engagement_pairs=engagement_pairs,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        repeat_events_removed=repeat_events_removed,
        eligible_user_count=len(user_id_map),
        filtered_item_count=len(item_id_map),
        n_interactions_total=n_interactions_total,
        n_with_watch_signal=n_with_watch_signal,
        n_observed_zero=n_observed_zero,
        n_missing_watch=n_missing_watch,
        n_positive_watch=n_positive_watch,
        coverage_pct=coverage_pct,
    )

    save_processed_outputs(
        output_dir=processed_dir,
        interactions=interactions,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        item_metadata=item_metadata,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        dataset_stats=dataset_stats,
        preprocessing_report=preprocessing_report,
    )


if __name__ == "__main__":
    main()
