"""
data/preprocess.py
==================
Preprocess raw interaction data into train/val/test splits.

Usage:
    uv run python data/preprocess.py

Outputs (written to data/processed/):
    train.csv, val.csv, test.csv
    interactions.csv, item_meta.csv
    user2idx.json, item2idx.json, dataset_stats.json
"""

import csv
import json
from pathlib import Path

import pandas as pd


CONFIDENCE_ALPHA = 0.5
BASELINE_CONFIDENCE = 1.0
LEAVE_ONE_OUT_HOLDOUTS = 2
MIN_TRAIN_HISTORY_LEN_FOR_NEURAL_MODELS = 2
MIN_BENCHMARK_SAFE_SEQUENCE_LEN = (
    MIN_TRAIN_HISTORY_LEN_FOR_NEURAL_MODELS + LEAVE_ONE_OUT_HOLDOUTS
)


def load_explicit_ratings(path: Path) -> pd.DataFrame:
    """Load explicit ratings and clip outlier watch/rating values.

    The raw CSV can contain malformed trailing quotes in numeric fields, so
    parse with QUOTE_NONE and normalize column/value quotes explicitly.
    """
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
        explicit[col] = pd.to_numeric(explicit[col], errors="coerce")

    explicit["created_at"] = pd.to_datetime(explicit["created_at"], errors="coerce")
    explicit = explicit.dropna(
        subset=["user_id", "item_id", "watch_percentage"]
    ).copy()

    explicit["user_id"] = explicit["user_id"].astype(int).astype(str)
    explicit["item_id"] = explicit["item_id"].astype(int).astype(str)
    explicit["watch_percentage"] = explicit["watch_percentage"].clip(upper=100)
    explicit["rating"] = explicit["rating"].clip(upper=10)

    return explicit.reset_index(drop=True)


def build_confidence_lookup(
    explicit: pd.DataFrame,
    alpha: float = CONFIDENCE_ALPHA,
) -> pd.DataFrame:
    explicit = explicit.copy()
    explicit["confidence"] = 1.0 + alpha * (
        explicit["watch_percentage"] / 100.0
    )
    return (
        explicit.groupby(["user_id", "item_id"], as_index=False)["confidence"]
        .max()
    )


def attach_confidence(
    implicit: pd.DataFrame,
    confidence_lookup: pd.DataFrame,
    baseline: float = BASELINE_CONFIDENCE,
) -> pd.DataFrame:
    result = implicit.merge(
        confidence_lookup,
        on=["user_id", "item_id"],
        how="left",
    )
    result["confidence"] = result["confidence"].fillna(baseline)
    return result


def build_user_sequences(implicit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, group in implicit.sort_values(["user_id", "created_at"]).groupby("user_id"):
        rows.append(
            {
                "user_id": user_id,
                "item_sequence": group["item_id"].tolist(),
                "confidence_sequence": group["confidence"].astype(float).tolist(),
                "seq_len": len(group),
            }
        )
    return pd.DataFrame(rows)


def _get_sequence_length(row: pd.Series) -> int:
    if "item_seq_idx" in row and isinstance(row["item_seq_idx"], list):
        return len(row["item_seq_idx"])
    if "item_sequence" in row and isinstance(row["item_sequence"], list):
        return len(row["item_sequence"])
    return int(row["seq_len"])


def _get_confidence_sequence_length(row: pd.Series) -> int | None:
    if "confidence_seq" in row and isinstance(row["confidence_seq"], list):
        return len(row["confidence_seq"])
    if "confidence_sequence" in row and isinstance(row["confidence_sequence"], list):
        return len(row["confidence_sequence"])
    return None


def _validate_row_sequence_alignment(row: pd.Series) -> None:
    seq_len = _get_sequence_length(row)
    confidence_len = _get_confidence_sequence_length(row)
    if confidence_len is not None and seq_len != confidence_len:
        user_id = row.get("user_id", "<unknown>")
        raise ValueError(
            "confidence sequence length mismatch "
            f"for user_id={user_id}: items={seq_len}, confidence={confidence_len}"
        )


def filter_benchmark_safe_user_sequences(
    user_sequences: pd.DataFrame,
    min_seq_len: int = MIN_BENCHMARK_SAFE_SEQUENCE_LEN,
) -> pd.DataFrame:
    if user_sequences.empty:
        return user_sequences.reset_index(drop=True)

    sequence_lengths = user_sequences.apply(_get_sequence_length, axis=1)
    confidence_lengths = user_sequences.apply(_get_confidence_sequence_length, axis=1)
    mismatched = confidence_lengths.notna() & (sequence_lengths != confidence_lengths)
    if mismatched.any():
        first_bad_row = user_sequences.loc[mismatched].iloc[0]
        _validate_row_sequence_alignment(first_bad_row)

    filtered = user_sequences[sequence_lengths >= min_seq_len].copy()
    filtered["seq_len"] = sequence_lengths[sequence_lengths >= min_seq_len].astype(int)
    return filtered.reset_index(drop=True)


def format_user_retention(after: int, before: int) -> str:
    if before == 0:
        return f"Users remaining: {after:,} / {before:,} (n/a)"
    return f"Users remaining: {after:,} / {before:,} ({after / before:.1%})"


def summarize_sequence_lengths(user_sequences: pd.DataFrame) -> str:
    if user_sequences.empty:
        return "Seq len - min: n/a, max: n/a, mean: n/a"
    return (
        "Seq len - min: "
        f"{user_sequences.seq_len.min()}, max: {user_sequences.seq_len.max()}, "
        f"mean: {user_sequences.seq_len.mean():.1f}"
    )


def split_leave_one_out(
    user_sequences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_columns = [
        "user_idx",
        "item_sequence",
        "seq_len",
        "target",
        "confidence",
        "confidence_sequence",
    ]
    eval_columns = ["user_idx", "train_seq", "target"]

    train_data, val_data, test_data = [], [], []

    for _, row in user_sequences.iterrows():
        _validate_row_sequence_alignment(row)
        seq = row["item_seq_idx"]
        uid = row["user_idx"]
        confidence_seq = row["confidence_seq"]
        n = len(seq)
        if n < MIN_BENCHMARK_SAFE_SEQUENCE_LEN:
            user_id = row.get("user_id", "<unknown>")
            raise ValueError(
                "sequence shorter than benchmark-safe minimum "
                f"for user_id={user_id}: length={n}, "
                f"minimum={MIN_BENCHMARK_SAFE_SEQUENCE_LEN}"
            )
        train_history = seq[:-2]
        train_confidence_history = confidence_seq[:-2]
        validation_target = seq[-2]
        validation_confidence = confidence_seq[-2]
        test_target = seq[-1]

        train_data.append(
            {
                "user_idx": uid,
                "item_sequence": train_history,
                "seq_len": n - 2,
                "target": validation_target,
                "confidence": validation_confidence,
                "confidence_sequence": train_confidence_history,
            }
        )
        val_data.append(
            {
                "user_idx": uid,
                "train_seq": train_history,
                "target": validation_target,
            }
        )
        test_data.append(
            {"user_idx": uid, "train_seq": seq[:-1], "target": test_target}
        )

    return (
        pd.DataFrame(train_data, columns=train_columns),
        pd.DataFrame(val_data, columns=eval_columns),
        pd.DataFrame(test_data, columns=eval_columns),
    )


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


def main():
    RAW_DIR       = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    MIN_SEQ_LEN   = MIN_BENCHMARK_SAFE_SEQUENCE_LEN
    MIN_ITEM_FREQ = 3   # Minimum interactions per item (after dedup)

    # Step 1: Load raw data
    implicit = pd.read_csv(
        RAW_DIR / "implicit_ratings.csv",
        parse_dates=["created_at"],
        dtype={"user_id": str, "item_id": str},
    )
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
    explicit = load_explicit_ratings(RAW_DIR / "explicit_ratings.csv")
    confidence_lookup = build_confidence_lookup(explicit)

    print(f"Implicit: {len(implicit):,} rows | {implicit['user_id'].nunique():,} users | {implicit['item_id'].nunique():,} items")
    print(f"Explicit: {len(explicit):,} rows | {explicit['user_id'].nunique():,} users | {explicit['item_id'].nunique():,} items")
    print(f"Confidence pairs: {len(confidence_lookup):,}")
    print(f"Items:    {len(items):,} rows")

    # Step 2: Dedup implicit — keep first occurrence per user-item pair
    print("\nStep 2: Dedup implicit (keep first occurrence per user-item pair)")
    implicit_sorted = implicit.sort_values("created_at")
    implicit_dedup  = implicit_sorted.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    print(f"Before dedup: {len(implicit):,} rows -> After dedup: {len(implicit_dedup):,} rows")

    # Step 3: Filter items with fewer than MIN_ITEM_FREQ interactions
    print(f"\nStep 3: Filter items with < {MIN_ITEM_FREQ} interactions")
    item_freq   = implicit_dedup.groupby("item_id").size()
    valid_items = item_freq[item_freq >= MIN_ITEM_FREQ].index
    implicit_dedup = implicit_dedup[implicit_dedup["item_id"].isin(valid_items)]
    implicit_dedup = attach_confidence(implicit_dedup, confidence_lookup)
    print(f"Items left: {implicit_dedup['item_id'].nunique():,} / {item_freq.shape[0]:,}")
    print(f"Interactions left: {len(implicit_dedup):,}")

    # Step 4: Build user sequences (sorted by time)
    print("\nStep 4: Build user sequences")
    user_sequences = build_user_sequences(implicit_dedup)

    # Step 5: Enforce benchmark-safe user histories so leave-one-out keeps
    # train history length >= 2 for all neural models.
    print(
        "Step 5: Filter users with "
        f"seq_len < {MIN_SEQ_LEN} to ensure train history length >= "
        f"{MIN_TRAIN_HISTORY_LEN_FOR_NEURAL_MODELS} for all neural models"
    )
    before         = len(user_sequences)
    user_sequences = filter_benchmark_safe_user_sequences(user_sequences, MIN_SEQ_LEN)
    after          = len(user_sequences)
    print(format_user_retention(after, before))
    print(summarize_sequence_lengths(user_sequences))

    # Step 6: ID remapping (1-indexed; 0 reserved for padding)
    print("\nStep 6: Remap IDs -> integer index")
    all_users = sorted(user_sequences["user_id"].unique())
    all_items = sorted(implicit_dedup[implicit_dedup["user_id"].isin(all_users)]["item_id"].unique())
    user2idx  = {u: i + 1 for i, u in enumerate(all_users)}
    item2idx  = {it: i + 1 for i, it in enumerate(all_items)}
    print(f"Total users: {len(user2idx):,} | Total items: {len(item2idx):,}")

    with open(PROCESSED_DIR / "user2idx.json", "w") as f:
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    with open(PROCESSED_DIR / "item2idx.json", "w") as f:
        json.dump({str(k): v for k, v in item2idx.items()}, f)

    def remap_seq(seq):
        return [item2idx[it] for it in seq if it in item2idx]

    def remap_confidence_seq(row):
        return [
            confidence
            for item, confidence in zip(
                row["item_sequence"], row["confidence_sequence"]
            )
            if item in item2idx
        ]

    user_sequences["item_seq_idx"] = user_sequences["item_sequence"].apply(remap_seq)
    user_sequences["confidence_seq"] = user_sequences.apply(
        remap_confidence_seq,
        axis=1,
    )
    user_sequences["user_idx"]     = user_sequences["user_id"].map(user2idx)
    implicit_dedup["user_idx"]     = implicit_dedup["user_id"].map(user2idx)
    implicit_dedup["item_idx"]     = implicit_dedup["item_id"].map(item2idx)

    # Step 7: Leave-one-out train/val/test split
    print("\nStep 7: Leave-one-out train/val/test split")
    train_df, val_df, test_df = split_leave_one_out(user_sequences)

    avg_train_seq_len = (
        f"{train_df.seq_len.mean():.1f}" if not train_df.empty else "n/a"
    )
    print(f"Train: {len(train_df):,} users | avg seq_len = {avg_train_seq_len}")
    print(f"Val:   {len(val_df):,} users")
    print(f"Test:  {len(test_df):,} users")

    # Step 8: Save
    print("\nStep 8: Save files to data/processed/")
    implicit_out = implicit_dedup.dropna(subset=["user_idx", "item_idx"])
    implicit_out[["user_idx", "item_idx", "created_at", "confidence"]].to_csv(
        PROCESSED_DIR / "interactions.csv",
        index=False,
    )
    explicit.to_csv(PROCESSED_DIR / "explicit.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv",     index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv",   index=False)

    items["item_idx"] = items["item_id"].map(item2idx)
    items[items["item_idx"].notna()].to_csv(PROCESSED_DIR / "item_meta.csv", index=False)

    stats = build_dataset_stats(
        n_users=len(user2idx),
        n_items=len(item2idx),
        n_interactions=len(implicit_out),
        min_seq_len=MIN_SEQ_LEN,
        min_item_freq=MIN_ITEM_FREQ,
    )
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone! Files saved:")
    for p in sorted(PROCESSED_DIR.iterdir()):
        print(f"  {p.name}")
    print("\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
