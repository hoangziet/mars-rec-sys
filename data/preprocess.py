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

import json
from pathlib import Path

import pandas as pd


def main():
    RAW_DIR       = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    MIN_SEQ_LEN   = 3   # Minimum unique items per user
    MIN_ITEM_FREQ = 3   # Minimum interactions per item (after dedup)

    # Step 1: Load raw data
    implicit = pd.read_csv(
        RAW_DIR / "implicit_ratings.csv",
        parse_dates=["created_at"],
        dtype={"user_id": str, "item_id": str},
    )
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})

    print(f"Implicit: {len(implicit):,} rows | {implicit['user_id'].nunique():,} users | {implicit['item_id'].nunique():,} items")
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
    print(f"Items left: {implicit_dedup['item_id'].nunique():,} / {item_freq.shape[0]:,}")
    print(f"Interactions left: {len(implicit_dedup):,}")

    # Step 4: Build user sequences (sorted by time)
    print("\nStep 4: Build user sequences")
    user_sequences = (
        implicit_dedup.sort_values(["user_id", "created_at"])
        .groupby("user_id")["item_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"item_id": "item_sequence"})
    )
    user_sequences["seq_len"] = user_sequences["item_sequence"].apply(len)

    # Step 5: Filter users with fewer than MIN_SEQ_LEN items
    print(f"Step 5: Filter users with seq_len < {MIN_SEQ_LEN}")
    before         = len(user_sequences)
    user_sequences = user_sequences[user_sequences["seq_len"] >= MIN_SEQ_LEN].reset_index(drop=True)
    after          = len(user_sequences)
    print(f"Users remaining: {after:,} / {before:,} ({after / before:.1%})")
    print(f"Seq len — min: {user_sequences.seq_len.min()}, max: {user_sequences.seq_len.max()}, mean: {user_sequences.seq_len.mean():.1f}")

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

    user_sequences["item_seq_idx"] = user_sequences["item_sequence"].apply(remap_seq)
    user_sequences["user_idx"]     = user_sequences["user_id"].map(user2idx)
    implicit_dedup["user_idx"]     = implicit_dedup["user_id"].map(user2idx)
    implicit_dedup["item_idx"]     = implicit_dedup["item_id"].map(item2idx)

    # Step 7: Leave-one-out train/val/test split
    print("\nStep 7: Leave-one-out train/val/test split")
    train_data, val_data, test_data = [], [], []

    for _, row in user_sequences.iterrows():
        seq  = row["item_seq_idx"]
        uid  = row["user_idx"]
        n    = len(seq)
        train_history      = seq[:-2]
        validation_target  = seq[-2]
        test_target        = seq[-1]

        train_data.append({
            "user_idx":      uid,
            "item_sequence": train_history,
            "seq_len":       n - 2,
            "target":        validation_target,
        })
        val_data.append({"user_idx": uid, "train_seq": train_history, "target": validation_target})
        test_data.append({"user_idx": uid, "train_seq": seq[:-1],     "target": test_target})

    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)

    print(f"Train: {len(train_df):,} users | avg seq_len = {train_df.seq_len.mean():.1f}")
    print(f"Val:   {len(val_df):,} users")
    print(f"Test:  {len(test_df):,} users")

    # Step 8: Save
    print("\nStep 8: Save files to data/processed/")
    implicit_out = implicit_dedup.dropna(subset=["user_idx", "item_idx"])
    implicit_out[["user_idx", "item_idx", "created_at"]].to_csv(PROCESSED_DIR / "interactions.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv",     index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv",   index=False)

    items["item_idx"] = items["item_id"].map(item2idx)
    items[items["item_idx"].notna()].to_csv(PROCESSED_DIR / "item_meta.csv", index=False)

    stats = {
        "n_users":        len(user2idx),
        "n_items":        len(item2idx),
        "n_interactions": len(implicit_out),
        "min_seq_len":    int(MIN_SEQ_LEN),
        "min_item_freq":  int(MIN_ITEM_FREQ),
        "sparsity":       round(1 - len(implicit_out) / (len(user2idx) * len(item2idx)), 6),
        "pad_token":      0,
        "max_item_idx":   len(item2idx),
    }
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
