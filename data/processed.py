import os 
import json
import pandas as pd
from pathlib import Path

# Config
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEQ_LEN   = 3    # Minimum number of unique items per user
MIN_ITEM_FREQ = 3    # Minimum number of interactions per item (after dedup)

# Step 1: Load raw data
print("=" * 50)
print("Step 1: Load raw data")
print("=" * 50)

implicit = pd.read_csv(RAW_DIR / "implicit_ratings.csv",
                       parse_dates=["created_at"])
explicit = pd.read_csv(RAW_DIR / "explicit_ratings.csv",
                       parse_dates=["created_at"])
items    = pd.read_csv(RAW_DIR / "items.csv")

print(f"Implicit: {len(implicit):,} rows | {implicit['user_id'].nunique():,} users | {implicit['item_id'].nunique():,} items")
print(f"Explicit: {len(explicit):,} rows | {explicit['user_id'].nunique():,} users")
print(f"Items:    {len(items):,} rows")

# Step 2: Dedup implicit - keep first occurrence
print("\nStep 2: Dedup implicit (keep first occurrence per user-item pair)")
implicit_sorted = implicit.sort_values("created_at")
implicit_dedup  = implicit_sorted.drop_duplicates(subset=["user_id", "item_id"], keep="first")
print(f"Before dedup: {len(implicit):,} rows -> After dedup: {len(implicit_dedup):,} rows")

# Step 3: Filter item < MIN_ITEM_FREQ interactions
print(f"\nStep 3: Filter item < {MIN_ITEM_FREQ} interactions")
item_freq   = implicit_dedup.groupby("item_id").size()
valid_items = item_freq[item_freq >= MIN_ITEM_FREQ].index
implicit_dedup = implicit_dedup[implicit_dedup["item_id"].isin(valid_items)]
print(f"Items left: {implicit_dedup['item_id'].nunique():,} / {item_freq.shape[0]:,}")
print(f"Interactions left: {len(implicit_dedup):,}")

# Step 4: Build user sequences (sort by time)
print("\nStep 4: Build user sequences")
user_sequences = (
    implicit_dedup
    .sort_values(["user_id", "created_at"])
    .groupby("user_id")["item_id"]
    .apply(list)
    .reset_index()
    .rename(columns={"item_id": "item_sequence"})
)
user_sequences["seq_len"] = user_sequences["item_sequence"].apply(len)

# Step 5: Filter user have seq_len < MIN_SEQ_LEN
print(f"Step 5: Filter user have seq_len < {MIN_SEQ_LEN}")
before = len(user_sequences)
user_sequences = user_sequences[user_sequences["seq_len"] >= MIN_SEQ_LEN].reset_index(drop=True)
after = len(user_sequences)
print(f"Users còn lại: {after:,} / {before:,} ({after/before:.1%})")
print(f"Seq len — min: {user_sequences.seq_len.min()}, max: {user_sequences.seq_len.max()}, mean: {user_sequences.seq_len.mean():.1f}")

# Step 6: Merge watch_percentage from explicit
print("\nStep 6: Merge watch_percentage from explicit data")
watch_df = (
    explicit
    .groupby(["user_id", "item_id"])["watch_percentage"]
    .mean()
    .reset_index()
    .rename(columns={"watch_percentage": "watch_pct"})
)
implicit_dedup = implicit_dedup.merge(watch_df, on=["user_id", "item_id"], how="left")
# NaN -> 1.0 (considered positive normal, no judge)
implicit_dedup["confidence"] = implicit_dedup["watch_pct"].fillna(1.0).clip(0, 100) / 100.0
merged_count = implicit_dedup["watch_pct"].notna().sum()
print(f"Interactions có watch_pct: {merged_count:,} / {len(implicit_dedup):,} ({merged_count/len(implicit_dedup):.1%})")
print("Confidence aggregation rule: explicit duplicates use mean watch_percentage; implicit duplicates keep earliest created_at row")

# Step 7: ID Remapping
print("\nStep 7: Remap IDs -> integer index (start from 1, 0 reserved for padding)")
all_users = sorted(user_sequences["user_id"].unique())
all_items = sorted(implicit_dedup[implicit_dedup["user_id"].isin(all_users)]["item_id"].unique())

user2idx = {u: i + 1 for i, u in enumerate(all_users)}
item2idx = {it: i + 1 for i, it in enumerate(all_items)}

print(f"Total users: {len(user2idx):,} | Total items: {len(item2idx):,}")

# Save mappings
with open(PROCESSED_DIR / "user2idx.json", "w") as f:
    json.dump({str(k): v for k, v in user2idx.items()}, f)
with open(PROCESSED_DIR / "item2idx.json", "w") as f:
    json.dump({str(k): v for k, v in item2idx.items()}, f)

# Remap sequences
def remap_seq(seq):
    return [item2idx[it] for it in seq if it in item2idx]

user_sequences["item_seq_idx"] = user_sequences["item_sequence"].apply(remap_seq)
user_sequences["user_idx"]     = user_sequences["user_id"].map(user2idx)
implicit_dedup["user_idx"] = implicit_dedup["user_id"].map(user2idx)
implicit_dedup["item_idx"] = implicit_dedup["item_id"].map(item2idx)

confidence_by_user_item = (
    implicit_dedup.dropna(subset=["user_idx", "item_idx"])
    .groupby(["user_idx", "item_idx"])["confidence"]
    .max()
)

# Step 8: Leave-one-out split
print("\nStep 8: Leave-one-out train/val/test split")
train_data, val_data, test_data = [], [], []

for _, row in user_sequences.iterrows():
    seq = row["item_seq_idx"]
    uid = row["user_idx"]
    n   = len(seq)
    train_history = seq[:-2]
    validation_target = seq[-2]
    test_target = seq[-1]

    if (uid, validation_target) not in confidence_by_user_item.index:
        raise KeyError(
            f"Missing confidence for train target: user_idx={uid}, item_idx={validation_target}"
        )

    train_data.append({
        "user_idx": uid,
        "item_sequence": train_history,
        "seq_len": n - 2,
        "target": validation_target,
        "confidence": confidence_by_user_item.loc[(uid, validation_target)],
    })
    val_data.append({"user_idx": uid, "train_seq": train_history, "target": validation_target})
    test_data.append({"user_idx": uid, "train_seq": seq[:-1], "target": test_target})

train_df = pd.DataFrame(train_data)
val_df   = pd.DataFrame(val_data)
test_df  = pd.DataFrame(test_data)

print(f"Train: {len(train_df):,} users | avg seq_len = {train_df.seq_len.mean():.1f}")
print(f"Val:   {len(val_df):,} users")
print(f"Test:  {len(test_df):,} users")

# Step 9: Save processed files
print("\nStep 9: Save files to data/processed/")

implicit_dedup_out = implicit_dedup.dropna(subset=["user_idx", "item_idx"])
implicit_dedup_out[["user_idx", "item_idx", "created_at", "confidence"]].to_csv(
    PROCESSED_DIR / "interactions.csv", index=False
)

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(  PROCESSED_DIR / "val.csv",   index=False)
test_df.to_csv( PROCESSED_DIR / "test.csv",  index=False)

items["item_idx"] = items["item_id"].map(item2idx)
items[items["item_idx"].notna()].to_csv(PROCESSED_DIR / "item_meta.csv", index=False)

stats = {
    "n_users":        len(user2idx),
    "n_items":        len(item2idx),
    "n_interactions": len(implicit_dedup_out),
    "min_seq_len":    int(MIN_SEQ_LEN),
    "min_item_freq":  int(MIN_ITEM_FREQ),
    "sparsity":       round(1 - len(implicit_dedup_out) / (len(user2idx) * len(item2idx)), 6),
    "pad_token":      0,
    "max_item_idx":   len(item2idx),
}
with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nDone! Files saved:")
for f in sorted(PROCESSED_DIR.iterdir()):
    print(f"  {f.name}")
print("\nDataset stats:")
for k, v in stats.items():
    print(f"  {k}: {v}")