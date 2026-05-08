"""
dataloader.py
=============
Shared DataLoader for all 7 models.
Each model will use its appropriate class.

Classes:
  - SequenceDataset      : GRU4Rec, SASRec, gSASRec
  - MaskedSequenceDataset: BERT4Rec
  - BPRDataset           : BPR-MF
  - MatrixDataset        : Item-based CF, Popularity
"""

import json
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Helpers 

def load_stats(stats_path="data/processed/dataset_stats.json"):
    with open(stats_path) as f:
        return json.load(f)


def parse_seq(s):
    """Parse string "[1, 2, 3]" or list to list of int."""
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)


def pad_sequence(seq, max_len, pad_token=0):
    """Pad left by pad_token, cut if > max_len."""
    seq = seq[-max_len:]                         
    pad_len = max_len - len(seq)
    return [pad_token] * pad_len + seq       



# 1. SequenceDataset — use for GRU4Rec, SASRec, gSASRec

class SequenceDataset(Dataset):
    """
    Each sample:
      - input_seq : tensor [max_len]        — sequence of items (left padded)
      - target    : tensor scalar           — item to predict
      - mask      : tensor [max_len] bool   — True = valid position, False = padding
      - confidence: tensor scalar (float)   — watch_percentage / 100, used only by gSASRec

    Used for: train / val / test
    """

    def __init__(self, csv_path, max_len=50, pad_token=0, use_confidence=False):
        """
        Args:
            csv_path       : path to train.csv / val.csv / test.csv
            max_len        : maximum sequence length (padding/truncation)
            pad_token      : token padding (default 0)
            use_confidence : True if used for gSASRec
        """
        df = pd.read_csv(csv_path)
        self.max_len        = max_len
        self.pad_token      = pad_token
        self.use_confidence = use_confidence

        self.users      = df["user_idx"].tolist()
        self.train_seqs = [parse_seq(s) for s in df["train_seq"]]
        self.targets    = df["target"].tolist()

        # confidence is only available in train.csv (gSASRec)
        if use_confidence and "confidence" in df.columns:
            self.confidences = df["confidence"].fillna(1.0).tolist()
        else:
            self.confidences = [1.0] * len(df)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        seq    = self.train_seqs[idx]
        padded = pad_sequence(seq, self.max_len, self.pad_token)
        mask   = [0 if t == self.pad_token else 1 for t in padded]

        return {
            "user"      : torch.tensor(self.users[idx],       dtype=torch.long),
            "input_seq" : torch.tensor(padded,                dtype=torch.long),
            "target"    : torch.tensor(self.targets[idx],     dtype=torch.long),
            "mask"      : torch.tensor(mask,                  dtype=torch.bool),
            "confidence": torch.tensor(self.confidences[idx], dtype=torch.float),
        }



# 2. TrainSequenceDataset — sliding window for training
#    (SASRec paper uses this approach instead of only predicting the last item)

class TrainSequenceDataset(Dataset):
    """
    Used for the TRAIN phase of SASRec / gSASRec / GRU4Rec.
    From a sequence [a,b,c,d,e], it creates pairs:
      input=[a,b,c,d]  target=[b,c,d,e]  (next-item prediction at each step)

    More efficient than SequenceDataset as it utilizes the full sequence,
    not just predicting the last item.
    """

    def __init__(self, csv_path, max_len=50, pad_token=0, use_confidence=False):
        df = pd.read_csv(csv_path)
        self.max_len        = max_len
        self.pad_token      = pad_token
        self.use_confidence = use_confidence

        self.samples = []
        for _, row in df.iterrows():
            seq  = parse_seq(row["item_sequence"])   # full train sequence
            conf = float(row.get("confidence", 1.0)) if use_confidence else 1.0

            if len(seq) < 2:
                # Sequence too short, cannot create input-target pairs
                # Still added to ensure every user has a sample
                self.samples.append({
                    "input": [pad_token] * max_len,
                    "target": seq[-1] if seq else pad_token,
                    "confidence": conf,
                })
                continue

            # Create sliding window
            for i in range(1, len(seq)):
                inp = seq[:i]                         # [a], [a,b], [a,b,c]...
                tgt = seq[i]                          # b, c, d...
                self.samples.append({
                    "input": pad_sequence(inp, max_len, pad_token),
                    "target": tgt,
                    "confidence": conf,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        mask = [0 if t == self.pad_token else 1 for t in s["input"]]
        return {
            "input_seq" : torch.tensor(s["input"],      dtype=torch.long),
            "target"    : torch.tensor(s["target"],     dtype=torch.long),
            "mask"      : torch.tensor(mask,            dtype=torch.bool),
            "confidence": torch.tensor(s["confidence"], dtype=torch.float),
        }



# 3. MaskedSequenceDataset — used for BERT4Rec

class MaskedSequenceDataset(Dataset):
    """
    BERT4Rec uses Masked Item Modeling (similar to BERT).
    Randomly masks 15% of items in the sequence -> model predicts masked items.

    Special tokens:
      pad_token  = 0
      mask_token = n_items + 1  (adds 1 special token apart from item IDs)
    """

    def __init__(self, csv_path, n_items, max_len=50,
                 pad_token=0, mask_prob=0.15, is_train=True):
        """
        Args:
            n_items   : total number of items (from dataset_stats.json)
            mask_prob : xác suất mask mỗi item (default 0.15)
            is_train  : True = random mask | False = mask last item (val/test)
        """
        df = pd.read_csv(csv_path)
        self.max_len    = max_len
        self.pad_token  = pad_token
        self.mask_token = n_items + 1     # special token
        self.mask_prob  = mask_prob
        self.is_train   = is_train

        if is_train:
            self.seqs    = [parse_seq(s) for s in df["item_sequence"]]
            self.targets = None
        else:
            self.seqs    = [parse_seq(s) for s in df["train_seq"]]
            self.targets = df["target"].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        if self.is_train:
            # Random mask 15% items
            masked_seq = seq.copy()
            labels     = [0] * len(seq)        # 0 = do not predict at this position
            for i, item in enumerate(seq):
                if np.random.random() < self.mask_prob:
                    masked_seq[i] = self.mask_token
                    labels[i]     = item        # ground truth
            padded_seq = pad_sequence(masked_seq, self.max_len, self.pad_token)
            padded_lbl = pad_sequence(labels,     self.max_len, 0)
        else:
            # Val/Test: mask the last item → predict
            masked_seq = seq + [self.mask_token]
            labels     = [0] * len(masked_seq)
            labels[-1] = self.targets[idx]
            padded_seq = pad_sequence(masked_seq, self.max_len, self.pad_token)
            padded_lbl = pad_sequence(labels,     self.max_len, 0)

        return {
            "input_seq": torch.tensor(padded_seq, dtype=torch.long),
            "labels"   : torch.tensor(padded_lbl, dtype=torch.long),
        }



# 4. BPRDataset — used for BPR-MF

class BPRDataset(Dataset):
    """
    BPR requires triplets: (user, pos_item, neg_item)
    neg_item is randomly sampled from items the user hasn't seen.

    Each epoch re-samples a different neg_item -> creating new data.
    """

    def __init__(self, csv_path, n_items):
        """
        Args:
            n_items: total items (for random negative sampling)
        """
        df = pd.read_csv(csv_path)
        self.n_items = n_items

        # Set of items each user has seen -> avoid sampling as negative
        self.user_items = {}
        self.samples    = []

        for _, row in df.iterrows():
            uid  = int(row["user_idx"])
            seq  = parse_seq(row["item_sequence"])
            self.user_items[uid] = set(seq)
            # Each item in the sequence is a positive sample
            for pos in seq:
                self.samples.append((uid, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, pos = self.samples[idx]

        # Sample random negative item (not in user's history)
        while True:
            neg = np.random.randint(1, self.n_items + 1)
            if neg not in self.user_items[uid]:
                break

        return {
            "user"    : torch.tensor(uid, dtype=torch.long),
            "pos_item": torch.tensor(pos, dtype=torch.long),
            "neg_item": torch.tensor(neg, dtype=torch.long),
        }



# 5. EvalDataset — shared for val/test across all models

class EvalDataset(Dataset):
    """
    Used for evaluation (val.csv / test.csv).
    Returns:
      - input_seq : history sequence (padded)
      - target    : item cần predict (ground truth)
      - neg_items : 99 items random (unseen) + 1 target = 100 candidates

    Standard evaluation: model rank 100 items, calculates HR@K and NDCG@K.
    """

    def __init__(self, csv_path, n_items, max_len=50, pad_token=0, num_neg=99):
        df = pd.read_csv(csv_path)
        self.max_len  = max_len
        self.pad_token = pad_token
        self.n_items  = n_items
        self.num_neg  = num_neg

        self.users   = df["user_idx"].tolist()
        self.seqs    = [parse_seq(s) for s in df["train_seq"]]
        self.targets = df["target"].tolist()

        # Pre-build user history for negative sampling
        self.user_history = {}
        for uid, seq, tgt in zip(self.users, self.seqs, self.targets):
            self.user_history[uid] = set(seq) | {tgt}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        seq = self.seqs[idx]
        tgt = self.targets[idx]

        padded = pad_sequence(seq, self.max_len, self.pad_token)
        mask   = [0 if t == self.pad_token else 1 for t in padded]

        # Sample 99 negative items
        neg_items = []
        seen      = self.user_history.get(uid, set())
        while len(neg_items) < self.num_neg:
            neg = np.random.randint(1, self.n_items + 1)
            if neg not in seen and neg not in neg_items:
                neg_items.append(neg)

        # candidates = [target] + [99 negs] - model must rank target at the top
        candidates = [tgt] + neg_items

        return {
            "user"      : torch.tensor(uid,        dtype=torch.long),
            "input_seq" : torch.tensor(padded,     dtype=torch.long),
            "mask"      : torch.tensor(mask,       dtype=torch.bool),
            "target"    : torch.tensor(tgt,        dtype=torch.long),
            "candidates": torch.tensor(candidates, dtype=torch.long),
        }



# 6. Factory functions — convenience

def get_train_loader(model_type, train_csv, stats, batch_size=256,
                     max_len=50, num_workers=0, use_confidence=False):
    """
    Returns the appropriate DataLoader for each model.

    Args:
        model_type: "sasrec" | "gsasrec" | "gru4rec" |
                    "bert4rec" | "bprmf" | "popularity" | "itemcf"
    """
    n_items = stats["n_items"]

    if model_type in ("sasrec", "gsasrec", "gru4rec"):
        dataset = TrainSequenceDataset(
            train_csv, max_len=max_len, pad_token=0,
            use_confidence=(model_type == "gsasrec")
        )
    elif model_type == "bert4rec":
        dataset = MaskedSequenceDataset(
            train_csv, n_items=n_items, max_len=max_len, is_train=True
        )
    elif model_type == "bprmf":
        dataset = BPRDataset(train_csv, n_items=n_items)
    else:
        # Popularity and ItemCF do not need DataLoader
        return None

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers)


def get_eval_loader(eval_csv, stats, batch_size=64,
                    max_len=50, num_workers=0, num_neg=99):
    """Shared for val and test across all models."""
    dataset = EvalDataset(
        eval_csv, n_items=stats["n_items"],
        max_len=max_len, pad_token=0, num_neg=num_neg
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)



# Quick test

if __name__ == "__main__":
    stats = load_stats("data/processed/dataset_stats.json")
    print("Dataset stats:", stats)

    # Test SequenceDataset
    print("-- Test SequenceDataset (val) --")
    val_loader = get_eval_loader("data/processed/val.csv", stats, batch_size=4)
    batch = next(iter(val_loader))
    print("input_seq shape:", batch["input_seq"].shape)   # [4, 50]
    print("target shape   :", batch["target"].shape)      # [4]
    print("candidates     :", batch["candidates"].shape)  # [4, 100]
    print("mask sample    :", batch["mask"][0])

    # Test BPRDataset
    print("-- Test BPRDataset (train) --")
    bpr_loader = get_train_loader("bprmf", "data/processed/train.csv",
                                   stats, batch_size=4)
    batch = next(iter(bpr_loader))
    print("user    :", batch["user"])
    print("pos_item:", batch["pos_item"])
    print("neg_item:", batch["neg_item"])

    # Test TrainSequenceDataset (SASRec)
    print("-- Test TrainSequenceDataset (SASRec) --")
    train_loader = get_train_loader("sasrec", "data/processed/train.csv",
                                     stats, batch_size=4)
    batch = next(iter(train_loader))
    print("input_seq shape:", batch["input_seq"].shape)
    print("target shape   :", batch["target"].shape)
    print("mask sample    :", batch["mask"][0])

    print("\nDataLoader test passed!")