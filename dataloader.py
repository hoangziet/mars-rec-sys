"""
dataloader.py
=============
Shared DataLoader for all 7 models.

Classes:
  - TrainSequenceDataset : SASRec, gSASRec, GRU4Rec — sliding-window next-item prediction
  - MaskedSequenceDataset: BERT4Rec — masked item modelling
  - BPRDataset           : BPR-MF — (user, pos, neg) triplets
  - EvalDataset          : shared val / test — 100-candidate ranking
"""

import ast
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_stats(stats_path: str = "data/processed/dataset_stats.json") -> dict:
    with open(stats_path) as f:
        return json.load(f)


def parse_seq(s) -> list[int]:
    """Parse a stringified list '"[1, 2, 3]"' or a Python list to list[int]."""
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)


def pad_sequence(seq: list[int], max_len: int, pad_token: int = 0) -> list[int]:
    """Truncate to max_len (keep the most recent items), then left-pad."""
    seq = seq[-max_len:]
    return [pad_token] * (max_len - len(seq)) + seq


# ---------------------------------------------------------------------------
# 1. TrainSequenceDataset  — SASRec / gSASRec / GRU4Rec
# ---------------------------------------------------------------------------


class TrainSequenceDataset(Dataset):
    """Sliding-window dataset for next-item prediction training.

    From a full sequence ``[a, b, c, d, e]`` the dataset creates:

    .. code-block:: text

        input_seq = [pad, pad, a, b, c]   (left-padded)
        pos_items = b  (the next item to predict at this position)
        neg_items = ?  (randomly sampled, pre-computed at init)

    This provides a training signal at every sequence position, not just
    the last one — matching the original SASRec paper training protocol.

    ``neg_items`` are pre-sampled at construction time so ``__getitem__``
    is O(1) and DataLoader workers don't block each other.
    """

    def __init__(
        self,
        csv_path: str,
        max_len: int = 50,
        pad_token: int = 0,
        use_confidence: bool = False,
        n_items: int | None = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pad_token = pad_token
        self.use_confidence = use_confidence

        if use_confidence and "confidence" not in df.columns:
            raise ValueError(
                "TrainSequenceDataset: 'confidence' column not found. "
                "Set use_confidence=False or provide the column."
            )

        all_items: set[int] = set()
        raw: list[tuple[list[int], list[float]]] = []

        for row in df.itertuples(index=False):
            seq = parse_seq(row.item_sequence)
            all_items.update(seq)
            conf_val = float(row.confidence) if use_confidence else 1.0
            confs = [conf_val] * len(seq)
            raw.append((seq, confs))

        self._n_items = n_items if n_items is not None else max(all_items)
        self._all_items = np.arange(1, self._n_items + 1, dtype=np.int64)

        # Expand into per-position samples + pre-sample negatives
        self.input_seqs: list[list[int]] = []
        self.pos_targets: list[int] = []
        self.neg_targets: list[int] = []
        self.confidences: list[float] = []

        for seq, confs in raw:
            if len(seq) < 2:
                continue
            seen = set(seq)
            neg_pool = np.setdiff1d(self._all_items, list(seen))
            if len(neg_pool) == 0:
                neg_pool = self._all_items

            for i in range(1, len(seq)):
                inp = seq[:i]
                tgt = seq[i]
                neg = int(np.random.choice(neg_pool))

                self.input_seqs.append(pad_sequence(inp, max_len, pad_token))
                self.pos_targets.append(tgt)
                self.neg_targets.append(neg)
                self.confidences.append(confs[i])

    def __len__(self) -> int:
        return len(self.pos_targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inp = self.input_seqs[idx]
        mask = [int(t != self.pad_token) for t in inp]
        return {
            "input_seq": torch.tensor(inp, dtype=torch.long),
            "pos_items": torch.tensor(self.pos_targets[idx], dtype=torch.long),
            "neg_items": torch.tensor(self.neg_targets[idx], dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "confidence": torch.tensor(self.confidences[idx], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# 2. MaskedSequenceDataset  — BERT4Rec
# ---------------------------------------------------------------------------


class MaskedSequenceDataset(Dataset):
    """Masked Item Modelling dataset for BERT4Rec.

    Special tokens
    --------------
    - ``pad_token``  = 0
    - ``mask_token`` = n_items + 1
    """

    def __init__(
        self,
        csv_path: str,
        n_items: int,
        max_len: int = 50,
        pad_token: int = 0,
        mask_prob: float = 0.15,
        is_train: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pad_token = pad_token
        self.mask_token = n_items + 1
        self.mask_prob = mask_prob
        self.is_train = is_train

        if is_train:
            self.seqs = [parse_seq(s) for s in df["item_sequence"]]
            self.targets = None
        else:
            self.seqs = [parse_seq(s) for s in df["train_seq"]]
            self.targets = df["target"].tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.seqs[idx]

        if self.is_train:
            masked_seq = seq.copy()
            labels = [0] * len(seq)
            for i, item in enumerate(seq):
                if np.random.random() < self.mask_prob:
                    masked_seq[i] = self.mask_token
                    labels[i] = item
            if seq and not any(labels):
                force = np.random.randint(len(seq))
                masked_seq[force] = self.mask_token
                labels[force] = seq[force]
            padded_seq = pad_sequence(masked_seq, self.max_len, self.pad_token)
            padded_lbl = pad_sequence(labels, self.max_len, 0)
        else:
            masked_seq = seq + [self.mask_token]
            labels = [0] * len(masked_seq)
            labels[-1] = self.targets[idx]
            padded_seq = pad_sequence(masked_seq, self.max_len, self.pad_token)
            padded_lbl = pad_sequence(labels, self.max_len, 0)

        return {
            "input_seq": torch.tensor(padded_seq, dtype=torch.long),
            "labels": torch.tensor(padded_lbl, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3. BPRDataset  — BPR-MF
# ---------------------------------------------------------------------------


class BPRDataset(Dataset):
    """(user, pos_item, neg_item) triplets for BPR-MF training.

    Includes a ``max_retry`` guard to prevent infinite loops when a user
    has seen nearly every item in the catalogue.
    """

    _MAX_NEG_RETRY: int = 500

    def __init__(self, csv_path: str, n_items: int) -> None:
        df = pd.read_csv(csv_path)
        self.n_items = n_items
        self.user_items: dict[int, set[int]] = {}
        self.samples: list[tuple[int, int]] = []

        for row in df.itertuples(index=False):
            uid = int(row.user_idx)
            seq = parse_seq(row.item_sequence)
            self.user_items.setdefault(uid, set()).update(seq)
            for pos in seq:
                self.samples.append((uid, pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        uid, pos = self.samples[idx]
        seen = self.user_items[uid]
        neg = 1
        for _ in range(self._MAX_NEG_RETRY):
            neg = np.random.randint(1, self.n_items + 1)
            if neg not in seen:
                break
        return {
            "user": torch.tensor(uid, dtype=torch.long),
            "pos_item": torch.tensor(pos, dtype=torch.long),
            "neg_item": torch.tensor(neg, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 4. EvalDataset  — shared val / test
# ---------------------------------------------------------------------------


class EvalDataset(Dataset):
    """Evaluation dataset: 1 ground-truth + ``num_neg`` negatives.

    Negatives are pre-sampled once at construction → deterministic metrics.
    """

    def __init__(
        self,
        csv_path: str,
        n_items: int,
        max_len: int = 50,
        pad_token: int = 0,
        num_neg: int = 99,
        neg_mode: str = "random",
        item_popularity: np.ndarray | None = None,
    ) -> None:
        if neg_mode not in {"random", "popularity", "mixed"}:
            raise ValueError(f"Unknown neg_mode '{neg_mode}'.")
        if neg_mode in {"popularity", "mixed"} and item_popularity is None:
            raise ValueError(
                "item_popularity required for popularity / mixed neg_mode."
            )

        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pad_token = pad_token
        self.n_items = n_items

        self.users = df["user_idx"].tolist()
        self.seqs = [parse_seq(s) for s in df["train_seq"]]
        self.targets = df["target"].tolist()

        user_history: dict[int, set[int]] = {}
        for uid, seq, tgt in zip(self.users, self.seqs, self.targets):
            user_history.setdefault(uid, set()).update(seq)
            user_history[uid].add(tgt)

        all_items = np.arange(1, n_items + 1, dtype=np.int64)
        self._candidates: list[list[int]] = []

        for uid, tgt in zip(self.users, self.targets):
            seen = user_history[uid]
            available = np.setdiff1d(all_items, list(seen))

            if len(available) < num_neg:
                negs = np.random.choice(available, size=num_neg, replace=True).tolist()
            elif neg_mode == "random":
                negs = np.random.choice(available, size=num_neg, replace=False).tolist()
            elif neg_mode == "popularity":
                weights = np.array(
                    [
                        item_popularity[i] if i < len(item_popularity) else 0.0
                        for i in available
                    ],
                    dtype=float,
                )
                weights = weights / weights.sum() if weights.sum() > 0 else None
                negs = np.random.choice(
                    available, size=num_neg, replace=False, p=weights
                ).tolist()
            else:  # mixed
                half = num_neg // 2
                pop_w = np.array(
                    [
                        item_popularity[i] if i < len(item_popularity) else 0.0
                        for i in available
                    ],
                    dtype=float,
                )
                pop_w = pop_w / pop_w.sum() if pop_w.sum() > 0 else None
                pop_negs = set(
                    np.random.choice(
                        available, size=half, replace=False, p=pop_w
                    ).tolist()
                )
                rand_pool = np.setdiff1d(available, list(pop_negs))
                rand_negs = np.random.choice(
                    rand_pool, size=num_neg - half, replace=False
                ).tolist()
                negs = list(pop_negs) + rand_negs

            self._candidates.append([tgt] + negs)

        self._padded_seqs = [pad_sequence(seq, max_len, pad_token) for seq in self.seqs]

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        padded = self._padded_seqs[idx]
        mask = [int(t != self.pad_token) for t in padded]
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "input_seq": torch.tensor(padded, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": torch.tensor(self.targets[idx], dtype=torch.long),
            "candidates": torch.tensor(self._candidates[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def get_train_loader(
    model_type: str,
    train_csv: str,
    stats: dict,
    batch_size: int = 256,
    max_len: int = 50,
    num_workers: int = 0,
    use_confidence: bool = False,
) -> DataLoader | None:
    n_items = stats["n_items"]

    if model_type in ("sasrec", "gsasrec", "gru4rec"):
        dataset = TrainSequenceDataset(
            train_csv,
            max_len=max_len,
            pad_token=0,
            use_confidence=(use_confidence or model_type == "gsasrec"),
            n_items=n_items,
        )
    elif model_type == "bert4rec":
        dataset = MaskedSequenceDataset(
            train_csv, n_items=n_items, max_len=max_len, is_train=True
        )
    elif model_type == "bprmf":
        dataset = BPRDataset(train_csv, n_items=n_items)
    else:
        return None

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def get_eval_loader(
    eval_csv: str,
    stats: dict,
    batch_size: int = 64,
    max_len: int = 50,
    num_workers: int = 0,
    num_neg: int = 99,
    neg_mode: str = "random",
    item_popularity: np.ndarray | None = None,
) -> DataLoader:
    dataset = EvalDataset(
        eval_csv,
        n_items=stats["n_items"],
        max_len=max_len,
        pad_token=0,
        num_neg=num_neg,
        neg_mode=neg_mode,
        item_popularity=item_popularity,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
