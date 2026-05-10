"""
pipeline/loaders.py
===================
Shared DataLoader for all 7 models.

Classes:
  - TrainSequenceDataset  : SASRec, gSASRec, GRU4Rec — sliding-window next-item prediction
  - MaskedSequenceDataset : BERT4Rec — masked item modelling
  - BPRDataset            : BPR-MF — (user, pos, neg) triplets
  - FullSortEvalDataset   : shared val / test — full-catalog ranking with history mask
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
        num_neg: int = 1,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pad_token = pad_token
        self.use_confidence = use_confidence
        self.num_neg = num_neg

        if use_confidence and (
            "confidence_sequence" not in df.columns and "confidence" not in df.columns
        ):
            raise ValueError(
                "TrainSequenceDataset: neither 'confidence_sequence' nor 'confidence' "
                "column found. Set use_confidence=False or provide the column."
            )

        all_items: set[int] = set()
        raw: list[tuple[list[int], list[float]]] = []

        has_conf_seq = "confidence_sequence" in df.columns

        for row in df.itertuples(index=False):
            seq = parse_seq(row.item_sequence)
            all_items.update(seq)
            if not use_confidence:
                confs = [1.0] * len(seq)
            elif has_conf_seq:
                # Per-item confidence list: confidence_sequence[i] = watch% for seq[i]
                confs = [float(c) for c in parse_seq(row.confidence_sequence)]
                if len(confs) != len(seq):
                    confs = confs[:len(seq)] + [1.0] * max(0, len(seq) - len(confs))
            else:
                # Legacy: single scalar confidence applied to all positions
                confs = [float(row.confidence)] * len(seq)
            raw.append((seq, confs))

        self._n_items = n_items if n_items is not None else max(all_items)
        self._all_items = np.arange(1, self._n_items + 1, dtype=np.int64)

        # Expand into per-position samples + pre-sample negatives
        self.input_seqs: list[list[int]] = []
        self.pos_targets: list[int] = []
        self.neg_targets: list[list[int]] = []  # (num_neg,) per sample
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
                negs = np.random.choice(
                    neg_pool,
                    size=num_neg,
                    replace=len(neg_pool) < num_neg,
                ).tolist()

                self.input_seqs.append(pad_sequence(inp, max_len, pad_token))
                self.pos_targets.append(tgt)
                self.neg_targets.append(negs)
                self.confidences.append(confs[i])

    def __len__(self) -> int:
        return len(self.pos_targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inp = self.input_seqs[idx]
        mask = [int(t != self.pad_token) for t in inp]
        negs = self.neg_targets[idx]
        # Return neg_items as (num_neg,) — squeeze to scalar when num_neg==1
        # for backward compatibility with SASRec/GRU4Rec which expect (B,) neg.
        neg_tensor = torch.tensor(negs, dtype=torch.long)
        if self.num_neg == 1:
            neg_tensor = neg_tensor.squeeze(0)
        return {
            "input_seq": torch.tensor(inp, dtype=torch.long),
            "pos_items": torch.tensor(self.pos_targets[idx], dtype=torch.long),
            "neg_items": neg_tensor,
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
# 4. FullSortEvalDataset  — shared val / test
# ---------------------------------------------------------------------------


class FullSortEvalDataset(Dataset):
    """Full-catalog evaluation dataset.

    For each user, provides a boolean ``history_mask`` of shape ``(n_items+1,)``
    where ``True`` marks items the model should not be credited for recommending
    (padding token + items in the user's training history).  The target item is
    explicitly kept unmasked so the model can rank it against the full catalog.

    Evaluation protocol: rank target item among all ``n_items`` items after
    applying the history mask (setting masked logits to ``-inf``).
    """

    def __init__(
        self,
        csv_path: str,
        n_items: int,
        max_len: int = 50,
        pad_token: int = 0,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pad_token = pad_token
        self.n_items = n_items

        self.users = df["user_idx"].tolist()
        self.seqs = [parse_seq(s) for s in df["train_seq"]]
        self.targets = df["target"].tolist()

        self._padded_seqs = [
            pad_sequence(seq, max_len, pad_token) for seq in self.seqs
        ]

        # Pre-build history masks: (n_users, n_items+1) bool tensor.
        # True = masked (seen or padding), False = rankable.
        n = len(self.users)
        history_masks = torch.zeros(n, n_items + 1, dtype=torch.bool)
        history_masks[:, 0] = True  # padding token always masked
        for i, (seq, tgt) in enumerate(zip(self.seqs, self.targets)):
            for item in seq:
                history_masks[i, item] = True
            history_masks[i, tgt] = False  # target must remain rankable
        self._history_masks = history_masks

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
            "history_mask": self._history_masks[idx],  # (n_items+1,)
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
    num_neg: int = 1,
    **kwargs,
) -> DataLoader | None:
    n_items = stats["n_items"]

    if model_type in ("sasrec", "gsasrec", "gru4rec"):
        dataset = TrainSequenceDataset(
            train_csv,
            max_len=max_len,
            pad_token=0,
            use_confidence=(use_confidence or model_type == "gsasrec"),
            n_items=n_items,
            num_neg=num_neg,
        )
    elif model_type == "bert4rec":
        mask_prob = kwargs.pop("mask_prob", 0.15)
        dataset = MaskedSequenceDataset(
            train_csv, n_items=n_items, max_len=max_len, is_train=True,
            mask_prob=mask_prob,
        )
    elif model_type == "bprmf":
        dataset = BPRDataset(train_csv, n_items=n_items)
    else:
        return None

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


class _ValLossDataset(Dataset):
    """Flat val-loss dataset built from pre-tensorised arrays.

    Defined at module level (not nested inside a function) so that it
    is picklable and DataLoader can use ``num_workers > 0``.
    """

    def __init__(
        self,
        input_seqs_t: torch.Tensor,
        pos_items_t: torch.Tensor,
        neg_items_t: torch.Tensor,
        user_t: torch.Tensor,
        conf_t: torch.Tensor,
        num_neg: int = 1,
    ) -> None:
        self._input_seqs = input_seqs_t
        self._pos_items  = pos_items_t
        self._neg_items  = neg_items_t
        self._user       = user_t
        self._conf       = conf_t
        self._num_neg    = num_neg

    def __len__(self) -> int:
        return len(self._pos_items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_seq":  self._input_seqs[idx],
            "pos_items":  self._pos_items[idx],
            "neg_items":  self._neg_items[idx],
            "confidence": self._conf[idx],
            "user":       self._user[idx],
            "pos_item":   self._pos_items[idx],
            "neg_item":   self._neg_items[idx] if self._num_neg == 1 else self._neg_items[idx, 0],
        }


def get_val_loss_loader(
    model_type: str,
    val_csv: str,
    stats: dict,
    batch_size: int = 256,
    max_len: int = 50,
    num_workers: int = 0,
    num_neg: int = 1,
    seed: int = 0,
) -> DataLoader | None:
    """Build a val-loss DataLoader in training format from val.csv.

    val.csv columns: user_idx, train_seq, target
    Produces one sample per user: pad(train_seq) → predict target.
    Supports sequential models and BPR-MF.  Returns None for bert4rec
    (MLM val-loss requires masking logic not present in val.csv).
    """
    if model_type == "bert4rec":
        return None

    n_items = stats["n_items"]
    df = pd.read_csv(val_csv)
    rng = np.random.default_rng(seed)

    input_seqs, pos_items_list, neg_items_list, user_list = [], [], [], []

    for _, row in df.iterrows():
        seq = parse_seq(row["train_seq"])
        target = int(row["target"])
        user_idx = int(row["user_idx"])
        seen = set(seq) | {target, 0}

        padded = pad_sequence(seq, max_len)
        input_seqs.append(padded)
        pos_items_list.append(target)
        user_list.append(user_idx)

        if num_neg == 1:
            neg = int(rng.integers(1, n_items + 1))
            while neg in seen:
                neg = int(rng.integers(1, n_items + 1))
            neg_items_list.append(neg)
        else:
            negs: list[int] = []
            while len(negs) < num_neg:
                n = int(rng.integers(1, n_items + 1))
                if n not in seen and n not in negs:
                    negs.append(n)
            neg_items_list.append(negs)

    input_seqs_t = torch.tensor(input_seqs, dtype=torch.long)
    pos_items_t  = torch.tensor(pos_items_list, dtype=torch.long)
    neg_items_t  = torch.tensor(neg_items_list, dtype=torch.long)
    user_t       = torch.tensor(user_list, dtype=torch.long)
    conf_t       = torch.ones(len(pos_items_list))

    dataset = _ValLossDataset(
        input_seqs_t, pos_items_t, neg_items_t, user_t, conf_t, num_neg=num_neg
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_eval_loader(
    eval_csv: str,
    stats: dict,
    batch_size: int = 64,
    max_len: int = 50,
    num_workers: int = 0,
) -> DataLoader:
    """Build full-sort evaluation DataLoader.

    Each batch contains ``history_mask`` (bool, shape ``n_items+1``) so the
    evaluation functions can mask seen items before ranking.
    """
    dataset = FullSortEvalDataset(
        eval_csv,
        n_items=stats["n_items"],
        max_len=max_len,
        pad_token=0,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
