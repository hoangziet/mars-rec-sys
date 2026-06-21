"""
pipeline/metadata_utils.py
===========================
Metadata vocabulary builder and tensor pre-builder for RQ3.

Classes:
    MetadataVocab — vocabulary for categorical/multi-label fields

Functions:
    load_item_metadata()      — load and normalize item metadata CSV
    build_metadata_tensors()  — pre-build (n_items+1, ...) tensors for fast lookup
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PAD = 0
MISSING = 1
UNK = 2
SPECIAL_TOKENS = {"PAD": PAD, "MISSING": MISSING, "UNK": UNK}


class MetadataVocab:
    """Vocabulary for item metadata fields.

    Supports:
    - Categorical (language, difficulty): str → int index
    - Multi-label (theme, software, job, type): str → list[int] indices
    - Numeric (duration): float → float (log1p + normalize)
    """

    def __init__(self, categorical: dict[str, dict[str, int]], multilabel: dict[str, dict[str, int]],
                 duration_mean: float, duration_std: float):
        self.categorical = categorical
        self.multilabel = multilabel
        self.duration_mean = duration_mean
        self.duration_std = duration_std

    @classmethod
    def build(cls, df: pd.DataFrame) -> MetadataVocab:
        categorical = {}
        multilabel = {}

        categorical_fields = ["language", "difficulty"]
        multilabel_fields = ["theme", "software", "job", "type"]

        for field in categorical_fields:
            vocab = {}
            idx = 3
            for val in df[field].dropna().unique():
                if val not in vocab:
                    vocab[val] = idx
                    idx += 1
            categorical[field] = vocab

        for field in multilabel_fields:
            vocab = {}
            idx = 3
            for val in df[field].dropna():
                for tag in str(val).split(";"):
                    tag = tag.strip()
                    if tag and tag not in vocab:
                        vocab[tag] = idx
                        idx += 1
            multilabel[field] = vocab

        durations = pd.to_numeric(df["duration"], errors="coerce").dropna()
        if len(durations) > 0:
            log_durations = np.log1p(durations.values)
            duration_mean = float(log_durations.mean())
            duration_std = float(log_durations.std()) if len(log_durations) > 1 else 1.0
            if duration_std == 0:
                duration_std = 1.0
        else:
            duration_mean = 0.0
            duration_std = 1.0

        return cls(categorical, multilabel, duration_mean, duration_std)

    def encode_categorical(self, field: str, value) -> int:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return MISSING
        vocab = self.categorical.get(field, {})
        return vocab.get(value, UNK)

    def encode_multilabel(self, field: str, value) -> list[int]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return [MISSING]
        vocab = self.multilabel.get(field, {})
        indices = []
        for tag in str(value).split(";"):
            tag = tag.strip()
            if tag:
                indices.append(vocab.get(tag, UNK))
        return indices if indices else [MISSING]

    def encode_duration(self, value) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        log_val = np.log1p(float(value))
        return (log_val - self.duration_mean) / self.duration_std

    def save(self, path: str | Path) -> None:
        data = {
            "categorical": self.categorical,
            "multilabel": self.multilabel,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> MetadataVocab:
        data = json.loads(Path(path).read_text())
        return cls(data["categorical"], data["multilabel"],
                   data["duration_mean"], data["duration_std"])


def load_item_metadata(csv_path: str | Path, n_items: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("item_idx").reset_index(drop=True)
    if len(df) < n_items:
        missing = pd.DataFrame({"item_idx": range(len(df) + 1, n_items + 1)})
        df = pd.concat([df, missing], ignore_index=True)
    return df.head(n_items)


def build_metadata_tensors(vocab: MetadataVocab, df: pd.DataFrame, n_items: int) -> dict[str, torch.Tensor]:
    tensors = {}

    for field in vocab.categorical:
        arr = [PAD]
        for i in range(n_items):
            val = df.iloc[i][field] if i < len(df) else None
            arr.append(vocab.encode_categorical(field, val))
        tensors[field] = torch.tensor(arr, dtype=torch.long)

    for field in vocab.multilabel:
        all_labels = []
        for i in range(n_items):
            val = df.iloc[i][field] if i < len(df) else None
            all_labels.append(vocab.encode_multilabel(field, val))
        max_labels = max(len(labels) for labels in all_labels) if all_labels else 1
        padded = [[PAD] * max_labels]
        for labels in all_labels:
            padded.append(labels + [PAD] * (max_labels - len(labels)))
        tensors[field] = torch.tensor(padded, dtype=torch.long)

    durations = [0.0]
    for i in range(n_items):
        val = df.iloc[i]["duration"] if i < len(df) else None
        durations.append(vocab.encode_duration(val))
    tensors["duration"] = torch.tensor(durations, dtype=torch.float32)

    return tensors
