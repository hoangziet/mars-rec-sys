"""
scripts/rq3_build_vocab.py
==========================
Build metadata vocabulary for RQ3.

Input:  data/processed/item_features/item_metadata.csv
Output: data/processed/item_features/metadata_vocab.json

Usage:
    uv run python scripts/rq3_build_vocab.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import parse_seq
from pipeline.metadata_utils import MetadataVocab, load_item_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build metadata vocabulary.")
    parser.add_argument("--input", default="data/processed/item_features/item_metadata.csv")
    parser.add_argument("--vocab-output", default="data/processed/item_features/metadata_vocab.json")
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--train-sequences", default="data/processed/splits/train_sequences.csv")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _get_train_item_idx(train_csv: str) -> set[int]:
    """Collect the set of item_idx that appear in the training sequences."""
    df = pd.read_csv(train_csv)
    items: set[int] = set()
    for seq_str in df["item_sequence"]:
        items.update(parse_seq(seq_str))
    return items


def main() -> None:
    args = parse_args()

    if args.n_items is None:
        stats_path = Path("data/processed/reports/dataset_stats.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            n_items = stats["n_items"]
        else:
            raise ValueError("--n-items is required when dataset_stats.json is not found")
    else:
        n_items = args.n_items

    print(f"Loading metadata for {n_items} items...")
    df = load_item_metadata(args.input, n_items)

    train_csv = Path(args.train_sequences)
    if train_csv.exists():
        train_items = _get_train_item_idx(str(train_csv))
        train_item_sha256 = hashlib.sha256(
            str(sorted(train_items)).encode()
        ).hexdigest()
        fit_label = f"{len(train_items)} train items"
    else:
        train_items = None
        train_item_sha256 = None
        fit_label = "all items (no train sequences found)"

    print(f"Building vocabulary (fit on {fit_label})...")
    vocab = MetadataVocab.build(df, train_item_idx=train_items)

    vocab_path = Path(args.vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path, train_item_sha256=train_item_sha256)
    print(f"Saved vocab: {vocab_path}")
    print(f"  Categorical: {len(vocab.categorical)}")
    print(f"  Multilabel: {len(vocab.multilabel)}")


if __name__ == "__main__":
    main()
