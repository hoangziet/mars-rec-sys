"""
scripts/rq3_build_vocab.py
==========================
Build metadata vocabulary and pre-built tensors for RQ3.

Input:  data/processed/item_features/item_metadata.csv
Output: data/processed/item_features/metadata_vocab.json
        data/processed/item_features/metadata_tensors.pt

Usage:
    uv run python scripts/rq3_build_vocab.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metadata_utils import MetadataVocab, build_metadata_tensors, load_item_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build metadata vocabulary and tensors.")
    parser.add_argument("--input", default="data/processed/item_features/item_metadata.csv")
    parser.add_argument("--vocab-output", default="data/processed/item_features/metadata_vocab.json")
    parser.add_argument("--tensors-output", default="data/processed/item_features/metadata_tensors.pt")
    parser.add_argument("--n-items", type=int, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    if args.n_items is None:
        stats_path = Path("data/processed/dataset_stats.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            n_items = stats["n_items"]
        else:
            raise ValueError("--n-items is required when dataset_stats.json is not found")
    else:
        n_items = args.n_items

    print(f"Loading metadata for {n_items} items...")
    df = load_item_metadata(args.input, n_items)

    print("Building vocabulary...")
    vocab = MetadataVocab.build(df)

    print("Building tensors...")
    tensors = build_metadata_tensors(vocab, df, n_items)

    vocab_path = Path(args.vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    print(f"Saved vocab: {vocab_path}")

    tensors_path = Path(args.tensors_output)
    tensors_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, tensors_path)
    print(f"Saved tensors: {tensors_path}")

    for field, tensor in tensors.items():
        print(f"  {field}: shape={tensor.shape}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
