"""
scripts/rq4_subgroup.py
=======================
RQ4: Subgroup analysis.

Groups:
    - Users with watch signal vs without
    - Short history vs long history (median split)
    - Head items vs tail items (top 20% vs bottom 80% by train interaction count)
    - Items with complete metadata vs missing metadata

Usage:
    uv run python scripts/rq4_subgroup.py --data-dir data/processed --output-dir experiments/rq4/rq4-ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import parse_seq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: subgroup analysis.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _derive_subgroups(data_dir: Path) -> dict:
    train_df = pd.read_csv(data_dir / "splits" / "train_sequences.csv")

    has_watch = []
    for _, row in train_df.iterrows():
        seq = parse_seq(row.get("engagement_sequence", ""))
        has_watch.append(any(e > 0 for e in seq) if seq else False)

    seq_lengths = train_df["sequence_length"]
    median_len = int(seq_lengths.median())

    item_counts: dict[int, int] = {}
    for _, row in train_df.iterrows():
        seq = parse_seq(row.item_sequence)
        for item in seq:
            item_counts[item] = item_counts.get(item, 0) + 1
    sorted_items = sorted(item_counts.items(), key=lambda x: -x[1])
    n_head = max(1, int(len(sorted_items) * 0.2))
    head_items = {item for item, _ in sorted_items[:n_head]}

    meta_path = data_dir / "item_features" / "item_metadata.csv"
    complete_meta_items = set()
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        for _, row in meta_df.iterrows():
            missing = 0
            for col in ["difficulty", "theme", "software", "job", "type"]:
                if pd.isna(row.get(col)):
                    missing += 1
            if missing == 0:
                complete_meta_items.add(int(row["item_idx"]))

    return {
        "has_watch_signal": has_watch,
        "median_seq_length": median_len,
        "head_items": head_items,
        "complete_meta_items": complete_meta_items,
        "item_counts": item_counts,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Deriving subgroups from train data...")
    subgroups = _derive_subgroups(data_dir)

    thresholds = {
        "median_seq_length": subgroups["median_seq_length"],
        "head_item_fraction": 0.2,
        "n_head_items": len(subgroups["head_items"]),
        "n_complete_meta_items": len(subgroups["complete_meta_items"]),
        "n_items_with_watch_signal": sum(1 for x in subgroups["has_watch_signal"] if x),
    }
    (output_dir / "rq4_subgroup_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"Thresholds: {thresholds}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
