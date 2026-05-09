import ast
import json
from pathlib import Path

import pandas as pd


def _parse_item_sequence(raw_value: object, row_number: int) -> list[int]:
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return []

    try:
        parsed = ast.literal_eval(raw_value) if isinstance(raw_value, str) else raw_value
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid item_sequence at row {row_number}: {raw_value!r}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Invalid item_sequence at row {row_number}: {raw_value!r}")

    return parsed


def run_audit(data_dir: Path) -> dict:
    stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_sequences = [
        _parse_item_sequence(raw_value, row_number)
        for row_number, raw_value in enumerate(train_df["item_sequence"], start=1)
    ]
    seq_lens = pd.Series([len(sequence) for sequence in train_sequences], dtype=float)
    train_items = {item for sequence in train_sequences for item in sequence}
    eval_targets = list(val_df["target"].tolist()) + list(test_df["target"].tolist())
    cold_target_count = sum(1 for target in eval_targets if target not in train_items)
    cold_target_ratio = 0.0 if not eval_targets else cold_target_count / len(eval_targets)

    return {
        "n_users": int(stats["n_users"]),
        "n_items": int(stats["n_items"]),
        "train_seq_len_p50": float(seq_lens.median()) if len(seq_lens) else 0.0,
        "train_seq_len_p90": float(seq_lens.quantile(0.9)) if len(seq_lens) else 0.0,
        "cold_target_ratio": float(cold_target_ratio),
    }
