import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import get_val_loss_loader


def test_val_loss_loader_rejects_when_no_negative_items_available(tmp_path):
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(
        [{"user_idx": 1, "item_sequence": "1 2 3", "target_item": 4}]
    ).to_csv(val_csv, index=False)

    with pytest.raises(RuntimeError, match="seen all items"):
        get_val_loss_loader(
            "sasrec",
            str(val_csv),
            {"n_items": 4},
            batch_size=1,
            max_len=5,
            num_neg=1,
            seed=42,
        )


def test_val_loss_loader_rejects_when_negative_pool_smaller_than_num_neg(tmp_path):
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(
        [{"user_idx": 1, "item_sequence": "1 2", "target_item": 3}]
    ).to_csv(val_csv, index=False)

    with pytest.raises(RuntimeError, match="Cannot sample 2 validation negatives"):
        get_val_loss_loader(
            "sasrec",
            str(val_csv),
            {"n_items": 4},
            batch_size=1,
            max_len=5,
            num_neg=2,
            seed=42,
        )
