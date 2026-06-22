import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.training_grid import enforce_final_grid
from scripts import rq2_report, rq3_report


def test_rq2_tune_forces_batch_size_128():
    """Final-grid batch_size is forced to 128, ignoring input value."""
    train_kwargs = {"batch_size": 256, "epochs": 30, "lr": 1e-3}
    out = enforce_final_grid(train_kwargs)
    assert out["batch_size"] == 128
    # Other fields preserved
    assert out["epochs"] == 30
    assert out["lr"] == 1e-3
    # Original is not mutated
    assert train_kwargs["batch_size"] == 256


def test_rq2_report_requires_exact_alpha_grid():
    selected = [
        {"alpha": 0.0, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 0.0, "seed": 123, "val_ndcg_at_10": 0.1},
        {"alpha": 0.5, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 0.5, "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    with pytest.raises(RuntimeError, match="Expected alphas"):
        rq2_report._validate_rq2_grid(selected)


def test_rq3_report_requires_exact_variant_grid():
    selected = [
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M0", "seed": 123, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    with pytest.raises(RuntimeError, match="Expected variants"):
        rq3_report._validate_rq3_grid(selected)
