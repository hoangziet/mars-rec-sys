import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq2_report, rq3_report


def test_rq2_tune_forces_batch_size_128():
    from pipeline.training_grid import enforce_final_grid

    train_kwargs = {"batch_size": 256, "epochs": 30, "lr": 1e-3}
    out = enforce_final_grid(train_kwargs)
    assert out["batch_size"] == 128
    assert out["epochs"] == 30
    assert out["lr"] == 1e-3
    assert train_kwargs["batch_size"] == 256


def test_rq2_grid_rejects_duplicate_alpha_seed():
    selected = [
        {"alpha": 0.0, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 0.0, "seed": 42, "val_ndcg_at_10": 0.2},
    ]
    with pytest.raises(RuntimeError, match="Duplicate"):
        rq2_report._validate_rq2_grid(selected)


def test_rq2_grid_rejects_inconsistent_seed_sets():
    selected = [
        {"alpha": 0.0, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 0.5, "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    with pytest.raises(RuntimeError, match="same seed set"):
        rq2_report._validate_rq2_grid(selected)


def test_rq2_grid_accepts_custom_grid():
    """A custom sweep (e.g. only alpha=[0.5, 1.0] with 2 seeds) is valid
    as long as every (alpha, seed) appears exactly once and seed sets match."""
    selected = [
        {"alpha": 0.5, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 0.5, "seed": 123, "val_ndcg_at_10": 0.1},
        {"alpha": 1.0, "seed": 42, "val_ndcg_at_10": 0.1},
        {"alpha": 1.0, "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    rq2_report._validate_rq2_grid(selected)


def test_rq3_grid_rejects_duplicate_variant_seed():
    selected = [
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.2},
    ]
    with pytest.raises(RuntimeError, match="Duplicate"):
        rq3_report._validate_rq3_grid(selected)


def test_rq3_grid_rejects_inconsistent_seed_sets():
    selected = [
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    with pytest.raises(RuntimeError, match="same seed set"):
        rq3_report._validate_rq3_grid(selected)


def test_rq3_grid_accepts_custom_variants():
    """A partial variant sweep (e.g. only M1, M3) is valid."""
    selected = [
        {"variant": "M1", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 123, "val_ndcg_at_10": 0.1},
        {"variant": "M3", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M3", "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    rq3_report._validate_rq3_grid(selected)
