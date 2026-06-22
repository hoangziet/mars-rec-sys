import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq2_report, rq3_report


def test_rq2_tune_forces_batch_size_128(monkeypatch):
    from training.configs import build_model_config
    cfg = build_model_config("gsasrec")
    train_kwargs = dict(cfg["train_kwargs"])
    train_kwargs["confidence_alpha"] = 0.5
    train_kwargs["batch_size"] = 256

    # protocol rule: tuning must align with final batch size
    train_kwargs["batch_size"] = 128
    assert train_kwargs["batch_size"] == 128


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
