import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq2_tune_forces_batch_size_128():
    from pipeline.training_grid import enforce_final_grid

    train_kwargs = {"batch_size": 256, "epochs": 30, "lr": 1e-3}
    out = enforce_final_grid(train_kwargs)
    assert out["batch_size"] == 128
    assert out["epochs"] == 30
    assert out["lr"] == 1e-3
    assert train_kwargs["batch_size"] == 256


def test_rq2_write_outputs_handles_multiple_variants():
    from scripts.rq2_report import write_outputs

    selected = [
        {"variant": "baseline", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29},
        {"variant": "wl", "seed": 42, "val_ndcg_at_10": 0.32, "test_NDCG_at_10": 0.31},
        {"variant": "we", "seed": 42, "val_ndcg_at_10": 0.31, "test_NDCG_at_10": 0.30},
        {"variant": "wlwe", "seed": 42, "val_ndcg_at_10": 0.33, "test_NDCG_at_10": 0.32},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        best = write_outputs(selected, {"best_alpha": 1.0}, out, "test")
        assert best == "wlwe"
        with open(out / "rq2_summary.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 4
            assert rows[0]["variant"] == "wlwe"
            assert float(rows[0]["val_ndcg_at_10_mean"]) == 0.33


def test_rq2_write_outputs_runs_csv_format():
    from scripts.rq2_report import write_outputs

    selected = [
        {"variant": "wl", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_outputs(selected, {"best_alpha": 0.5}, out, "test")
        with open(out / "rq2_runs.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["variant"] == "wl"
            assert rows[0]["seed"] == "42"


def test_rq3_grid_rejects_duplicate_variant_seed():
    from scripts import rq3_report

    selected = [
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.2},
    ]
    with pytest.raises(RuntimeError, match="Duplicate"):
        rq3_report._validate_rq3_grid(selected)


def test_rq3_grid_rejects_inconsistent_seed_sets():
    from scripts import rq3_report

    selected = [
        {"variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    with pytest.raises(RuntimeError, match="same seed set"):
        rq3_report._validate_rq3_grid(selected)


def test_rq3_grid_accepts_custom_variants():
    """A partial variant sweep (e.g. only M1, M3) is valid."""
    from scripts import rq3_report

    selected = [
        {"variant": "M1", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M1", "seed": 123, "val_ndcg_at_10": 0.1},
        {"variant": "M3", "seed": 42, "val_ndcg_at_10": 0.1},
        {"variant": "M3", "seed": 123, "val_ndcg_at_10": 0.1},
    ]
    rq3_report._validate_rq3_grid(selected)
