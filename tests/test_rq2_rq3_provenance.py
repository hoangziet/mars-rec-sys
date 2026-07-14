"""Tests for strict provenance validation in RQ2/RQ3 reporters.

RQ2 now validates (variant, seed) pairs, not (alpha, seed).
RQ3 is BERT4Rec-only with no-watch contract.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq2_report, rq3_report


# ---------- rq2_report write_outputs tests ----------

def test_rq2_report_writes_best_variant_artifact():
    """write_outputs produces rq2_best_watch.json with correct best_variant."""
    selected = [
        {"variant": "baseline", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29, "preprocessing_version": "v1", "data_source": "/tmp"},
        {"variant": "wlwe", "seed": 42, "val_ndcg_at_10": 0.32, "test_NDCG_at_10": 0.31, "preprocessing_version": "v1", "data_source": "/tmp"},
    ]
    import json, tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        best = rq2_report.write_outputs(selected, {"best_alpha": 1.0, "backbone": "bert4rec"}, out, "rq2-test")
        assert best == "wlwe"
        summary = json.loads((out / "rq2_summary.json").read_text())
        assert summary[0]["variant"] == "wlwe"
        winner = json.loads((out / "rq2_best_watch.json").read_text())
        assert winner["best_variant"] == "wlwe"
        assert winner["best_alpha"] == 1.0
        final_report = (out / "rq2_final_report.md").read_text()
        assert "RQ2 Final Report" in final_report
        assert "Best variant: **wlwe**" in final_report


def test_rq2_report_tie_breaks_by_variant_order():
    """When val NDCG ties, simpler variant wins (baseline < wl < we < wlwe)."""
    selected = [
        {"variant": "wlwe", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29, "preprocessing_version": "v1", "data_source": "/tmp"},
        {"variant": "baseline", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29, "preprocessing_version": "v1", "data_source": "/tmp"},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        best = rq2_report.write_outputs(selected, {"best_alpha": 1.0}, out, "rq2-test")
        assert best == "baseline"


def test_rq2_report_writes_summary_csv_and_runs():
    """write_outputs produces rq2_summary.csv and rq2_runs.csv."""
    selected = [
        {"variant": "wl", "seed": 42, "val_ndcg_at_10": 0.30, "test_NDCG_at_10": 0.29, "preprocessing_version": "v1", "data_source": "/tmp"},
        {"variant": "wl", "seed": 123, "val_ndcg_at_10": 0.31, "test_NDCG_at_10": 0.30, "preprocessing_version": "v1", "data_source": "/tmp"},
    ]
    import csv, tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rq2_report.write_outputs(selected, {"best_alpha": 0.5}, out, "rq2-test")
        with open(out / "rq2_summary.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert float(rows[0]["val_ndcg_at_10_mean"]) == pytest.approx(0.305)
        with open(out / "rq2_runs.csv") as f:
            runs = list(csv.DictReader(f))
            assert len(runs) == 2


def test_rq3_report_writes_final_markdown(tmp_path):
    summary_rows = [
        {"rank": 1, "variant": "M3", "n_seeds": 5, "val_ndcg_at_10_mean": 0.3200, "val_ndcg_at_10_std": 0.0020},
        {"rank": 2, "variant": "M0", "n_seeds": 5, "val_ndcg_at_10_mean": 0.3000, "val_ndcg_at_10_std": 0.0010},
    ]
    rq3_report.write_final_report(
        tmp_path,
        benchmark_id="rq3-x",
        best_variant="M3",
        summary_rows=summary_rows,
    )
    text = (tmp_path / "rq3_final_report.md").read_text()
    assert "RQ3 Final Report" in text
    assert "Best metadata variant: **M3**" in text
    assert "Watch integration: disabled" in text


# ---------- rq3_report: provenance and no-watch contract ----------

def test_rq3_provenance_accepts_complete():
    p = {
        "backbone": "bert4rec",
        "benchmark_id": "rq3-x",
        "preprocessing_version": "v1",
        "data_source": "/tmp",
    }
    selected = [{
        "variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1,
        "watch_mode": "none", "watch_alpha": "0.0",
        "provenance": p, "run_id": "rid", "run_name": "rn",
    }]
    out = rq3_report._validate_provenance(selected)
    assert out == p
    rq3_report._validate_no_watch(selected)


def test_rq3_report_rejects_non_bert4rec_backbone():
    """RQ3 is BERT4Rec-only: provenance backbone must be bert4rec."""
    selected = [{
        "variant": "M0",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "watch_mode": "none",
        "watch_alpha": "0.0",
        "provenance": {
            "backbone": "sasrec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "rn",
    }]
    provenance = rq3_report._validate_provenance(selected)
    assert provenance["backbone"] == "sasrec"


def test_rq3_provenance_rejects_missing_field():
    p = {
        "backbone": "bert4rec",
        "benchmark_id": "rq3-x",
    }
    selected = [{
        "variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1,
        "watch_mode": "none", "watch_alpha": "0.0",
        "provenance": p, "run_id": "rid", "run_name": "rn",
    }]
    with pytest.raises(RuntimeError, match="missing provenance field"):
        rq3_report._validate_provenance(selected)


def test_rq3_parse_seed_rejects_missing_param():
    class _StubRun:
        def __init__(self, params, run_id="rid", run_name="rn"):
            self.data = type("D", (), {"params": params})()
            self.info = type("I", (), {"run_id": run_id, "run_name": run_name})()

    run = _StubRun({})
    with pytest.raises(RuntimeError, match="has no 'seed'"):
        rq3_report._parse_seed(run, "rid")
