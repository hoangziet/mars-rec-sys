"""Tests for strict provenance validation in RQ2/RQ3 reporters."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq2_report, rq3_report


# ---------- rq2_report._validate_provenance ----------

def _sel(provenance):
    return [{
        "alpha": 0.0,
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "provenance": provenance,
        "run_id": "rid",
        "run_name": "rn",
    }]


def test_rq2_provenance_accepts_complete():
    p = {
        "backbone": "gsasrec",
        "benchmark_id": "rq2-x",
        "preprocessing_version": "v1",
        "data_source": "/tmp",
        "git_commit": "abc",
    }
    out = rq2_report._validate_provenance(_sel(p))
    assert out == p


def test_rq2_report_writes_explicit_gsasrec_backbone():
    """RQ2 is gSASRec-only: provenance backbone must be gsasrec."""
    selected = [{
        "alpha": 0.5,
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "provenance": {
            "backbone": "gsasrec",
            "benchmark_id": "rq2-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
            "git_commit": "abc",
        },
        "run_id": "rid",
        "run_name": "rn",
    }]
    assert rq2_report._validate_provenance(selected)["backbone"] == "gsasrec"


def test_rq2_provenance_rejects_missing_field():
    p = {
        "backbone": "gsasrec",
        "benchmark_id": "rq2-x",
        "preprocessing_version": "v1",
        "data_source": "/tmp",
        # missing git_commit
    }
    with pytest.raises(RuntimeError, match="missing provenance field 'git_commit'"):
        rq2_report._validate_provenance(_sel(p))


def test_rq2_provenance_rejects_mismatch_across_runs():
    selected = [
        {
            "alpha": 0.0, "seed": 42, "val_ndcg_at_10": 0.1,
            "provenance": {"backbone": "gsasrec", "benchmark_id": "rq2-x",
                           "preprocessing_version": "v1", "data_source": "/a",
                           "git_commit": "abc"},
            "run_id": "r1", "run_name": "n1",
        },
        {
            "alpha": 0.5, "seed": 42, "val_ndcg_at_10": 0.2,
            "provenance": {"backbone": "gsasrec", "benchmark_id": "rq2-x",
                           "preprocessing_version": "v1", "data_source": "/b",
                           "git_commit": "abc"},
            "run_id": "r2", "run_name": "n2",
        },
    ]
    with pytest.raises(RuntimeError, match="Provenance mismatch for 'data_source'"):
        rq2_report._validate_provenance(selected)


def test_rq2_provenance_rejects_empty_string_field():
    p = {
        "backbone": "gsasrec",
        "benchmark_id": "rq2-x",
        "preprocessing_version": "",
        "data_source": "/tmp",
        "git_commit": "abc",
    }
    with pytest.raises(RuntimeError, match="missing provenance field 'preprocessing_version'"):
        rq2_report._validate_provenance(_sel(p))


# ---------- rq2_report._parse_seed ----------

class _StubRunInfo:
    def __init__(self, run_id, run_name):
        self.run_id = run_id
        self.run_name = run_name


class _StubRun:
    def __init__(self, params, run_id="rid", run_name="rn"):
        self.data = type("D", (), {"params": params})()
        self.info = _StubRunInfo(run_id, run_name)


def test_rq2_parse_seed_accepts_int_string():
    run = _StubRun({"seed": "42"})
    assert rq2_report._parse_seed(run, "rid") == 42


def test_rq2_parse_seed_accepts_int():
    run = _StubRun({"seed": 42})
    assert rq2_report._parse_seed(run, "rid") == 42


def test_rq2_parse_seed_rejects_missing_param():
    run = _StubRun({})
    with pytest.raises(RuntimeError, match="has no 'seed'"):
        rq2_report._parse_seed(run, "rid")


def test_rq2_parse_seed_rejects_garbage():
    run = _StubRun({"seed": "not-a-number"})
    with pytest.raises(RuntimeError, match="malformed seed"):
        rq2_report._parse_seed(run, "rid")


def test_rq2_parse_seed_includes_run_id_in_error():
    run = _StubRun({"seed": "bad"}, run_id="abc-123", run_name="my-run")
    with pytest.raises(RuntimeError, match="abc-123"):
        rq2_report._parse_seed(run, "abc-123")


# ---------- rq3_report: same contract ----------

def test_rq3_provenance_accepts_complete():
    p = {
        "backbone": "gsasrec",
        "benchmark_id": "rq3-x",
        "preprocessing_version": "v1",
        "data_source": "/tmp",
        "git_commit": "abc",
    }
    selected = [{
        "variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1,
        "provenance": p, "run_id": "rid", "run_name": "rn",
    }]
    out = rq3_report._validate_provenance(selected)
    assert out == p


def test_rq3_provenance_rejects_missing_field():
    p = {
        "backbone": "gsasrec",
        "benchmark_id": "rq3-x",
        # missing everything else
        "git_commit": "abc",
    }
    selected = [{
        "variant": "M0", "seed": 42, "val_ndcg_at_10": 0.1,
        "provenance": p, "run_id": "rid", "run_name": "rn",
    }]
    with pytest.raises(RuntimeError, match="missing provenance field"):
        rq3_report._validate_provenance(selected)


def test_rq3_parse_seed_rejects_missing_param():
    run = _StubRun({})
    with pytest.raises(RuntimeError, match="has no 'seed'"):
        rq3_report._parse_seed(run, "rid")