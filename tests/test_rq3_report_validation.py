import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq3_report


def test_rq3_report_rejects_unknown_metadata_variant_tag():
    selected = [{
        "variant": "?",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "watch_mode": "none",
        "watch_alpha": "0.0",
        "provenance": {
            "backbone": "bert4rec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    with pytest.raises(RuntimeError, match="Invalid metadata_variant"):
        rq3_report._validate_variant_names(selected)


def test_rq3_report_rejects_watch_enabled_runs():
    selected = [{
        "variant": "M0",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "watch_mode": "loss",
        "watch_alpha": "0.5",
        "provenance": {
            "backbone": "bert4rec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    with pytest.raises(RuntimeError, match="expected 'none'"):
        rq3_report._validate_no_watch(selected)


def test_rq3_report_rejects_nonzero_watch_alpha():
    selected = [{
        "variant": "M0",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "watch_mode": "none",
        "watch_alpha": "0.5",
        "provenance": {
            "backbone": "bert4rec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    with pytest.raises(RuntimeError, match="expected 0.0"):
        rq3_report._validate_no_watch(selected)


def test_rq3_report_accepts_no_watch_runs():
    selected = [{
        "variant": "M0",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "watch_mode": "none",
        "watch_alpha": "0.0",
        "provenance": {
            "backbone": "bert4rec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    rq3_report._validate_no_watch(selected)


def test_rq3_report_rejects_missing_watch_tags():
    selected = [{
        "variant": "M0",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "provenance": {
            "backbone": "bert4rec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    with pytest.raises(RuntimeError, match="missing watch tags"):
        rq3_report._validate_no_watch(selected)
