import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import train_all


def test_new_benchmark_manifest_starts_running():
    manifest = train_all.build_benchmark_manifest(
        benchmark_id="rq1-x",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        expected_models=["sasrec"],
        neural_seeds=[42],
        heuristic_seed=42,
    )

    assert manifest["status"] == "running"
    assert manifest["completed_run_keys"] == []
    assert manifest["failed_run_keys"] == []


def test_completed_manifest_rejects_new_runs():
    """A completed campaign must not accept new runs (immutable)."""
    manifest = train_all.build_benchmark_manifest(
        benchmark_id="rq1-x",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        expected_models=["sasrec"],
        neural_seeds=[42],
        heuristic_seed=42,
    )
    manifest["status"] = "completed"

    with pytest.raises(RuntimeError, match="already completed"):
        train_all._reject_if_completed(manifest)


def test_running_manifest_allows_resume(tmp_path):
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest = train_all.build_benchmark_manifest(
        benchmark_id="rq1-x",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        expected_models=["sasrec"],
        neural_seeds=[42],
        heuristic_seed=42,
    )
    manifest_path.write_text(json.dumps(manifest))

    result = train_all._validate_or_prepare_manifest_for_resume(
        manifest_path=manifest_path,
        benchmark_id="rq1-x",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
    )

    assert result["status"] == "running"


def test_new_benchmark_creates_manifest(tmp_path):
    """When no manifest exists, a new one should be created with status=running."""
    manifest_path = tmp_path / "benchmark_manifest.json"

    result = train_all._validate_or_prepare_manifest_for_resume(
        manifest_path=manifest_path,
        benchmark_id="rq1-x",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        expected_models=["sasrec"],
        neural_seeds=[42],
        heuristic_seed=42,
    )

    assert result["status"] == "running"
    assert manifest_path.exists()


def test_build_run_key_format():
    assert train_all._build_run_key("gsasrec", 42, "base") == "gsasrec:42:base"


def test_finished_run_key_is_skipped():
    """Completed run keys should be detected and skipped."""
    finished = {"sasrec:42:base", "gsasrec:123:base"}

    assert train_all._build_run_key("sasrec", 42, "base") in finished
    assert train_all._build_run_key("gsasrec", 99, "base") not in finished


# ---------------------------------------------------------------------------
# RQ1 report tests
# ---------------------------------------------------------------------------

from scripts import rq1_report


def test_rq1_report_rejects_running_manifest(tmp_path):
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "running",
        "benchmark_id": "rq1-x",
        "protocol_version": "rq1-v1",
        "preprocessing_version": "mars-preprocess-v1",
        "expected_models": ["sasrec"],
        "neural_seeds": [42],
        "heuristic_seed": 42,
    }))

    with pytest.raises(RuntimeError, match="not completed"):
        rq1_report._validate_manifest_completed(manifest_path)


def test_rq1_report_accepts_completed_manifest(tmp_path):
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "completed",
        "benchmark_id": "rq1-x",
        "protocol_version": "rq1-v1",
        "preprocessing_version": "mars-preprocess-v1",
        "expected_models": ["sasrec"],
        "neural_seeds": [42],
        "heuristic_seed": 42,
    }))

    manifest = rq1_report._validate_manifest_completed(manifest_path)
    assert manifest["status"] == "completed"
