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
