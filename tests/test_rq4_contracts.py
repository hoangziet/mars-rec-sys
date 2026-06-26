import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_ablation, rq4_collect


def test_rq4_ablation_rejects_data_dir_mismatch_with_protocol(tmp_path):
    protocol = {
        "backbone": "bert4rec",
        "benchmark_id": "rq4-x",
        "rq2_best_alpha": 0.5,
        "rq2_best_variant": "wlwe",
        "best_metadata_variant": "M3",
        "neural_seeds": [42],
        "variants": ["V0"],
        "preprocessing_version": "v1",
        "data_source": str(tmp_path / "expected-data"),
    }

    with pytest.raises(RuntimeError, match="data_source mismatch"):
        rq4_ablation._validate_protocol_data_dir(protocol, tmp_path / "actual-data")


def test_rq4_collect_requires_protocol_for_comparison_manifest(tmp_path, monkeypatch):
    class _FakeExperiment:
        experiment_id = "exp-1"

    class _FakeClient:
        def get_experiment_by_name(self, _name):
            return _FakeExperiment()

        def search_runs(self, _ids):
            return []

    class _FakeMlflow:
        tracking = type("T", (), {"MlflowClient": lambda self=None: _FakeClient()})()

    monkeypatch.setattr(rq4_collect, "mlflow", _FakeMlflow())
    monkeypatch.setattr(rq4_collect, "configure_mlflow", lambda **_kw: None)

    saved = sys.argv
    sys.argv = [
        "rq4_collect.py",
        "--benchmark-id", "rq4-x",
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(RuntimeError, match="--protocol is required"):
            rq4_collect.main()
    finally:
        sys.argv = saved
