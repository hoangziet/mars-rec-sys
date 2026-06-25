"""Smoke test: rq1_report writes a machine-readable winner artifact."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq1_report_writes_winner_artifact(tmp_path, monkeypatch):
    """Drive rq1_report.main with a fake MLflow that returns one run per
    neural model and check it writes a valid winner artifact."""
    from scripts import rq1_report

    runs = []
    for model in ("gsasrec", "sasrec"):
        runs.append(_StubRun(
            run_id=f"rid-{model}",
            run_name=f"{model}-base-s42",
            status="FINISHED",
            tags={
                "benchmark_id": "test-rq1",
                "reportable": "true",
                "variant": "base",
                "protocol_version": "rq1-v1",
                "preprocessing_version": "v1",
                "data_source": str(tmp_path),
                "model": model,
            },
            params={"seed": "42"},
            metrics={
                "best_val_ndcg_at_10": 0.6 if model == "gsasrec" else 0.5,
                "best_epoch": 1.0,
                "test_NDCG_at_10": 0.55 if model == "gsasrec" else 0.45,
                "test_Recall_at_10": 0.6 if model == "gsasrec" else 0.5,
                "test_NDCG_at_20": 0.65 if model == "gsasrec" else 0.55,
                "test_Recall_at_20": 0.7 if model == "gsasrec" else 0.6,
            },
        ))

    monkeypatch.setattr(rq1_report.mlflow, "tracking", _StubMlflowTracking(runs))
    monkeypatch.setattr(rq1_report, "configure_mlflow", lambda **kw: None)

    # Build a manifest at the expected path.
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True)
    manifest = {
        "benchmark_id": "test-rq1",
        "protocol_version": "rq1-v1",
        "preprocessing_version": "v1",
        "expected_models": ["gsasrec", "sasrec"],
        "neural_seeds": [42],
        "heuristic_seed": 42,
    }
    (output_dir.parent / "benchmark_manifest.json").write_text(json.dumps(manifest))

    saved = sys.argv
    sys.argv = [
        "rq1_report.py",
        "--benchmark-id", "test-rq1",
        "--manifest", str(output_dir.parent / "benchmark_manifest.json"),
        "--output-dir", str(output_dir),
    ]
    try:
        rq1_report.main()
    finally:
        sys.argv = saved

    winner_path = output_dir / "rq1_winner.json"
    assert winner_path.exists(), "rq1_report did not write rq1_winner.json"
    winner = json.loads(winner_path.read_text())

    assert winner["benchmark_id"] == "test-rq1"
    assert winner["winner_model"] == "gsasrec", winner
    assert winner["selection_metric"] == "best_val_ndcg_at_10"
    assert winner["selection_split"] == "val"
    assert winner["seed_set"] == [42]
    assert winner["preprocessing_version"] == "v1"
    assert str(tmp_path) in winner["data_source"]


def test_rq1_report_warns_on_heuristic_winner(tmp_path, monkeypatch, capsys):
    from scripts import rq1_report

    runs = [_StubRun(
        run_id="rid-pop",
        run_name="popularity-base-s42",
        status="FINISHED",
        tags={
            "benchmark_id": "test-rq1",
            "reportable": "true",
            "variant": "base",
            "protocol_version": "rq1-v1",
            "preprocessing_version": "v1",
            "data_source": str(tmp_path),
            "model": "popularity",
        },
        params={"seed": "42"},
        metrics={
            "best_val_ndcg_at_10": 0.3,
            "best_epoch": 0.0,
            "test_NDCG_at_10": 0.3,
            "test_Recall_at_10": 0.3,
            "test_NDCG_at_20": 0.3,
            "test_Recall_at_20": 0.3,
        },
    )]
    monkeypatch.setattr(rq1_report.mlflow, "tracking", _StubMlflowTracking(runs))
    monkeypatch.setattr(rq1_report, "configure_mlflow", lambda **kw: None)

    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True)
    manifest = {
        "benchmark_id": "test-rq1",
        "protocol_version": "rq1-v1",
        "preprocessing_version": "v1",
        "expected_models": ["popularity"],
        "neural_seeds": [42],
        "heuristic_seed": 42,
    }
    (output_dir.parent / "benchmark_manifest.json").write_text(json.dumps(manifest))

    saved = sys.argv
    sys.argv = [
        "rq1_report.py",
        "--benchmark-id", "test-rq1",
        "--manifest", str(output_dir.parent / "benchmark_manifest.json"),
        "--output-dir", str(output_dir),
    ]
    try:
        rq1_report.main()
    finally:
        sys.argv = saved

    out = capsys.readouterr().out
    assert "heuristic" in out.lower(), out


def test_rq1_report_rejects_mixed_data_sources(tmp_path, monkeypatch):
    from scripts import rq1_report

    runs = [
        _StubRun(
            run_id="rid-gsasrec",
            run_name="gsasrec-base-s42",
            status="FINISHED",
            tags={
                "benchmark_id": "test-rq1",
                "reportable": "true",
                "variant": "base",
                "protocol_version": "rq1-v1",
                "preprocessing_version": "v1",
                "data_source": str(tmp_path / "source-a"),
                "model": "gsasrec",
            },
            params={"seed": "42"},
            metrics={
                "best_val_ndcg_at_10": 0.6,
                "best_epoch": 1.0,
                "test_NDCG_at_10": 0.55,
                "test_Recall_at_10": 0.6,
                "test_NDCG_at_20": 0.65,
                "test_Recall_at_20": 0.7,
            },
        ),
        _StubRun(
            run_id="rid-sasrec",
            run_name="sasrec-base-s42",
            status="FINISHED",
            tags={
                "benchmark_id": "test-rq1",
                "reportable": "true",
                "variant": "base",
                "protocol_version": "rq1-v1",
                "preprocessing_version": "v1",
                "data_source": str(tmp_path / "source-b"),
                "model": "sasrec",
            },
            params={"seed": "42"},
            metrics={
                "best_val_ndcg_at_10": 0.5,
                "best_epoch": 1.0,
                "test_NDCG_at_10": 0.45,
                "test_Recall_at_10": 0.5,
                "test_NDCG_at_20": 0.55,
                "test_Recall_at_20": 0.6,
            },
        ),
    ]

    monkeypatch.setattr(rq1_report.mlflow, "tracking", _StubMlflowTracking(runs))
    monkeypatch.setattr(rq1_report, "configure_mlflow", lambda **kw: None)

    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True)
    manifest = {
        "benchmark_id": "test-rq1",
        "protocol_version": "rq1-v1",
        "preprocessing_version": "v1",
        "expected_models": ["gsasrec", "sasrec"],
        "neural_seeds": [42],
        "heuristic_seed": 42,
    }
    (output_dir.parent / "benchmark_manifest.json").write_text(json.dumps(manifest))

    saved = sys.argv
    sys.argv = [
        "rq1_report.py",
        "--benchmark-id", "test-rq1",
        "--manifest", str(output_dir.parent / "benchmark_manifest.json"),
        "--output-dir", str(output_dir),
    ]
    try:
        with pytest.raises(RuntimeError, match="data_source mismatch"):
            rq1_report.main()
    finally:
        sys.argv = saved


# ---- fakes ----

class _StubRun:
    def __init__(self, run_id, run_name, status, tags, params, metrics):
        self.info = type("I", (), {"run_id": run_id, "run_name": run_name, "status": status})()
        self.data = type("D", (), {"tags": tags, "params": params, "metrics": metrics})()


class _StubMlflowTracking:
    def __init__(self, runs):
        self._runs = runs

    def MlflowClient(self):
        return _StubClient(self._runs)


class _StubClient:
    def __init__(self, runs):
        self._runs = runs

    def get_experiment_by_name(self, name):
        return type("E", (), {"experiment_id": "1"})()

    def search_runs(self, exp_ids):
        return list(self._runs)
