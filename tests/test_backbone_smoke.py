"""Smoke tests: verify RQ2/RQ3/RQ4 are gSASRec-only (no RQ1 winner-artifact wiring)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq2_tune_help_has_no_winner_artifact_flag():
    """RQ2 is gSASRec-only: no --winner-artifact flag should be exposed."""
    from scripts.rq2_tune_alpha import build_parser

    parser = build_parser()
    args = parser.parse_args(["--benchmark-id", "rq2-x"])
    assert not hasattr(args, "winner_artifact")


def test_rq3_tune_help_has_no_winner_artifact_flag():
    """RQ3 is BERT4Rec-only: accepts --rq2-winner, no --winner-artifact flag."""
    from scripts.rq3_tune_metadata import build_parser

    parser = build_parser()
    args = parser.parse_args(["--rq2-winner", "/tmp/rq2_best_watch.json", "--benchmark-id", "rq3-x"])
    assert not hasattr(args, "winner_artifact")


def test_rq4_init_help_has_no_winner_artifact_flag():
    """RQ4 is gSASRec-only: no --winner-artifact flag should be exposed."""
    from scripts.rq4_init_protocol import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--benchmark-id", "rq4-x",
        "--rq2-winners", "/tmp/rq2.json",
        "--rq3-winners", "/tmp/rq3.json",
    ])
    assert not hasattr(args, "winner_artifact")


def test_rq4_init_rejects_non_gsasrec_rq2_backbone(tmp_path):
    """If RQ2 winner declares a non-gsasrec backbone, rq4-init must refuse."""
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "sasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dataset_manifest.json").write_text("{}")

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(RuntimeError, match="RQ2 winner artifact backbone must be 'gsasrec'"):
            rq4_init_main()
    finally:
        sys.argv = saved


def test_rq4_init_rejects_non_gsasrec_rq3_backbone(tmp_path):
    """If RQ3 winner declares a non-gsasrec backbone, rq4-init must refuse."""
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "sasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dataset_manifest.json").write_text("{}")

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(RuntimeError, match="RQ3 winner artifact backbone must be 'gsasrec'"):
            rq4_init_main()
    finally:
        sys.argv = saved


def test_rq4_ablation_rejects_protocol_backbone_not_gsasrec():
    """rq4_ablation must refuse to train against a non-gsasrec protocol."""
    from scripts import rq4_ablation

    protocol = {"backbone": "sasrec"}
    with pytest.raises(RuntimeError, match="RQ4 is gSASRec-only"):
        rq4_ablation._validate_protocol_backbone(protocol)


def test_rq4_ablation_accepts_protocol_backbone_gsasrec():
    """rq4_ablation accepts a gsasrec protocol and returns 'gsasrec'."""
    from scripts import rq4_ablation

    assert rq4_ablation._validate_protocol_backbone({"backbone": "gsasrec"}) == "gsasrec"


def test_rq4_ablation_rejects_missing_protocol_backbone():
    """rq4_ablation must refuse a protocol with no backbone field at all."""
    from scripts import rq4_ablation

    with pytest.raises(RuntimeError, match="RQ4 is gSASRec-only"):
        rq4_ablation._validate_protocol_backbone({})


def test_rq4_collect_result_manifest_carries_gsasrec_backbone(tmp_path, monkeypatch):
    """End-to-end: rq4_collect propagates the protocol backbone to the result
    manifest so downstream reports can show it.
    """
    from scripts import rq4_collect

    protocol = {
        "variants": ["V0", "V1"],
        "neural_seeds": [42, 43],
        "best_alpha": 0.5,
        "best_metadata_variant": "M3",
        "metadata_variants": {"M3": {"use_structured": True, "use_text": True}},
        "benchmark_id": "rq4-test",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))

    def _make_run(variant, seed, alpha, use_structured, use_text):
        from types import SimpleNamespace
        rid = f"rid_{variant}_s{seed}"
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED", run_id=rid, run_name=f"{variant}_s{seed}"),
            data=SimpleNamespace(
                tags={
                    "reportable": "true",
                    "benchmark_id": "rq4-test",
                    "ablation_variant": variant,
                    "confidence_alpha": str(alpha),
                    "use_structured": str(use_structured).lower(),
                    "use_text": str(use_text).lower(),
                    "per_user_complete": "true",
                },
                params={"seed": str(seed)},
                metrics={
                    "best_val_ndcg_at_10": 0.3,
                    "test_NDCG_at_10": 0.25,
                    "test_Recall_at_10": 0.4,
                    "test_NDCG_at_20": 0.3,
                    "test_Recall_at_20": 0.5,
                },
            ),
        )

    runs = []
    for variant, alpha, us, ut in [
        ("V0", 0.0, False, False),
        ("V1", 0.5, False, False),
    ]:
        for seed in (42, 43):
            runs.append(_make_run(variant, seed, alpha, us, ut))

    class _FakeExperiment:
        experiment_id = "exp-1"

    class _FakeClient:
        def get_experiment_by_name(self, name):
            return _FakeExperiment()

        def search_runs(self, ids):
            return runs

    class _FakeMlflow:
        tracking = type("T", (), {"MlflowClient": lambda self=None: _FakeClient()})()

    def _fake_configure_mlflow(mlflow_module=None):
        return None

    monkeypatch.setattr(rq4_collect, "mlflow", _FakeMlflow())
    monkeypatch.setattr(rq4_collect, "configure_mlflow", _fake_configure_mlflow)
    monkeypatch.setattr(rq4_collect, "_validate_provenance_tags", lambda *a, **k: [])
    monkeypatch.setattr(rq4_collect, "_validate_per_user_on_disk", lambda *a, **k: [])

    out = tmp_path / "out"
    (out / "per_user").mkdir(parents=True)
    saved = sys.argv
    sys.argv = [
        "rq4_collect.py",
        "--benchmark-id", "rq4-test",
        "--protocol", str(tmp_path / "protocol.json"),
        "--output-dir", str(out),
        "--data-dir", str(tmp_path),
    ]
    try:
        rq4_collect.main()
    finally:
        sys.argv = saved

    manifest = json.loads((out / "rq4_result_manifest.json").read_text())
    assert manifest["backbone"] == "gsasrec"


def test_rq4_init_freezes_backbone_as_gsasrec(tmp_path):
    """When RQ2 and RQ3 winners both declare gsasrec, the manifest records
    backbone='gsasrec' without any RQ1 winner-artifact input."""
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": str(tmp_path),
    }))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": str(tmp_path),
    }))
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dataset_manifest.json").write_text("{}")

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
        "--seeds", "42",
    ]
    try:
        rq4_init_main()
    finally:
        sys.argv = saved

    manifest = json.loads((tmp_path / "out" / "rq4_protocol_manifest.json").read_text())
    assert manifest["backbone"] == "gsasrec", manifest
    assert manifest["baseline_variant"] == "V0"
    assert "rq1_benchmark_id" not in manifest


def test_rq4_init_rejects_unknown_light_provenance(tmp_path):
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "gsasrec",
    }))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "gsasrec",
    }))

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
        "--seeds", "42",
    ]
    try:
        with pytest.raises(RuntimeError, match="missing preprocessing_version|missing data_source"):
            rq4_init_main()
    finally:
        sys.argv = saved
