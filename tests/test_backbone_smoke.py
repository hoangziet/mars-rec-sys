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
    """RQ3 is gSASRec-only: no --winner-artifact flag should be exposed."""
    from scripts.rq3_tune_metadata import build_parser

    parser = build_parser()
    args = parser.parse_args(["--benchmark-id", "rq3-x"])
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