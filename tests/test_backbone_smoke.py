"""Smoke tests: verify RQ2 is gSASRec-only (no RQ1 winner-artifact wiring)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_winner_artifact(path, backbone="sasrec"):
    path.write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": "rq1-x",
        "winner_model": backbone,
        "selection_metric": "best_val_ndcg_at_10",
        "selection_split": "val",
        "seed_set": [42, 123],
        "data_source": "/tmp/data",
        "preprocessing_version": "v1",
    }))


def test_rq2_tune_help_has_no_winner_artifact_flag():
    """RQ2 is gSASRec-only: no --winner-artifact flag should be exposed."""
    from scripts.rq2_tune_alpha import build_parser

    parser = build_parser()
    args = parser.parse_args(["--benchmark-id", "rq2-x"])
    assert not hasattr(args, "winner_artifact")


def test_rq3_tune_resolves_backbone_from_winner_artifact():
    from scripts.rq3_tune_metadata import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--winner-artifact", "/tmp/winner.json",
        "--benchmark-id", "rq3-x",
    ])
    assert args.winner_artifact == "/tmp/winner.json"


def test_rq4_init_requires_winner_artifact(tmp_path):
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({"best_alpha": 0.5, "benchmark_id": "x", "backbone": "gsasrec"}))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({"best_variant": "M3", "benchmark_id": "y", "backbone": "gsasrec"}))
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
        with pytest.raises(SystemExit, match="requires --winner-artifact"):
            rq4_init_main()
    finally:
        sys.argv = saved


def test_rq4_init_rejects_unsupported_backbone_from_artifact(tmp_path):
    from scripts.rq4_init_protocol import main as rq4_init_main
    from training.winner_artifact import WinnerArtifactError

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({"best_alpha": 0.5, "benchmark_id": "x", "backbone": "gsasrec"}))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({"best_variant": "M3", "benchmark_id": "y", "backbone": "gsasrec"}))
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dataset_manifest.json").write_text("{}")
    winner = tmp_path / "winner.json"
    _make_winner_artifact(winner, backbone="transformerxl")

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--winner-artifact", str(winner),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(WinnerArtifactError, match="not in the allowed backbone set"):
            rq4_init_main()
    finally:
        sys.argv = saved


def test_rq4_init_accepts_sasrec_backbone(tmp_path):
    from scripts.rq4_init_protocol import main as rq4_init_main

    rq2 = tmp_path / "rq2.json"
    rq2.write_text(json.dumps({"best_alpha": 0.5, "benchmark_id": "x", "backbone": "sasrec"}))
    rq3 = tmp_path / "rq3.json"
    rq3.write_text(json.dumps({"best_variant": "M3", "benchmark_id": "y", "backbone": "sasrec"}))
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dataset_manifest.json").write_text("{}")
    winner = tmp_path / "winner.json"
    _make_winner_artifact(winner, backbone="sasrec")

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "rq4-x",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--winner-artifact", str(winner),
        "--output-dir", str(tmp_path / "out"),
        "--data-dir", str(tmp_path),
        "--seeds", "42",
    ]
    try:
        rq4_init_main()
    finally:
        sys.argv = saved

    manifest = json.loads((tmp_path / "out" / "rq4_protocol_manifest.json").read_text())
    assert manifest["backbone"] == "sasrec", manifest
    assert manifest["baseline_variant"] == "V0"