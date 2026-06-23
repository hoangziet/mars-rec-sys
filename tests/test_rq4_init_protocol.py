import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_init_protocol


def _write_winners(
    tmp_path,
    alpha=0.5,
    variant="M3",
    rq2_bid="rq2-x",
    rq3_bid="rq3-x",
    backbone="gsasrec",
    preprocessing_version="mars-preprocess-v1",
    data_source=None,
):
    if data_source is None:
        data_source = str(tmp_path)
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({
        "best_alpha": alpha,
        "benchmark_id": rq2_bid,
        "backbone": backbone,
        "preprocessing_version": preprocessing_version,
        "data_source": data_source,
    }))
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({
        "best_variant": variant,
        "benchmark_id": rq3_bid,
        "backbone": backbone,
        "preprocessing_version": preprocessing_version,
        "data_source": data_source,
    }))
    return rq2, rq3


def _run_with_argv(argv):
    saved = sys.argv
    sys.argv = argv
    try:
        rq4_init_protocol.main()
    finally:
        sys.argv = saved


def test_init_reads_winners_from_json(tmp_path):
    rq2, rq3 = _write_winners(tmp_path, alpha=0.5, variant="M3")
    output = tmp_path / "out"
    _run_with_argv([
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
        "--seeds", "42", "123",
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert manifest["best_alpha"] == 0.5
    assert manifest["best_metadata_variant"] == "M3"
    assert manifest["rq2_benchmark_id"] == "rq2-x"
    assert manifest["rq3_benchmark_id"] == "rq3-x"
    assert manifest["neural_seeds"] == [42, 123]
    assert manifest["backbone"] == "gsasrec"
    assert manifest["baseline_variant"] == "V0"


def test_init_does_not_contain_sha256_fields(tmp_path):
    """Lightweight provenance contract: no SHA256 fields in the manifest."""
    rq2, rq3 = _write_winners(tmp_path)
    output = tmp_path / "out"
    _run_with_argv([
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
        "--seeds", "42",
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert "git_commit" in manifest
    assert "metadata_variants" in manifest
    assert "M0" in manifest["metadata_variants"]
    assert manifest["metadata_variants"]["M0"]["use_structured"] is False
    assert manifest["metadata_variants"]["M3"]["use_structured"] is True
    assert manifest["metadata_variants"]["M3"]["use_text"] is True
    # No SHA256 provenance fields.
    for sha_field in (
        "protocol_sha256",
        "dataset_manifest_sha256",
        "config_sha256",
        "text_artifact_sha256",
    ):
        assert sha_field not in manifest, sha_field


def test_init_rejects_rq2_winner_missing_key(tmp_path):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({"benchmark_id": "rq2-x"}))  # missing best_alpha
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({"best_variant": "M3", "benchmark_id": "rq3-x", "backbone": "gsasrec", "preprocessing_version": "v1", "data_source": "/tmp"}))

    output = tmp_path / "out"

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
    ]
    try:
        with pytest.raises(ValueError, match="best_alpha"):
            rq4_init_protocol.main()
    finally:
        sys.argv = saved


def test_init_rejects_rq3_winner_missing_key(tmp_path):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({"best_alpha": 0.5, "benchmark_id": "rq2-x", "backbone": "gsasrec", "preprocessing_version": "v1", "data_source": "/tmp"}))
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({"benchmark_id": "rq3-x"}))  # missing best_variant
    output = tmp_path / "out"

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
    ]
    try:
        with pytest.raises(ValueError, match="best_variant"):
            rq4_init_protocol.main()
    finally:
        sys.argv = saved


def test_init_rejects_existing_manifest(tmp_path):
    rq2, rq3 = _write_winners(tmp_path)
    output = tmp_path / "out"
    output.mkdir(parents=True, exist_ok=True)
    (output / "rq4_protocol_manifest.json").write_text("{}")  # pre-existing

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
    ]
    try:
        with pytest.raises(RuntimeError, match="already exists"):
            rq4_init_protocol.main()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# gSASRec-only contract: RQ4 freezes backbone="gsasrec" without RQ1 artifact.
# ---------------------------------------------------------------------------


def test_rq4_init_does_not_require_winner_artifact(tmp_path):
    rq2, rq3 = _write_winners(tmp_path)
    output = tmp_path / "out"
    _run_with_argv([
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
        "--seeds", "42",
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert manifest["backbone"] == "gsasrec"


def test_rq4_init_rejects_non_gsasrec_rq2_backbone(tmp_path):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "sasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    output = tmp_path / "out"

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(RuntimeError, match="RQ2 winner artifact backbone must be 'gsasrec'"):
            rq4_init_protocol.main()
    finally:
        sys.argv = saved


def test_rq4_init_rejects_non_gsasrec_rq3_backbone(tmp_path):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({
        "best_alpha": 0.5,
        "benchmark_id": "x",
        "backbone": "gsasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({
        "best_variant": "M3",
        "benchmark_id": "y",
        "backbone": "sasrec",
        "preprocessing_version": "v1",
        "data_source": "/tmp/data",
    }))
    output = tmp_path / "out"

    saved = sys.argv
    sys.argv = [
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
    ]
    try:
        with pytest.raises(RuntimeError, match="RQ3 winner artifact backbone must be 'gsasrec'"):
            rq4_init_protocol.main()
    finally:
        sys.argv = saved


def test_rq4_init_freezes_backbone_as_gsasrec_no_rq1_leak(tmp_path):
    """Manifest must record backbone='gsasrec' and must NOT include
    rq1_benchmark_id (no RQ1 leak in the protocol)."""
    rq2, rq3 = _write_winners(tmp_path)
    output = tmp_path / "out"
    _run_with_argv([
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(tmp_path),
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert manifest["backbone"] == "gsasrec"
    assert "rq1_benchmark_id" not in manifest
