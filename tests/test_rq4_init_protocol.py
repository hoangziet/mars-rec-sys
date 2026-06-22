import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_init_protocol


def _write_winners(tmp_path, alpha=0.5, variant="M3", rq2_bid="rq2-x", rq3_bid="rq3-x"):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({"best_alpha": alpha, "benchmark_id": rq2_bid}))
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({"best_variant": variant, "benchmark_id": rq3_bid}))
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
        "--seeds", "42", "123",
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert manifest["best_alpha"] == 0.5
    assert manifest["best_metadata_variant"] == "M3"
    assert manifest["rq2_benchmark_id"] == "rq2-x"
    assert manifest["rq3_benchmark_id"] == "rq3-x"
    assert manifest["neural_seeds"] == [42, 123]


def test_init_writes_sha256_hashes(tmp_path):
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
    # protocol_sha256 is the self-hash
    assert "protocol_sha256" in manifest
    assert isinstance(manifest["protocol_sha256"], str)
    assert len(manifest["protocol_sha256"]) == 64  # sha256 hex length


def test_init_writes_data_manifest_hash_when_present(tmp_path):
    rq2, rq3 = _write_winners(tmp_path)
    output = tmp_path / "out"
    data_dir = tmp_path / "data"
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "preprocessing_report.json").write_text("{}")
    _run_with_argv([
        "rq4_init_protocol.py",
        "--benchmark-id", "test-rq4",
        "--rq2-winners", str(rq2),
        "--rq3-winners", str(rq3),
        "--output-dir", str(output),
        "--data-dir", str(data_dir),
    ])

    manifest = json.loads((output / "rq4_protocol_manifest.json").read_text())
    assert manifest["data_manifest_sha256"] is not None
    assert len(manifest["data_manifest_sha256"]) == 64


def test_init_rejects_rq2_winner_missing_key(tmp_path):
    rq2 = tmp_path / "rq2_best_alpha.json"
    rq2.write_text(json.dumps({"benchmark_id": "rq2-x"}))  # missing best_alpha
    rq3 = tmp_path / "rq3_best_variant.json"
    rq3.write_text(json.dumps({"best_variant": "M3", "benchmark_id": "rq3-x"}))
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
    rq2.write_text(json.dumps({"best_alpha": 0.5, "benchmark_id": "rq2-x"}))
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
# verify_protocol_hashes: runtime enforcement of data/config/text provenance
# ---------------------------------------------------------------------------


def _make_full_data_dir(tmp_path: Path) -> Path:
    """Build a data_dir containing reports/preprocessing_report.json and
    item_features/text_embeddings.pt, plus a configs/model/*.yaml.
    Returns the data_dir.
    """
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "reports" / "preprocessing_report.json").write_text("{}")
    (tmp_path / "item_features").mkdir(parents=True)
    (tmp_path / "item_features" / "text_embeddings.pt").write_bytes(b"abc")
    configs_dir = tmp_path / "configs" / "model"
    configs_dir.mkdir(parents=True)
    (configs_dir / "x.yaml").write_text("a: 1\n")
    return tmp_path


def test_verify_protocol_hashes_passes_when_all_match(tmp_path):
    data_dir = _make_full_data_dir(tmp_path)

    manifest = {
        "data_manifest_sha256": rq4_init_protocol._sha256_file(
            data_dir / "reports" / "preprocessing_report.json"
        ),
        "config_sha256": rq4_init_protocol._sha256_concat(
            str(data_dir / "configs" / "model" / "*.yaml")
        ),
        "text_artifact_sha256": rq4_init_protocol._sha256_file(
            data_dir / "item_features" / "text_embeddings.pt"
        ),
    }
    rq4_init_protocol.verify_protocol_hashes(
        manifest=manifest,
        data_dir=data_dir,
        configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
    )


def test_verify_protocol_hashes_raises_on_data_drift(tmp_path):
    data_dir = _make_full_data_dir(tmp_path)
    manifest = {
        "data_manifest_sha256": "0" * 64,
        "config_sha256": rq4_init_protocol._sha256_concat(
            str(data_dir / "configs" / "model" / "*.yaml")
        ),
        "text_artifact_sha256": rq4_init_protocol._sha256_file(
            data_dir / "item_features" / "text_embeddings.pt"
        ),
    }
    with pytest.raises(RuntimeError, match="data_manifest_sha256 mismatch"):
        rq4_init_protocol.verify_protocol_hashes(
            manifest=manifest,
            data_dir=data_dir,
            configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
        )


def test_verify_protocol_hashes_raises_on_config_drift(tmp_path):
    data_dir = _make_full_data_dir(tmp_path)
    manifest = {
        "data_manifest_sha256": rq4_init_protocol._sha256_file(
            data_dir / "reports" / "preprocessing_report.json"
        ),
        "config_sha256": "0" * 64,
        "text_artifact_sha256": rq4_init_protocol._sha256_file(
            data_dir / "item_features" / "text_embeddings.pt"
        ),
    }
    with pytest.raises(RuntimeError, match="config_sha256 mismatch"):
        rq4_init_protocol.verify_protocol_hashes(
            manifest=manifest,
            data_dir=data_dir,
            configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
        )


def test_verify_protocol_hashes_raises_on_text_artifact_drift(tmp_path):
    data_dir = _make_full_data_dir(tmp_path)
    manifest = {
        "data_manifest_sha256": rq4_init_protocol._sha256_file(
            data_dir / "reports" / "preprocessing_report.json"
        ),
        "config_sha256": rq4_init_protocol._sha256_concat(
            str(data_dir / "configs" / "model" / "*.yaml")
        ),
        "text_artifact_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="text_artifact_sha256 mismatch"):
        rq4_init_protocol.verify_protocol_hashes(
            manifest=manifest,
            data_dir=data_dir,
            configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
        )


def test_verify_protocol_hashes_skips_unrecorded_hashes(tmp_path):
    """If a hash was None at init time (e.g. data not preprocessed yet), the
    verifier must skip it instead of erroring."""
    manifest = {
        "data_manifest_sha256": None,
        "config_sha256": None,
        "text_artifact_sha256": None,
    }
    rq4_init_protocol.verify_protocol_hashes(
        manifest=manifest, data_dir=tmp_path, configs_glob=str(tmp_path / "missing" / "*.yaml")
    )


# ---------------------------------------------------------------------------
# protocol_sha256 self-hash verification
# ---------------------------------------------------------------------------


def test_verify_protocol_hashes_validates_self_hash(tmp_path):
    """After init writes protocol_sha256, the verifier must check it matches
    a recomputation of the manifest contents (excluding the field itself)."""
    data_dir = _make_full_data_dir(tmp_path)
    report_hash = rq4_init_protocol._sha256_file(
        data_dir / "reports" / "preprocessing_report.json"
    )
    config_hash = rq4_init_protocol._sha256_concat(
        str(data_dir / "configs" / "model" / "*.yaml")
    )
    text_hash = rq4_init_protocol._sha256_file(
        data_dir / "item_features" / "text_embeddings.pt"
    )

    manifest_no_self = {
        "benchmark_id": "test",
        "best_alpha": 0.5,
        "best_metadata_variant": "M3",
        "data_manifest_sha256": report_hash,
        "config_sha256": config_hash,
        "text_artifact_sha256": text_hash,
    }
    manifest_no_self["protocol_sha256"] = hashlib.sha256(
        json.dumps(manifest_no_self, sort_keys=True, indent=2).encode()
    ).hexdigest()

    rq4_init_protocol.verify_protocol_hashes(
        manifest=manifest_no_self,
        data_dir=data_dir,
        configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
    )


def test_verify_protocol_hashes_raises_on_tampered_self_hash(tmp_path):
    """If someone edits best_alpha without recomputing protocol_sha256,
    the verifier must reject the manifest."""
    data_dir = _make_full_data_dir(tmp_path)
    report_hash = rq4_init_protocol._sha256_file(
        data_dir / "reports" / "preprocessing_report.json"
    )
    config_hash = rq4_init_protocol._sha256_concat(
        str(data_dir / "configs" / "model" / "*.yaml")
    )
    text_hash = rq4_init_protocol._sha256_file(
        data_dir / "item_features" / "text_embeddings.pt"
    )

    manifest_no_self = {
        "benchmark_id": "test",
        "best_alpha": 0.5,
        "best_metadata_variant": "M3",
        "data_manifest_sha256": report_hash,
        "config_sha256": config_hash,
        "text_artifact_sha256": text_hash,
    }
    manifest_no_self["protocol_sha256"] = hashlib.sha256(
        json.dumps(manifest_no_self, sort_keys=True, indent=2).encode()
    ).hexdigest()

    # Tamper: change alpha but keep the old self-hash
    manifest_no_self["best_alpha"] = 2.0

    with pytest.raises(RuntimeError, match="protocol_sha256 mismatch"):
        rq4_init_protocol.verify_protocol_hashes(
            manifest=manifest_no_self,
            data_dir=data_dir,
            configs_glob=str(data_dir / "configs" / "model" / "*.yaml"),
        )
