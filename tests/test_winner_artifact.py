"""Tests for the RQ1 winner artifact contract."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.winner_artifact import (
    REQUIRED_FIELDS,
    WinnerArtifactError,
    is_supported_backbone,
    load_winner_artifact,
    write_winner_artifact,
)


def _write_minimal(path, **overrides):
    base = {
        "schema_version": 1,
        "benchmark_id": "rq1-x",
        "winner_model": "gsasrec",
        "selection_metric": "best_val_ndcg_at_10",
        "selection_split": "val",
        "seed_set": [42, 123],
        "data_source": "/tmp/data",
        "preprocessing_version": "mars-preprocess-v1",
    }
    base.update(overrides)
    path.write_text(json.dumps(base, indent=2))


def test_write_then_load_roundtrip(tmp_path):
    out = tmp_path / "w.json"
    write_winner_artifact(
        out,
        benchmark_id="b1",
        winner_model="gsasrec",
        selection_metric="best_val_ndcg_at_10",
        selection_split="val",
        seed_set=[1, 2, 3],
        data_source="/abs/data",
        preprocessing_version="v1",
    )
    loaded = load_winner_artifact(out)
    assert loaded["benchmark_id"] == "b1"
    assert loaded["winner_model"] == "gsasrec"
    assert loaded["seed_set"] == [1, 2, 3]


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(WinnerArtifactError, match="Winner artifact not found"):
        load_winner_artifact(tmp_path / "nope.json")


def test_load_rejects_malformed_json(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("{not json")
    with pytest.raises(WinnerArtifactError, match="not valid JSON"):
        load_winner_artifact(p)


def test_load_rejects_non_dict(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("[1,2,3]")
    with pytest.raises(WinnerArtifactError, match="must be a JSON object"):
        load_winner_artifact(p)


def test_load_rejects_missing_required_field(tmp_path):
    p = tmp_path / "w.json"
    base = {k: "x" for k in REQUIRED_FIELDS if k != "benchmark_id"}
    base["schema_version"] = 1
    p.write_text(json.dumps(base))
    with pytest.raises(WinnerArtifactError, match="missing required fields"):
        load_winner_artifact(p)


def test_load_rejects_unknown_backbone(tmp_path):
    p = tmp_path / "w.json"
    _write_minimal(p, winner_model="transformerxl")
    with pytest.raises(WinnerArtifactError, match="not in the allowed backbone set"):
        load_winner_artifact(p)


def test_load_rejects_heuristic_winner_by_default(tmp_path):
    """Heuristic winners cannot drive RQ2/RQ3/RQ4 neural pipeline."""
    p = tmp_path / "w.json"
    _write_minimal(p, winner_model="popularity")
    with pytest.raises(WinnerArtifactError, match="not in the allowed backbone set"):
        load_winner_artifact(p)


def test_load_allows_heuristic_when_explicitly_allowed(tmp_path):
    p = tmp_path / "w.json"
    _write_minimal(p, winner_model="popularity")
    loaded = load_winner_artifact(p, allowed_backbones=["popularity", "itemcf"])
    assert loaded["winner_model"] == "popularity"


def test_load_rejects_benchmark_mismatch(tmp_path):
    p = tmp_path / "w.json"
    _write_minimal(p, benchmark_id="rq1-A")
    with pytest.raises(WinnerArtifactError, match="does not match expected"):
        load_winner_artifact(p, expected_benchmark_id="rq1-B")


def test_load_rejects_non_integer_seed(tmp_path):
    p = tmp_path / "w.json"
    base = {k: "x" for k in REQUIRED_FIELDS}
    base["schema_version"] = 1
    base["winner_model"] = "gsasrec"
    base["seed_set"] = ["not", "an", "int"]
    p.write_text(json.dumps(base))
    with pytest.raises(WinnerArtifactError, match="non-integers"):
        load_winner_artifact(p)


def test_load_rejects_empty_seed_set(tmp_path):
    p = tmp_path / "w.json"
    base = {k: "x" for k in REQUIRED_FIELDS}
    base["schema_version"] = 1
    base["winner_model"] = "gsasrec"
    base["seed_set"] = []
    p.write_text(json.dumps(base))
    with pytest.raises(WinnerArtifactError, match="non-empty list"):
        load_winner_artifact(p)


def test_is_supported_backbone():
    assert is_supported_backbone("gsasrec")
    assert is_supported_backbone("sasrec")
    assert not is_supported_backbone("popularity")
    assert not is_supported_backbone("transformerxl")


def test_required_fields_includes_backbone():
    """Backbone must be part of the required contract, not a silent default."""
    assert "backbone" in REQUIRED_FIELDS or "winner_model" in REQUIRED_FIELDS
    assert "winner_model" in REQUIRED_FIELDS