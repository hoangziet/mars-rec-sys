import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.study_manifest import (
    build_run_key,
    create_manifest,
    finalize_manifest,
    is_completed,
    load_manifest,
    mark_completed,
    mark_failed,
)

M = {"variants": ["a", "b"], "seeds": [1, 2], "benchmark_id": "test", "backbone": "bert4rec"}


def test_create_manifest_writes_correct_structure(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["baseline", "wl"], seeds=[42, 123], benchmark_id="rq2-test", backbone="bert4rec")
    data = json.loads(path.read_text())
    assert data["benchmark_id"] == "rq2-test"
    assert data["backbone"] == "bert4rec"
    assert data["status"] == "running"
    assert data["expected_variants"] == ["baseline", "wl"]
    assert data["expected_seeds"] == [42, 123]
    assert data["completed_run_keys"] == []
    assert data["failed_run_keys"] == []


def test_create_manifest_refuses_overwrite(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a"], seeds=[1], benchmark_id="test", backbone="bert4rec")
    with pytest.raises(RuntimeError, match="already exists"):
        create_manifest(path, variants=["b"], seeds=[2], benchmark_id="test2", backbone="bert4rec")


def test_build_run_key_format():
    assert build_run_key("wl", 42) == "wl:42"
    assert build_run_key("baseline", 123) == "baseline:123"


def test_mark_completed_and_is_completed(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a", "b"], seeds=[1, 2], benchmark_id="test", backbone="bert4rec")
    assert not is_completed(path, "a", 1)
    mark_completed(path, "a", 1)
    assert is_completed(path, "a", 1)
    assert not is_completed(path, "b", 1)
    data = json.loads(path.read_text())
    assert "a:1" in data["completed_run_keys"]


def test_mark_failed(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a"], seeds=[1], benchmark_id="test", backbone="bert4rec")
    mark_failed(path, "a", 1)
    data = json.loads(path.read_text())
    assert "a:1" in data["failed_run_keys"]


def test_finalize_manifest_sets_completed(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a"], seeds=[1], benchmark_id="test", backbone="bert4rec")
    mark_completed(path, "a", 1)
    finalize_manifest(path)
    data = json.loads(path.read_text())
    assert data["status"] == "completed"


def test_finalize_refuses_when_incomplete(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a", "b"], seeds=[1, 2], benchmark_id="test", backbone="bert4rec")
    mark_completed(path, "a", 1)
    with pytest.raises(RuntimeError, match="Not all expected runs completed"):
        finalize_manifest(path)


def test_load_manifest_rejects_running_for_report(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a"], seeds=[1], benchmark_id="test", backbone="bert4rec")
    with pytest.raises(RuntimeError, match="not yet completed"):
        load_manifest(path, require_completed=True)


def test_load_manifest_accepts_completed_for_report(tmp_path):
    path = tmp_path / "manifest.json"
    create_manifest(path, variants=["a"], seeds=[1], benchmark_id="test", backbone="bert4rec")
    mark_completed(path, "a", 1)
    finalize_manifest(path)
    data = load_manifest(path, require_completed=True)
    assert data["status"] == "completed"
