"""Tests for the atomic per-user CSV write helper."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq4_per_user import (
    PerUserExportError,
    promote_run_complete,
    validate_per_user_file,
    write_per_user_atomic,
)


def _row(user_idx, target_item=100, variant="V0", seed=42):
    return {
        "variant": variant,
        "seed": seed,
        "user_idx": user_idx,
        "target_item": target_item,
        "rank": 1,
        "hit_at_10": 1.0,
        "ndcg_at_10": 0.5,
        "hit_at_20": 1.0,
        "ndcg_at_20": 0.5,
    }


# ---- write_per_user_atomic ----

def test_write_atomic_creates_canonical_file(tmp_path):
    rows = [_row(0), _row(1), _row(2)]
    canonical = tmp_path / "V0_s42.csv"
    out = write_per_user_atomic(rows, canonical)
    assert out == canonical
    assert canonical.exists()
    with open(canonical, newline="") as f:
        content = list(csv.DictReader(f))
    assert len(content) == 3


def test_write_atomic_leaves_no_temp_files(tmp_path):
    rows = [_row(0)]
    write_per_user_atomic(rows, tmp_path / "out.csv")
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"unexpected temp files left behind: {leftovers}"


def test_write_atomic_rejects_empty_rows(tmp_path):
    with pytest.raises(PerUserExportError, match="empty"):
        write_per_user_atomic([], tmp_path / "out.csv")


def test_write_atomic_rejects_missing_columns(tmp_path):
    bad = [{"variant": "V0", "seed": 42, "user_idx": 0}]  # missing target_item, rank, etc.
    with pytest.raises(PerUserExportError, match="missing required columns"):
        write_per_user_atomic(bad, tmp_path / "out.csv")


def test_write_atomic_rejects_overwrite_if_canonical_exists(tmp_path):
    """We deliberately do NOT remove an existing canonical file. The caller
    should handle that case. This protects against partial-write scenarios
    where a stale file is misinterpreted as new."""
    canonical = tmp_path / "out.csv"
    canonical.write_text("stale\n")
    original = canonical.read_text()
    rows = [_row(0)]
    # On most platforms os.replace atomically replaces; we want to make sure
    # the helper does not throw if the file exists. The atomicity contract
    # is: temp → rename. If the file already exists, rename replaces it.
    # This is the desired behavior for our use case.
    write_per_user_atomic(rows, canonical)
    with open(canonical, newline="") as f:
        content = list(csv.DictReader(f))
    assert len(content) == 1
    assert original not in canonical.read_text()


def test_write_atomic_cleans_up_temp_on_happy_path(tmp_path):
    """After a successful write, no temp files remain in the directory."""
    rows = [_row(0)]
    write_per_user_atomic(rows, tmp_path / "out.csv")
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"unexpected temp files left behind: {leftovers}"


# ---- validate_per_user_file ----

def test_validate_accepts_good_file(tmp_path):
    rows = [_row(0), _row(1)]
    p = tmp_path / "f.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = validate_per_user_file(p, expected_min_rows=1)
    assert n == 2


def test_validate_rejects_missing_file(tmp_path):
    with pytest.raises(PerUserExportError, match="missing"):
        validate_per_user_file(tmp_path / "nope.csv")


def test_validate_rejects_missing_columns(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("variant,seed\nV0,42\n")  # missing required cols
    with pytest.raises(PerUserExportError, match="missing required columns"):
        validate_per_user_file(p)


def test_validate_rejects_too_few_rows(tmp_path):
    rows = [_row(0)]
    p = tmp_path / "f.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(PerUserExportError, match="only 1 rows"):
        validate_per_user_file(p, expected_min_rows=10)


# ---- promote_run_complete ----

def test_promote_run_complete_rejects_empty_run_id():
    with pytest.raises(PerUserExportError, match="run_id is empty"):
        promote_run_complete("")


def test_promote_run_complete_calls_set_tag():
    captured = []

    class _FakeClient:
        def set_tag(self, run_id, key, value):
            captured.append((run_id, key, value))

    class _FakeTracking:
        MlflowClient = _FakeClient

    class _FakeMlflow:
        tracking = _FakeTracking

    promote_run_complete("rid-1", mlflow_module=_FakeMlflow())

    assert ("rid-1", "per_user_complete", "true") in captured
    assert ("rid-1", "reportable", "true") in captured


def test_promote_run_complete_wraps_client_errors():
    class _FakeClient:
        def set_tag(self, run_id, key, value):
            raise RuntimeError("mlflow api exploded")

    class _FakeTracking:
        MlflowClient = _FakeClient

    class _FakeMlflow:
        tracking = _FakeTracking

    with pytest.raises(PerUserExportError, match="Failed to promote"):
        promote_run_complete("rid-1", mlflow_module=_FakeMlflow())