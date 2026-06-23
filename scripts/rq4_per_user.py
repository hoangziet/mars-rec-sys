"""
scripts/rq4_per_user.py
=======================
Atomic helper for the RQ4 per-user artifact lifecycle.

The per-user CSV is the single artifact that downstream statistical
comparison (rq4_compare) consumes per (variant, seed). Because the
artifact can be partially written if anything goes wrong between
disk write and MLflow upload, this helper enforces a strict
order:

    1. Write per-user CSV to a temp path (no overwrite risk).
    2. Validate it (non-empty, required columns present).
    3. Atomically rename into the canonical path.
    4. Optionally upload to MLflow.
    5. Only after all four succeed, promote ``per_user_complete=true``
       and ``reportable=true`` on the MLflow run.

If any step fails:
    - The temp file is removed.
    - The canonical file (if it existed) is left untouched.
    - The MLflow run is marked FAILED via ``MlflowClient.set_terminated``.
    - The original exception is re-raised.

The collector (rq4_collect) re-checks the artifact on disk before
including a run, so a stale partial file cannot be misread as valid.
"""

from __future__ import annotations

import csv
import importlib
import os
import sys
import tempfile
from pathlib import Path

REQUIRED_PER_USER_COLUMNS = (
    "variant",
    "seed",
    "user_idx",
    "target_item",
    "rank",
    "hit_at_10",
    "ndcg_at_10",
    "hit_at_20",
    "ndcg_at_20",
)


class PerUserExportError(RuntimeError):
    """Raised when per-user export cannot be completed cleanly."""


def write_per_user_atomic(
    per_user_rows: list[dict],
    canonical_path: str | Path,
) -> Path:
    """Write per-user rows to ``canonical_path`` atomically.

    Writes to a sibling temp file, validates content, then renames.
    Raises ``PerUserExportError`` on any failure (with the original
    exception chained).
    """
    canonical = Path(canonical_path)
    canonical.parent.mkdir(parents=True, exist_ok=True)

    if not per_user_rows:
        raise PerUserExportError(
            f"Cannot write empty per-user artifact to {canonical}: "
            "evaluation produced no rows."
        )

    missing = [c for c in REQUIRED_PER_USER_COLUMNS if c not in per_user_rows[0]]
    if missing:
        raise PerUserExportError(
            f"Per-user rows missing required columns: {missing}. "
            f"Required: {REQUIRED_PER_USER_COLUMNS}"
        )

    # Sibling temp file — same filesystem so rename is atomic.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{canonical.name}.",
        suffix=".tmp",
        dir=str(canonical.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(REQUIRED_PER_USER_COLUMNS))
            writer.writeheader()
            writer.writerows(per_user_rows)

        # Re-read and validate before promoting.
        with open(tmp_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or list(reader.fieldnames) != list(REQUIRED_PER_USER_COLUMNS):
                raise PerUserExportError(
                    f"Validation failed: written CSV header is "
                    f"{reader.fieldnames!r}, expected {REQUIRED_PER_USER_COLUMNS!r}"
                )
            row_count = sum(1 for _ in reader)
        if row_count != len(per_user_rows):
            raise PerUserExportError(
                f"Validation failed: wrote {len(per_user_rows)} rows but "
                f"file contains {row_count}."
            )

        os.replace(tmp_path, canonical)
    except Exception:
        # Best-effort cleanup of the temp file; never leave a partial
        # artifact behind.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return canonical


def validate_per_user_file(path: str | Path, expected_min_rows: int = 1) -> int:
    """Validate an on-disk per-user CSV. Returns the row count.

    Raises ``PerUserExportError`` if the file is missing, malformed,
    or has too few rows.
    """
    p = Path(path)
    if not p.exists():
        raise PerUserExportError(f"Per-user file missing: {p}")
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise PerUserExportError(f"Per-user file {p} has no header")
        missing = [c for c in REQUIRED_PER_USER_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise PerUserExportError(
                f"Per-user file {p} missing required columns: {missing}"
            )
        row_count = sum(1 for _ in reader)
    if row_count < expected_min_rows:
        raise PerUserExportError(
            f"Per-user file {p} has only {row_count} rows; "
            f"expected at least {expected_min_rows}."
        )
    return row_count


def promote_run_complete(
    run_id: str,
    *,
    mlflow_module=None,
    client_factory=None,
) -> None:
    """Flip per_user_complete and reportable to true on the run.

    Only call this after the per-user artifact has been written,
    validated, and (optionally) uploaded. Raises ``PerUserExportError``
    on any failure; the caller is responsible for marking the run
    FAILED if this throws.
    """
    if not run_id:
        raise PerUserExportError("Cannot promote run: run_id is empty")

    if mlflow_module is None:
        # Lazy import so test fixtures can monkeypatch this module's
        # ``mlflow`` attribute without triggering a real MLflow import.
        try:
            mlflow_module = sys.modules.get("mlflow")
            if mlflow_module is None:
                import mlflow as mlflow_module  # noqa: F811
        except Exception:
            try:
                mlflow_module = importlib.import_module("mlflow")
            except Exception as exc:
                raise PerUserExportError(
                    f"Cannot import mlflow to promote run {run_id}: {exc}"
                ) from exc

    if client_factory is None:
        client = mlflow_module.tracking.MlflowClient()
    else:
        client = client_factory()

    try:
        client.set_tag(run_id, "per_user_complete", "true")
        client.set_tag(run_id, "reportable", "true")
    except Exception as exc:
        raise PerUserExportError(
            f"Failed to promote per_user_complete/reportable on run {run_id}: {exc}"
        ) from exc


def fail_run_atomically(run_id: str, exc: BaseException) -> None:
    """Best-effort: mark the run FAILED so it cannot be picked up by the collector."""
    if not run_id:
        return
    try:
        import mlflow
        mlflow.tracking.MlflowClient().set_terminated(run_id, status="FAILED")
    except Exception:
        pass


# Allow direct execution for smoke checks.
if __name__ == "__main__":  # pragma: no cover
    sys.stderr.write("This module is a helper; import it instead.\n")
    sys.exit(1)