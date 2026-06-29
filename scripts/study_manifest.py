"""Shared manifest helpers for RQ2/RQ3/RQ4 study campaigns.

Follows the same lifecycle pattern as the RQ1 benchmark manifest.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_run_key(variant: str, seed: int) -> str:
    return f"{variant}:{seed}"


def manifest_path_for_output_dir(output_dir: Path) -> Path:
    """Resolve the benchmark manifest path from a root or reports/stats dir."""
    if output_dir.name in {"reports", "stats"}:
        return output_dir.parent / "benchmark_manifest.json"
    return output_dir / "benchmark_manifest.json"


def create_manifest(
    path: Path,
    *,
    variants: list[str],
    seeds: list[int],
    benchmark_id: str,
    backbone: str,
) -> None:
    if path.exists():
        raise RuntimeError(f"Study manifest already exists: {path}")
    data = {
        "benchmark_id": benchmark_id,
        "backbone": backbone,
        "status": "running",
        "expected_variants": variants,
        "expected_seeds": seeds,
        "completed_run_keys": [],
        "failed_run_keys": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_manifest(path: Path, require_completed: bool = False) -> dict:
    if not path.exists():
        raise RuntimeError(f"Study manifest not found: {path}")
    data = _read_manifest(path)
    if require_completed and data.get("status") != "completed":
        raise RuntimeError(
            f"Study campaign {data.get('benchmark_id', '?')} is not yet completed "
            f"(status={data.get('status')!r}). Resume training or mark as completed."
        )
    return data


def is_completed(path: Path, variant: str, seed: int) -> bool:
    key = build_run_key(variant, seed)
    data = _read_manifest(path)
    return key in data.get("completed_run_keys", [])


def mark_completed(path: Path, variant: str, seed: int) -> None:
    key = build_run_key(variant, seed)
    data = _read_manifest(path)
    if key not in data["completed_run_keys"]:
        data["completed_run_keys"].append(key)
        # Remove from failed if previously marked
        if key in data.get("failed_run_keys", []):
            data["failed_run_keys"].remove(key)
    _write_manifest(path, data)


def mark_failed(path: Path, variant: str, seed: int) -> None:
    key = build_run_key(variant, seed)
    data = _read_manifest(path)
    if key not in data["failed_run_keys"]:
        data["failed_run_keys"].append(key)
    _write_manifest(path, data)


def finalize_manifest(path: Path) -> None:
    data = _read_manifest(path)
    expected = len(data["expected_variants"]) * len(data["expected_seeds"])
    actual = len(data["completed_run_keys"])
    if actual < expected:
        raise RuntimeError(
            f"Not all expected runs completed: "
            f"{actual}/{expected} finished, {expected - actual} missing"
        )
    data["status"] = "completed"
    _write_manifest(path, data)
