from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class DatasetManifestComparison(str, Enum):
    REJECT_DUPLICATE = "reject_duplicate"
    REJECT_SAME_HASH_NEW_VERSION = "reject_same_hash_new_version"
    REJECT_DIFFERENT_HASH_SAME_VERSION = "reject_different_hash_same_version"
    ALLOW_NEW_VERSION = "allow_new_version"
    FIRST_PUBLISH = "first_publish"


def hash_directory(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_dataset_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    raw_data_hash: str,
    processed_data_hash: str,
    preprocessing_config_hash: str,
    git_commit: str,
    source_files: list[str],
    processed_files: list[str],
    split_strategy: str,
    preprocess_inputs: dict[str, str],
) -> dict:
    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "raw_data_hash": raw_data_hash,
        "processed_data_hash": processed_data_hash,
        "preprocessing_config_hash": preprocessing_config_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "split_strategy": split_strategy,
        "source_files": source_files,
        "processed_files": processed_files,
        "preprocess_inputs": preprocess_inputs,
    }


def compare_against_latest_manifest(*, current: dict, latest: dict | None) -> DatasetManifestComparison:
    if latest is None:
        return DatasetManifestComparison.FIRST_PUBLISH

    latest_identity = (
        latest["raw_data_hash"],
        latest["processed_data_hash"],
        latest["preprocessing_config_hash"],
    )
    current_identity = (
        current["raw_data_hash"],
        current["processed_data_hash"],
        current["preprocessing_config_hash"],
    )

    if current_identity == latest_identity and current["dataset_version"] == latest["dataset_version"]:
        return DatasetManifestComparison.REJECT_DUPLICATE
    if current_identity == latest_identity and current["dataset_version"] != latest["dataset_version"]:
        return DatasetManifestComparison.REJECT_SAME_HASH_NEW_VERSION
    if current_identity != latest_identity and current["dataset_version"] == latest["dataset_version"]:
        return DatasetManifestComparison.REJECT_DIFFERENT_HASH_SAME_VERSION
    return DatasetManifestComparison.ALLOW_NEW_VERSION
