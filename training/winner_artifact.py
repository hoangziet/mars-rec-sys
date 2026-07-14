"""
training/winner_artifact.py
============================
The RQ1 winner artifact contract.

`rq1_report` writes a small machine-readable JSON file recording the
winner of an RQ1 benchmark:

    {
      "schema_version":       1,
      "benchmark_id":         "rq1-2026-06-23",
      "winner_model":         "gsasrec",
      "selection_metric":     "best_val_ndcg_at_10",
      "selection_split":      "val",
      "seed_set":             [42, 123, 2024, 3407, 9999],
      "data_source":          "/abs/path/to/data/processed",
      "preprocessing_version": "mars-preprocess-v1"
    }

This artifact is kept for **reporting and audit** of the RQ1 benchmark.
It is **NOT** consumed by the RQ2–RQ3 follow-up studies:

- RQ2 alpha tuning and RQ3 metadata tuning are
  all hardcoded to BERT4Rec as the backbone.
- The RQ2 / RQ3 winner artifacts are
  the actual contract that drives RQ2–RQ3.

``load_winner_artifact`` is still useful for external analysis, audits,
or custom tooling that wants to read the RQ1 record. It validates the
schema and raises ``WinnerArtifactError`` (``ValueError``) on any
malformed input. Heuristic winners (popularity, itemcf) are rejected
because they are not buildable neural backbones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

# Allow direct execution: ``python training/winner_artifact.py ...``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import HEURISTIC_MODELS, SEQUENTIAL_BACKBONES

WINNER_ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_BACKBONES = frozenset(SEQUENTIAL_BACKBONES)
HEURISTIC_BACKBONES = frozenset(HEURISTIC_MODELS)


class WinnerArtifactError(ValueError):
    """Raised when the winner artifact is missing, malformed, or invalid."""


REQUIRED_FIELDS = (
    "benchmark_id",
    "winner_model",
    "selection_metric",
    "selection_split",
    "seed_set",
    "data_source",
    "preprocessing_version",
    "schema_version",
)


def write_winner_artifact(
    path: str | Path,
    *,
    benchmark_id: str,
    winner_model: str,
    selection_metric: str,
    selection_split: str,
    seed_set: Iterable[int],
    data_source: str,
    preprocessing_version: str,
) -> Path:
    """Write a winner artifact JSON file. Returns the resolved path."""
    artifact = {
        "schema_version": WINNER_ARTIFACT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "winner_model": winner_model,
        "selection_metric": selection_metric,
        "selection_split": selection_split,
        "seed_set": [int(s) for s in seed_set],
        "data_source": str(data_source),
        "preprocessing_version": str(preprocessing_version),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    return out


def load_winner_artifact(
    path: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    allowed_backbones: Iterable[str] | None = None,
) -> dict:
    """Load and validate a winner artifact.

    Args:
        path: Path to the JSON file.
        expected_benchmark_id: If given, ``benchmark_id`` must match.
        allowed_backbones: If given, ``winner_model`` must be one of these
            (defaults to ``SEQUENTIAL_BACKBONES``).

    Raises:
        WinnerArtifactError with a clear, fixable message on any issue.
    """
    p = Path(path)
    if not p.exists():
        raise WinnerArtifactError(
            f"Winner artifact not found at {p}. "
            "Run `make rq1-report` first to produce it."
        )

    try:
        artifact = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise WinnerArtifactError(
            f"Winner artifact at {p} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(artifact, dict):
        raise WinnerArtifactError(
            f"Winner artifact at {p} must be a JSON object, got {type(artifact).__name__}"
        )

    missing = [k for k in REQUIRED_FIELDS if k not in artifact]
    if missing:
        raise WinnerArtifactError(
            f"Winner artifact at {p} missing required fields: {missing}. "
            f"Required: {REQUIRED_FIELDS}"
        )

    winner_model = artifact["winner_model"]
    allowed = set(allowed_backbones) if allowed_backbones is not None else set(SUPPORTED_BACKBONES)
    if winner_model not in allowed:
        raise WinnerArtifactError(
            f"Winner artifact at {p} has winner_model={winner_model!r} "
            f"which is not in the allowed backbone set {sorted(allowed)}. "
            f"Supported: {sorted(SUPPORTED_BACKBONES)}. "
            "Re-run `make rq1-report` after a benchmark that includes a supported backbone."
        )

    seed_set = artifact["seed_set"]
    if not isinstance(seed_set, list) or not seed_set:
        raise WinnerArtifactError(
            f"Winner artifact at {p} has invalid seed_set={seed_set!r}; "
            "expected non-empty list of integers."
        )
    try:
        seed_set = [int(s) for s in seed_set]
    except (TypeError, ValueError) as exc:
        raise WinnerArtifactError(
            f"Winner artifact at {p} seed_set contains non-integers: {seed_set!r}"
        ) from exc
    artifact["seed_set"] = seed_set

    benchmark_id = str(artifact["benchmark_id"])
    if expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
        raise WinnerArtifactError(
            f"Winner artifact benchmark_id={benchmark_id!r} does not match "
            f"expected={expected_benchmark_id!r}. "
            "Pass a different winner artifact or rerun RQ1 with the matching benchmark_id."
        )

    return artifact


def is_supported_backbone(name: str) -> bool:
    """True iff the given name is a neural backbone we can build."""
    return name in SUPPORTED_BACKBONES


def is_heuristic_backbone(name: str) -> bool:
    return name in HEURISTIC_BACKBONES