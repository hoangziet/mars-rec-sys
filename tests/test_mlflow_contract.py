import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import (
    SHARED_EXPERIMENTS,
    TRAINING_EXPERIMENTS,
    build_run_name,
    build_training_tags,
    get_experiment_name_for_phase,
)


def test_get_experiment_name_for_phase_maps_all_training_phases():
    assert get_experiment_name_for_phase("smoke") == "mars_smoke"
    assert get_experiment_name_for_phase("benchmark") == "mars_benchmark"
    assert get_experiment_name_for_phase("tuning") == "mars_tuning"
    assert get_experiment_name_for_phase("ablation") == "mars_ablation"
    assert get_experiment_name_for_phase("final") == "mars_final"


def test_get_experiment_name_for_phase_rejects_unknown_phase():
    with pytest.raises(ValueError, match="Unsupported MLflow phase"):
        get_experiment_name_for_phase("custom")


def test_build_run_name_for_training_variant_without_alpha():
    assert build_run_name("sasrec", 42, variant="base") == "sasrec-base-s42"


def test_build_run_name_for_training_variant_with_alpha():
    assert build_run_name("sasrec", 42, variant="confidence", alpha=0.5) == "sasrec-confidence-a0.5-s42"


def test_build_training_tags_for_heuristic_model_uses_backbone_none():
    tags = build_training_tags(
        model_name="popularity",
        phase="benchmark",
        variant="base",
        git_commit="abc123",
        reportable=True,
    )

    assert tags["project"] == "mars-rec-sys"
    assert tags["phase"] == "benchmark"
    assert tags["backbone"] == "none"
    assert tags["reportable"] == "true"
    assert "dataset_version" not in tags


def test_contract_constants_expose_expected_shared_experiments():
    assert SHARED_EXPERIMENTS == {"reports": "mars_reports"}
    assert TRAINING_EXPERIMENTS["final"] == "mars_final"
