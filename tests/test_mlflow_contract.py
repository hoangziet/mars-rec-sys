import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import (
    BENCHMARK_EXPERIMENT_NAME,
    RQ2_EXPERIMENT_NAME,
    RQ3_EXPERIMENT_NAME,
    RQ4_EXPERIMENT_NAME,
    SHARED_EXPERIMENTS,
    SMOKE_EXPERIMENT_NAME,
    TRAINING_EXPERIMENTS,
    build_run_name,
    build_training_tags,
    get_experiment_name_for_phase,
)


def test_get_experiment_name_for_phase_maps_supported_phase_experiments():
    assert get_experiment_name_for_phase("smoke") == "mars_smoke"
    assert get_experiment_name_for_phase("benchmark") == "mars_benchmark"


def test_get_experiment_name_for_phase_rejects_unknown_phase():
    with pytest.raises(ValueError, match="Unsupported MLflow phase"):
        get_experiment_name_for_phase("tuning")


def test_build_run_name_for_training_variant_without_alpha():
    assert build_run_name("sasrec", 42, variant="base") == "sasrec-base-s42"


def test_build_run_name_for_training_variant_with_alpha():
    assert build_run_name("sasrec", 42, variant="confidence", alpha=0.5) == "sasrec-confidence-a0.5-s42"


def test_build_training_tags_for_heuristic_model_uses_backbone_none():
    tags = build_training_tags(
        model_name="popularity",
        phase="benchmark",
        variant="base",
        reportable=True,
    )

    assert tags["project"] == "mars-rec-sys"
    assert tags["phase"] == "benchmark"
    assert tags["backbone"] == "none"
    assert tags["reportable"] == "true"
    assert "dataset_version" not in tags
    assert "git_commit" not in tags


def test_contract_constants_expose_expected_shared_experiments():
    assert SMOKE_EXPERIMENT_NAME == "mars_smoke"
    assert BENCHMARK_EXPERIMENT_NAME == "mars_benchmark"
    assert RQ2_EXPERIMENT_NAME == "mars_confidence_tuning"
    assert RQ3_EXPERIMENT_NAME == "mars_metadata_tuning"
    assert RQ4_EXPERIMENT_NAME == "mars_final_ablation"
    assert SHARED_EXPERIMENTS == {"reports": "mars_reports"}
    assert TRAINING_EXPERIMENTS["rq4_final"] == "mars_final_ablation"
