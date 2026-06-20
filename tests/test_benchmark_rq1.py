import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.report_rq1 import (
    format_metric_summary,
    parse_args as parse_report_args,
    required_run_count_for_model,
    summarize_metric_values,
    validate_model_set,
    validate_seed_set,
)
from scripts.train_all import (
    build_benchmark_manifest,
    build_benchmark_run_dir,
    build_heuristic_save_target,
    get_seeds_for_model,
    parse_args as parse_train_all_args,
)
from training.trainer import Trainer


def test_get_seeds_for_neural_model_returns_all_requested_seeds():
    seeds = [42, 123, 2024, 3407, 9999]
    assert get_seeds_for_model("sasrec", seeds) == seeds


def test_get_seeds_for_heuristic_model_returns_first_seed_only():
    seeds = [42, 123, 2024, 3407, 9999]
    assert get_seeds_for_model("popularity", seeds) == [42]


def test_build_benchmark_run_dir_nests_benchmark_model_and_seed():
    run_dir = build_benchmark_run_dir("experiments", "rq1-v1", "sasrec", 42)
    assert run_dir == Path("experiments") / "benchmark" / "rq1-v1" / "sasrec" / "seed_42"


def test_build_heuristic_save_target_uses_benchmark_dir_only():
    run_dir = Path("experiments") / "benchmark" / "rq1-v1" / "popularity" / "seed_42"
    assert build_heuristic_save_target("popularity", run_dir) == run_dir / "popularity_model.json"
    assert build_heuristic_save_target("itemcf", run_dir) == run_dir


def test_trainer_uses_explicit_run_dir(tmp_path):
    run_dir = tmp_path / "benchmark" / "rq1-v1" / "sasrec" / "seed_42"
    trainer = Trainer("sasrec", "cpu", output_dir="experiments", run_dir=run_dir, use_mlflow=False)
    assert trainer.output_dir == run_dir


def test_required_run_count_for_neural_model_is_full_seed_count():
    assert required_run_count_for_model("sasrec", 5) == 5


def test_required_run_count_for_heuristic_model_is_one():
    assert required_run_count_for_model("popularity", 5) == 1


def test_summarize_metric_values_uses_sample_std():
    summary = summarize_metric_values([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)
    assert summary["runs"] == 3


def test_summarize_metric_values_for_single_run_has_zero_std():
    summary = summarize_metric_values([0.5])
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["std"] is None
    assert summary["ci95_low"] is None
    assert summary["ci95_high"] is None
    assert summary["runs"] == 1


def test_format_metric_summary_uses_na_for_missing_std_and_ci():
    rendered = format_metric_summary({
        "mean": 0.5,
        "std": None,
        "ci95_low": None,
        "ci95_high": None,
        "runs": 1,
    })
    assert rendered == "0.5000 (std/CI: N/A)"


def test_train_all_parse_args_supports_protocol_version_and_seeds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_all.py",
            "sasrec",
            "--benchmark-id",
            "rq1-smoke",
            "--protocol-version",
            "rq1-v1",
            "--preprocessing-version",
            "mars-preprocess-v1",
            "--seeds",
            "42",
        ],
    )
    args = parse_train_all_args()
    assert args.benchmark_id == "rq1-smoke"
    assert args.protocol_version == "rq1-v1"
    assert args.preprocessing_version == "mars-preprocess-v1"
    assert args.seeds == [42]


def test_report_rq1_parse_args_supports_expected_neural_runs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["report_rq1.py", "--benchmark-id", "rq1-smoke", "--manifest", "experiments/benchmark/rq1-smoke/benchmark_manifest.json", "--expected-neural-runs", "1"],
    )
    args = parse_report_args()
    assert args.expected_neural_runs == 1
    assert args.manifest.endswith("benchmark_manifest.json")


def test_build_benchmark_manifest_contains_expected_models_and_seed_policy():
    manifest = build_benchmark_manifest(
        benchmark_id="rq1-v1",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        expected_models=["sasrec", "popularity"],
        neural_seeds=[42, 123, 2024, 3407, 9999],
        heuristic_seed=42,
    )
    assert manifest["benchmark_id"] == "rq1-v1"
    assert manifest["protocol_version"] == "rq1-v1"
    assert manifest["preprocessing_version"] == "mars-preprocess-v1"
    assert manifest["expected_models"] == ["sasrec", "popularity"]
    assert manifest["neural_seeds"] == [42, 123, 2024, 3407, 9999]
    assert manifest["heuristic_seed"] == 42


def test_validate_model_set_raises_when_model_missing():
    with pytest.raises(RuntimeError, match="Missing=.*bert4rec"):
        validate_model_set({"sasrec", "gsasrec"}, {"sasrec", "gsasrec", "bert4rec"})


def test_validate_seed_set_detects_duplicates():
    with pytest.raises(RuntimeError, match="duplicated seeds"):
        validate_seed_set("sasrec", [42, 42], {42})


def test_validate_seed_set_rejects_wrong_seed_set():
    with pytest.raises(RuntimeError, match="expected seeds"):
        validate_seed_set("sasrec", [42, 123], {42, 2024})


def test_report_cli_no_longer_exposes_dataset_version(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["report_rq1.py", "--benchmark-id", "rq1-smoke"],
    )
    args = parse_report_args()
    assert not hasattr(args, "dataset_version")
