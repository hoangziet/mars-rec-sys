from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import HEURISTIC_MODELS
from training.mlflow_utils import configure_mlflow


def required_run_count_for_model(model_name: str, seed_count: int) -> int:
    return 1 if model_name in HEURISTIC_MODELS else seed_count


def summarize_metric_values(values: list[float]) -> dict[str, float | int]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())

    if n < 2:
        return {
            "mean": mean,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "runs": int(n),
        }

    stats = DescrStatsW(arr)
    ci_low, ci_high = stats.tconfint_mean(alpha=0.05)
    return {
        "mean": mean,
        "std": float(arr.std(ddof=1)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "runs": int(n),
    }


def format_metric_summary(summary: dict[str, float | int | None]) -> str:
    mean = summary["mean"]
    std = summary["std"]
    low = summary["ci95_low"]
    high = summary["ci95_high"]
    if std is None:
        return f"{mean:.4f} (std/CI: N/A)"
    return f"{mean:.4f} ± {std:.4f} [{low:.4f}, {high:.4f}]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate RQ1 benchmark runs from MLflow.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-neural-runs", type=int, default=5)
    parser.add_argument("--manifest", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_model_set(actual_models: set[str], expected_models: set[str]) -> None:
    missing_models = expected_models - actual_models
    unexpected_models = actual_models - expected_models
    if missing_models or unexpected_models:
        raise RuntimeError(
            f"Invalid model set. Missing={sorted(missing_models)}, Unexpected={sorted(unexpected_models)}"
        )


def validate_seed_set(model_name: str, actual_seed_list: list[int], expected_seed_set: set[int]) -> None:
    actual_seed_set = set(actual_seed_list)
    if len(actual_seed_list) != len(actual_seed_set):
        raise RuntimeError(f"{model_name}: duplicated seeds")
    if actual_seed_set != expected_seed_set:
        raise RuntimeError(
            f"{model_name}: expected seeds {sorted(expected_seed_set)}, got {sorted(actual_seed_set)}"
        )


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    manifest_path = Path(args.manifest) if args.manifest else Path("experiments") / "benchmark" / args.benchmark_id / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("mars_benchmark")
    if experiment is None:
        raise RuntimeError("Experiment 'mars_benchmark' does not exist")

    runs = client.search_runs([experiment.experiment_id])
    selected_runs = []
    for run in runs:
        tags = run.data.tags
        if run.info.status != "FINISHED":
            continue
        if tags.get("benchmark_id") != args.benchmark_id:
            continue
        if tags.get("reportable") != "true":
            continue
        if tags.get("variant") != "base":
            continue
        if tags.get("protocol_version") != manifest["protocol_version"]:
            continue
        if tags.get("preprocessing_version") != manifest["preprocessing_version"]:
            continue
        selected_runs.append(run)

    if not selected_runs:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    grouped: dict[str, list] = {}
    for run in selected_runs:
        grouped.setdefault(run.data.tags["model"], []).append(run)

    validate_model_set(set(grouped), set(manifest["expected_models"]))

    seed_count = args.expected_neural_runs
    for model_name, model_runs in grouped.items():
        expected = required_run_count_for_model(model_name, seed_count)
        if len(model_runs) != expected:
            raise RuntimeError(
                f"Model {model_name} has {len(model_runs)} runs but expected {expected} for benchmark {args.benchmark_id}"
            )
        if model_name not in HEURISTIC_MODELS:
            actual_seed_list = [int(run.data.params["seed"]) for run in model_runs]
            validate_seed_set(model_name, actual_seed_list, set(manifest["neural_seeds"]))

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "benchmark" / args.benchmark_id / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    summary_rows = []
    for model_name, model_runs in sorted(grouped.items()):
        val_values = []
        test_ndcg10_values = []
        test_recall10_values = []
        test_ndcg20_values = []
        test_recall20_values = []

        for run in model_runs:
            metrics = run.data.metrics
            required_metrics = {
                "best_val_ndcg_at_10",
                "best_epoch",
                "test_NDCG_at_10",
                "test_Recall_at_10",
                "test_NDCG_at_20",
                "test_Recall_at_20",
            }
            missing = sorted(required_metrics - set(metrics.keys()))
            if missing:
                raise RuntimeError(f"Run {run.info.run_id} is missing metrics: {missing}")
            row = {
                "model": model_name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "best_val_ndcg_at_10": metrics["best_val_ndcg_at_10"],
                "best_epoch": metrics["best_epoch"],
                "test_NDCG_at_10": metrics["test_NDCG_at_10"],
                "test_Recall_at_10": metrics["test_Recall_at_10"],
                "test_NDCG_at_20": metrics["test_NDCG_at_20"],
                "test_Recall_at_20": metrics["test_Recall_at_20"],
            }
            run_rows.append(row)
            val_values.append(row["best_val_ndcg_at_10"])
            test_ndcg10_values.append(row["test_NDCG_at_10"])
            test_recall10_values.append(row["test_Recall_at_10"])
            test_ndcg20_values.append(row["test_NDCG_at_20"])
            test_recall20_values.append(row["test_Recall_at_20"])

        summary_rows.append(
            {
                "model": model_name,
                "runs": len(model_runs),
                "val_ndcg_at_10": summarize_metric_values(val_values),
                "test_ndcg_at_10": summarize_metric_values(test_ndcg10_values),
                "test_recall_at_10": summarize_metric_values(test_recall10_values),
                "test_ndcg_at_20": summarize_metric_values(test_ndcg20_values),
                "test_recall_at_20": summarize_metric_values(test_recall20_values),
            }
        )

    summary_rows.sort(key=lambda row: row["val_ndcg_at_10"]["mean"], reverse=True)
    for rank, row in enumerate(summary_rows, start=1):
        row["validation_rank"] = rank

    with open(output_dir / "rq1_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()) if run_rows else ["model"])
        writer.writeheader()
        writer.writerows(run_rows)

    with open(output_dir / "rq1_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(output_dir / "rq1_summary.csv", "w", newline="") as f:
        fieldnames = [
            "validation_rank",
            "model",
            "runs",
            "val_ndcg_at_10_mean",
            "val_ndcg_at_10_std",
            "test_ndcg_at_10_mean",
            "test_ndcg_at_10_std",
            "test_recall_at_10_mean",
            "test_recall_at_10_std",
            "test_ndcg_at_20_mean",
            "test_ndcg_at_20_std",
            "test_recall_at_20_mean",
            "test_recall_at_20_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "validation_rank": row["validation_rank"],
                    "model": row["model"],
                    "runs": row["runs"],
                    "val_ndcg_at_10_mean": row["val_ndcg_at_10"]["mean"],
                    "val_ndcg_at_10_std": row["val_ndcg_at_10"]["std"] or "",
                    "test_ndcg_at_10_mean": row["test_ndcg_at_10"]["mean"],
                    "test_ndcg_at_10_std": row["test_ndcg_at_10"]["std"] or "",
                    "test_recall_at_10_mean": row["test_recall_at_10"]["mean"],
                    "test_recall_at_10_std": row["test_recall_at_10"]["std"] or "",
                    "test_ndcg_at_20_mean": row["test_ndcg_at_20"]["mean"],
                    "test_ndcg_at_20_std": row["test_ndcg_at_20"]["std"] or "",
                    "test_recall_at_20_mean": row["test_recall_at_20"]["mean"],
                    "test_recall_at_20_std": row["test_recall_at_20"]["std"] or "",
                }
            )

    with open(output_dir / "rq1_table.md", "w") as f:
        f.write("Models are ranked by mean validation NDCG@10. Test metrics are reported for final evaluation only.\n\n")
        f.write("| Validation Rank | Model | Runs | Val NDCG@10 | Test NDCG@10 | Recall@10 | NDCG@20 | Recall@20 |\n")
        f.write("| -------------: | ----- | ---: | ----------: | -----------: | --------: | ------: | --------: |\n")
        for row in summary_rows:
            rank = row["validation_rank"]
            f.write(
                f"| {rank} | {row['model']} | {row['runs']} | "
                f"{format_metric_summary(row['val_ndcg_at_10'])} | "
                f"{format_metric_summary(row['test_ndcg_at_10'])} | "
                f"{format_metric_summary(row['test_recall_at_10'])} | "
                f"{format_metric_summary(row['test_ndcg_at_20'])} | "
                f"{format_metric_summary(row['test_recall_at_20'])} |\n"
            )


if __name__ == "__main__":
    main()
