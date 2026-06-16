from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import HEURISTIC_MODELS
from training.mlflow_utils import configure_mlflow, load_dataset_freeze_record


def required_run_count_for_model(model_name: str, seed_count: int) -> int:
    return 1 if model_name in HEURISTIC_MODELS else seed_count


def summarize_metric_values(values: list[float]) -> dict[str, float | int]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "runs": int(len(arr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RQ1 benchmark runs from MLflow.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--expected-neural-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    dataset_version = args.dataset_version
    if dataset_version is None:
        freeze_record = load_dataset_freeze_record(Path("data/processed/reports/dataset_freeze.json"))
        dataset_version = freeze_record["dataset_version"]

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
        if tags.get("dataset_version") != dataset_version:
            continue
        selected_runs.append(run)

    grouped: dict[str, list] = {}
    for run in selected_runs:
        grouped.setdefault(run.data.tags["model"], []).append(run)

    seed_count = args.expected_neural_runs
    for model_name, model_runs in grouped.items():
        expected = required_run_count_for_model(model_name, seed_count)
        if len(model_runs) != expected:
            raise RuntimeError(
                f"Model {model_name} has {len(model_runs)} runs but expected {expected} for benchmark {args.benchmark_id}"
            )

    output_dir = Path(args.output_dir)
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
            row = {
                "model": model_name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "best_val_ndcg_at_10": metrics.get("best_val_ndcg_at_10", 0.0),
                "best_epoch": metrics.get("best_epoch", 0.0),
                "test_NDCG_at_10": metrics.get("test_NDCG_at_10", 0.0),
                "test_Recall_at_10": metrics.get("test_Recall_at_10", 0.0),
                "test_NDCG_at_20": metrics.get("test_NDCG_at_20", 0.0),
                "test_Recall_at_20": metrics.get("test_Recall_at_20", 0.0),
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

    with open(output_dir / "rq1_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()) if run_rows else ["model"])
        writer.writeheader()
        writer.writerows(run_rows)

    with open(output_dir / "rq1_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(output_dir / "rq1_summary.csv", "w", newline="") as f:
        fieldnames = [
            "rank",
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
        for rank, row in enumerate(summary_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "model": row["model"],
                    "runs": row["runs"],
                    "val_ndcg_at_10_mean": row["val_ndcg_at_10"]["mean"],
                    "val_ndcg_at_10_std": row["val_ndcg_at_10"]["std"],
                    "test_ndcg_at_10_mean": row["test_ndcg_at_10"]["mean"],
                    "test_ndcg_at_10_std": row["test_ndcg_at_10"]["std"],
                    "test_recall_at_10_mean": row["test_recall_at_10"]["mean"],
                    "test_recall_at_10_std": row["test_recall_at_10"]["std"],
                    "test_ndcg_at_20_mean": row["test_ndcg_at_20"]["mean"],
                    "test_ndcg_at_20_std": row["test_ndcg_at_20"]["std"],
                    "test_recall_at_20_mean": row["test_recall_at_20"]["mean"],
                    "test_recall_at_20_std": row["test_recall_at_20"]["std"],
                }
            )

    with open(output_dir / "rq1_table.md", "w") as f:
        f.write("| Rank | Model | Runs | Val NDCG@10 | Test NDCG@10 | Recall@10 | NDCG@20 | Recall@20 |\n")
        f.write("| ---: | ----- | ---: | ----------: | -----------: | --------: | ------: | --------: |\n")
        for rank, row in enumerate(summary_rows, start=1):
            f.write(
                f"| {rank} | {row['model']} | {row['runs']} | "
                f"{row['val_ndcg_at_10']['mean']:.4f} ± {row['val_ndcg_at_10']['std']:.4f} | "
                f"{row['test_ndcg_at_10']['mean']:.4f} ± {row['test_ndcg_at_10']['std']:.4f} | "
                f"{row['test_recall_at_10']['mean']:.4f} ± {row['test_recall_at_10']['std']:.4f} | "
                f"{row['test_ndcg_at_20']['mean']:.4f} ± {row['test_ndcg_at_20']['std']:.4f} | "
                f"{row['test_recall_at_20']['mean']:.4f} ± {row['test_recall_at_20']['std']:.4f} |\n"
            )


if __name__ == "__main__":
    main()
