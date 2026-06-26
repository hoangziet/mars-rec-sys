"""
scripts/rq2_alpha_report.py
============================
RQ2 Stage A report: aggregate alpha tuning results and select best alpha.
Reads from MLflow experiment 'mars_watch_alpha_tuning'.
Selection: highest mean validation NDCG@10. Tie-break: prefer smaller alpha.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import RQ2_ALPHA_EXPERIMENT_NAME
from training.mlflow_utils import configure_mlflow

EXPERIMENT_NAME = RQ2_ALPHA_EXPERIMENT_NAME
PRIMARY_METRIC = "best_val_ndcg_at_10"


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2: report best alpha from tuning.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    configure_mlflow(mlflow_module=mlflow)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' does not exist")

    runs = client.search_runs([experiment.experiment_id])
    selected = []
    provenance = {}
    for run in runs:
        tags = run.data.tags
        if run.info.status != "FINISHED":
            continue
        if tags.get("reportable") != "true":
            continue
        if tags.get("benchmark_id") != args.benchmark_id:
            continue
        alpha_str = tags.get("watch_alpha")
        if alpha_str is None:
            continue
        try:
            alpha = float(alpha_str)
        except (ValueError, TypeError):
            continue
        try:
            seed = int(run.data.params.get("seed", "0"))
        except (ValueError, TypeError):
            continue
        val_ndcg = run.data.metrics.get(PRIMARY_METRIC)
        if val_ndcg is None:
            continue
        selected.append({
            "alpha": alpha,
            "seed": seed,
            "val_ndcg_at_10": val_ndcg,
        })
        if not provenance:
            provenance = {
                "preprocessing_version": tags.get("preprocessing_version", "mars-preprocess-v1"),
                "data_source": tags.get("data_source", "data/processed"),
            }

    if not selected:
        raise RuntimeError(f"No reportable alpha runs found for benchmark {args.benchmark_id}")

    by_alpha: dict[float, list[float]] = {}
    for row in selected:
        by_alpha.setdefault(row["alpha"], []).append(row["val_ndcg_at_10"])

    best_alpha = None
    best_mean = -float("inf")
    for alpha in sorted(by_alpha.keys()):
        vals = np.array(by_alpha[alpha])
        mean_val = float(vals.mean())
        if mean_val > best_mean or (mean_val == best_mean and (best_alpha is None or alpha < best_alpha)):
            best_mean = mean_val
            best_alpha = alpha

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq2" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    winner = {
        "best_alpha": best_alpha,
        "benchmark_id": args.benchmark_id,
        "backbone": "bert4rec",
        "selection_metric": PRIMARY_METRIC,
        "preprocessing_version": provenance.get("preprocessing_version", "mars-preprocess-v1"),
        "data_source": provenance.get("data_source", "data/processed"),
    }
    (output_dir / "rq2_best_alpha.json").write_text(json.dumps(winner, indent=2) + "\n")
    print(f"Best alpha: {best_alpha} (mean val NDCG@10 = {best_mean:.4f})")
    print(f"Output: {output_dir / 'rq2_best_alpha.json'}")


if __name__ == "__main__":
    main()
