"""
scripts/rq4_collect.py
======================
RQ4: Collect ablation results from MLflow into local CSV/JSON.

Produces:
    rq4_runs.csv      — one row per (variant, seed)
    rq4_summary.json  — per-variant summary with validation_rank
    rq4_manifest.json — variant/seed metadata for rq4_compare

Usage:
    uv run python scripts/rq4_collect.py --benchmark-id rq4-ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_utils import configure_mlflow

EXPERIMENT_NAME = "mars_final_ablation"
VARIANT_ORDER = {"V0": 0, "V1": 1, "V2": 2, "V3": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: collect ablation results from MLflow.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' does not exist")

    runs = client.search_runs([experiment.experiment_id])
    selected = []
    for run in runs:
        tags = run.data.tags
        if run.info.status != "FINISHED":
            continue
        if tags.get("reportable") != "true":
            continue
        if tags.get("benchmark_id") != args.benchmark_id:
            continue
        variant = tags.get("ablation_variant", "?")
        seed = int(run.data.params.get("seed", "0"))
        val_ndcg = run.data.metrics.get("best_val_ndcg_at_10")
        test_ndcg = run.data.metrics.get("test_NDCG_at_10")
        test_recall = run.data.metrics.get("test_Recall_at_10")
        if val_ndcg is None or test_ndcg is None:
            continue
        selected.append({
            "variant": variant,
            "seed": seed,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "best_val_ndcg_at_10": val_ndcg,
            "test_NDCG_at_10": test_ndcg,
            "test_Recall_at_10": test_recall,
        })

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq4" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write rq4_runs.csv
    run_fields = ["variant", "seed", "run_id", "run_name", "best_val_ndcg_at_10", "test_NDCG_at_10", "test_Recall_at_10"]
    with open(output_dir / "rq4_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(selected)

    # Group by variant for summary
    by_variant: dict[str, list[dict]] = {}
    for r in selected:
        by_variant.setdefault(r["variant"], []).append(r)

    # Build summary with validation_rank
    summary_rows = []
    for variant in sorted(by_variant.keys(), key=lambda v: VARIANT_ORDER.get(v, 99)):
        runs_list = by_variant[variant]
        val_vals = [r["best_val_ndcg_at_10"] for r in runs_list]
        test_vals = [r["test_NDCG_at_10"] for r in runs_list]
        seeds = sorted({r["seed"] for r in runs_list})
        summary_rows.append({
            "model": variant,
            "variant": variant,
            "runs": len(runs_list),
            "seeds": seeds,
            "val_ndcg_at_10": {
                "mean": float(np.mean(val_vals)),
                "std": float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0,
                "ci95_low": None,
                "ci95_high": None,
            },
            "test_ndcg_at_10": {
                "mean": float(np.mean(test_vals)),
                "std": float(np.std(test_vals, ddof=1)) if len(test_vals) > 1 else 0.0,
            },
        })

    # Sort by val mean descending, assign validation_rank
    summary_rows.sort(key=lambda r: (-r["val_ndcg_at_10"]["mean"], VARIANT_ORDER.get(r["variant"], 99)))
    for rank, row in enumerate(summary_rows, start=1):
        row["validation_rank"] = rank

    with open(output_dir / "rq4_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Build manifest for rq4_compare
    all_seeds = sorted({r["seed"] for r in selected})
    all_variants = sorted(by_variant.keys(), key=lambda v: VARIANT_ORDER.get(v, 99))
    manifest = {
        "benchmark_id": args.benchmark_id,
        "variants": all_variants,
        "neural_seeds": all_seeds,
        "n_seeds": len(all_seeds),
        "n_variants": len(all_variants),
    }
    with open(output_dir / "rq4_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Collected {len(selected)} runs across {len(by_variant)} variants")
    print(f"Variants: {all_variants}")
    print(f"Seeds: {all_seeds}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
