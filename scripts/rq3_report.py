"""
scripts/rq3_report.py
=====================
RQ3: Aggregate metadata tuning results and select best variant.

Reads from MLflow experiment 'mars_metadata_tuning'.

Selection: highest mean validation NDCG@10.
Tie-break: prefer simpler config (M0 > M1 > M2 > M3).

Usage:
    uv run python scripts/rq3_report.py --benchmark-id rq3-metadata-tune
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

EXPERIMENT_NAME = "mars_metadata_tuning"
PRIMARY_METRIC = "best_val_ndcg_at_10"
VARIANT_ORDER = {"M0": 0, "M1": 1, "M2": 2, "M3": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ3: report metadata tuning results.")
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
        variant = tags.get("metadata_variant", "?")
        seed = int(run.data.params.get("seed", "0"))
        val_ndcg = run.data.metrics.get(PRIMARY_METRIC)
        test_ndcg = run.data.metrics.get("test_NDCG_at_10")
        if val_ndcg is None:
            continue
        selected.append({"variant": variant, "seed": seed, "val_ndcg_at_10": val_ndcg, "test_NDCG_at_10": test_ndcg})

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    by_variant: dict[str, list[float]] = {}
    for r in selected:
        by_variant.setdefault(r["variant"], []).append(r["val_ndcg_at_10"])

    summary_rows = []
    for variant in sorted(by_variant.keys(), key=lambda v: VARIANT_ORDER.get(v, 99)):
        vals = np.array(by_variant[variant])
        summary_rows.append({"variant": variant, "n_seeds": len(vals), "val_ndcg_at_10_mean": float(vals.mean()), "val_ndcg_at_10_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0})

    summary_rows.sort(key=lambda r: (-r["val_ndcg_at_10_mean"], VARIANT_ORDER.get(r["variant"], 99)))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    best_variant = summary_rows[0]["variant"]
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq3" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rq3_variant_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "variant", "n_seeds", "val_ndcg_at_10_mean", "val_ndcg_at_10_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(output_dir / "rq3_variant_table.md", "w") as f:
        f.write("# RQ3 Metadata Variant Tuning\n\n")
        f.write(f"Best variant: **{best_variant}**\n\n")
        f.write("Variants ranked by mean validation NDCG@10.\n\n")
        f.write("| Rank | Variant | Seeds | Val NDCG@10 |\n")
        f.write("| ---: | --- | ---: | ---: |\n")
        for row in summary_rows:
            f.write(f"| {row['rank']} | {row['variant']} | {row['n_seeds']} | {row['val_ndcg_at_10_mean']:.4f} ± {row['val_ndcg_at_10_std']:.4f} |\n")

    with open(output_dir / "rq3_best_variant.json", "w") as f:
        json.dump({"best_variant": best_variant, "benchmark_id": args.benchmark_id}, f, indent=2)

    print(f"Best variant: {best_variant}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
