"""
scripts/rq2_report.py
=====================
RQ2: Aggregate alpha tuning results and select best alpha.

Reads from MLflow experiment 'mars_confidence_tuning', produces:
    - rq2_alpha_summary.csv
    - rq2_alpha_table.md
    - rq2_best_alpha.json

Selection: highest mean validation NDCG@10.
Tie-break: prefer smaller alpha.

Usage:
    uv run python scripts/rq2_report.py --benchmark-id rq2-alpha-tune
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

EXPERIMENT_NAME = "mars_confidence_tuning"
PRIMARY_METRIC = "best_val_ndcg_at_10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2: report alpha tuning results.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-6:
        return f"{value:.2e}"
    return f"{value:.6f}"


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
        alpha = float(tags.get("confidence_alpha", "nan"))
        seed = int(run.data.params.get("seed", "0"))
        val_ndcg = run.data.metrics.get(PRIMARY_METRIC)
        test_ndcg = run.data.metrics.get("test_NDCG_at_10")
        if val_ndcg is None:
            continue
        selected.append({"alpha": alpha, "seed": seed, "val_ndcg_at_10": val_ndcg, "test_NDCG_at_10": test_ndcg})

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    # Validate seed consistency
    by_alpha_pair: dict[float, set[int]] = {}
    pair_counts: dict[tuple[float, int], int] = {}
    for r in selected:
        by_alpha_pair.setdefault(r["alpha"], set()).add(r["seed"])
        key = (r["alpha"], r["seed"])
        pair_counts[key] = pair_counts.get(key, 0) + 1

    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    if dupes:
        raise RuntimeError(f"Duplicate (alpha, seed) runs: {dupes}")

    seed_sets = list(by_alpha_pair.values())
    if len(set(frozenset(s) for s in seed_sets)) > 1:
        details = {alpha: sorted(seeds) for alpha, seeds in by_alpha_pair.items()}
        raise RuntimeError(f"All alphas must have the same seed set. Got: {details}")

    by_alpha: dict[float, list[float]] = {}
    for r in selected:
        by_alpha.setdefault(r["alpha"], []).append(r["val_ndcg_at_10"])

    summary_rows = []
    for alpha in sorted(by_alpha.keys()):
        vals = np.array(by_alpha[alpha])
        summary_rows.append({
            "alpha": alpha, "n_seeds": len(vals),
            "val_ndcg_at_10_mean": float(vals.mean()),
            "val_ndcg_at_10_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        })

    summary_rows.sort(key=lambda r: (-r["val_ndcg_at_10_mean"], r["alpha"]))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    best_alpha = summary_rows[0]["alpha"]
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq2" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rq2_alpha_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "alpha", "n_seeds", "val_ndcg_at_10_mean", "val_ndcg_at_10_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(output_dir / "rq2_alpha_table.md", "w") as f:
        f.write("# RQ2 Confidence Alpha Tuning\n\n")
        f.write(f"Best alpha: **{best_alpha}**\n\n")
        f.write("Models ranked by mean validation NDCG@10.\n\n")
        f.write("| Rank | Alpha | Seeds | Val NDCG@10 |\n")
        f.write("| ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            f.write(f"| {row['rank']} | {row['alpha']:.2f} | {row['n_seeds']} | {row['val_ndcg_at_10_mean']:.4f} ± {row['val_ndcg_at_10_std']:.4f} |\n")

    with open(output_dir / "rq2_best_alpha.json", "w") as f:
        json.dump({"best_alpha": best_alpha, "benchmark_id": args.benchmark_id}, f, indent=2)

    print(f"Best alpha: {best_alpha}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
