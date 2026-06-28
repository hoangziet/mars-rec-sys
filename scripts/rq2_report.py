"""
scripts/rq2_report.py
=====================
RQ2: Aggregate watch variant results and select best variant.
Reads from MLflow experiment 'mars_watch_variant_comparison'.
Produces: rq2_summary.csv, rq2_summary.json, rq2_runs.csv,
          rq2_watch_table.md, rq2_best_watch.json

Selection: highest mean validation NDCG@10.
Tie-break: prefer simpler config (baseline > wl > we > wlwe).
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

from scripts.study_manifest import load_manifest, manifest_path_for_output_dir
from training.mlflow_contract import RQ2_VARIANT_EXPERIMENT_NAME
from training.mlflow_utils import configure_mlflow

EXPERIMENT_NAME = RQ2_VARIANT_EXPERIMENT_NAME
PRIMARY_METRIC = "best_val_ndcg_at_10"
VARIANT_ORDER = {"baseline": 0, "wl": 1, "we": 2, "wlwe": 3}


def write_final_report(
    output_dir: Path,
    *,
    benchmark_id: str,
    best_variant: str,
    best_alpha: float | None,
    summary_rows: list[dict],
) -> None:
    with open(output_dir / "rq2_final_report.md", "w") as f:
        f.write("# RQ2 Final Report\n\n")
        f.write(f"Benchmark: {benchmark_id}\n\n")
        f.write(f"Best variant: **{best_variant}**\n\n")
        f.write(f"Best alpha: **{best_alpha}**\n\n")
        f.write("Variants ranked by mean validation NDCG@10.\n\n")
        f.write("| Rank | Variant | Seeds | Val NDCG@10 |\n")
        f.write("| ---: | --- | ---: | ---: |\n")
        for row in summary_rows:
            f.write(
                f"| {row['rank']} | {row['variant']} | {row['n_seeds']} | "
                f"{row['val_ndcg_at_10_mean']:.4f} ± {row['val_ndcg_at_10_std']:.4f} |\n"
            )


def write_outputs(selected_runs: list[dict], alpha_artifact: dict, output_dir: Path, benchmark_id: str) -> str:
    """Write summary CSV, JSON, runs CSV, markdown, and best-watch JSON artifact.

    Returns the best variant name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_variant: dict[str, list[float]] = {}
    by_variant_test: dict[str, list[float]] = {}
    for row in selected_runs:
        by_variant.setdefault(row["variant"], []).append(row["val_ndcg_at_10"])
        if "test_NDCG_at_10" in row:
            by_variant_test.setdefault(row["variant"], []).append(row["test_NDCG_at_10"])

    summary_rows = []
    for variant, values in by_variant.items():
        vals = np.asarray(values, dtype=float)
        row = {
            "variant": variant,
            "n_seeds": len(vals),
            "val_ndcg_at_10_mean": float(vals.mean()),
            "val_ndcg_at_10_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
        if variant in by_variant_test:
            tv = np.asarray(by_variant_test[variant], dtype=float)
            row["test_NDCG_at_10_mean"] = float(tv.mean())
            row["test_NDCG_at_10_std"] = float(tv.std(ddof=1)) if len(tv) > 1 else 0.0
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: (-r["val_ndcg_at_10_mean"], VARIANT_ORDER[r["variant"]]))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    best_variant = summary_rows[0]["variant"]

    # Write CSV
    fields = ["rank", "variant", "n_seeds", "val_ndcg_at_10_mean", "val_ndcg_at_10_std"]
    if "test_NDCG_at_10_mean" in summary_rows[0]:
        fields += ["test_NDCG_at_10_mean", "test_NDCG_at_10_std"]
    with open(output_dir / "rq2_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write JSON summary (list of dicts, for compare stage)
    with open(output_dir / "rq2_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Write runs CSV (for compare stage)
    run_fields = ["variant", "seed", "val_ndcg_at_10"]
    if any("test_NDCG_at_10" in r for r in selected_runs):
        run_fields.append("test_NDCG_at_10")
    with open(output_dir / "rq2_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_runs)

    # Write markdown
    with open(output_dir / "rq2_watch_table.md", "w") as f:
        f.write("# RQ2 Watch Integration Comparison\n\n")
        f.write(f"Best variant: **{best_variant}** | Best alpha: **{alpha_artifact['best_alpha']}**\n\n")
        f.write("| Rank | Variant | Seeds | Val NDCG@10 |\n")
        f.write("| ---: | --- | ---: | ---: |\n")
        for row in summary_rows:
            f.write(f"| {row['rank']} | {row['variant']} | {row['n_seeds']} | {row['val_ndcg_at_10_mean']:.4f} ± {row['val_ndcg_at_10_std']:.4f} |\n")

    write_final_report(
        output_dir,
        benchmark_id=benchmark_id,
        best_variant=best_variant,
        best_alpha=alpha_artifact.get("best_alpha"),
        summary_rows=summary_rows,
    )

    # Write winner artifact
    observed_variants = sorted({r["variant"] for r in selected_runs})
    observed_seeds = sorted({r["seed"] for r in selected_runs})
    winner = {
        "benchmark_id": benchmark_id,
        "backbone": "bert4rec",
        "best_variant": best_variant,
        "best_alpha": alpha_artifact.get("best_alpha"),
        "candidate_grid": observed_variants,
        "seeds": observed_seeds,
        "selection_metric": PRIMARY_METRIC,
        "preprocessing_version": selected_runs[0].get("preprocessing_version", "unknown"),
        "data_source": selected_runs[0].get("data_source", "unknown"),
    }
    with open(output_dir / "rq2_best_watch.json", "w") as f:
        json.dump(winner, f, indent=2)

    return best_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2: report watch variant results.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--alpha-artifact", required=True, help="Path to rq2_best_alpha.json from Stage A")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/rq2/{args.benchmark_id}")
    load_manifest(manifest_path_for_output_dir(output_dir), require_completed=True)

    configure_mlflow(mlflow_module=mlflow)
    alpha_artifact = json.loads(Path(args.alpha_artifact).read_text())

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
        variant = tags.get("variant", "?")
        try:
            seed = int(run.data.params.get("seed", "0"))
        except (ValueError, TypeError):
            continue
        val_ndcg = run.data.metrics.get(PRIMARY_METRIC)
        test_ndcg = run.data.metrics.get("test_NDCG_at_10")
        if val_ndcg is None:
            continue
        row = {
            "variant": variant,
            "seed": seed,
            "val_ndcg_at_10": val_ndcg,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "preprocessing_version": tags.get("preprocessing_version", "unknown"),
            "data_source": tags.get("data_source", "unknown"),
        }
        if test_ndcg is not None:
            row["test_NDCG_at_10"] = test_ndcg
        selected.append(row)

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    # Validate grid: no duplicate (variant, seed) pairs, consistent seed sets
    pair_counts: dict[tuple[str, int], int] = {}
    for row in selected:
        key = (row["variant"], row["seed"])
        pair_counts[key] = pair_counts.get(key, 0) + 1
    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    if dupes:
        raise RuntimeError(
            f"Duplicate (variant, seed) runs found: {dupes}. "
            "Delete stale runs from MLflow before re-running the report."
        )
    by_variant_seeds: dict[str, set[int]] = {}
    for row in selected:
        by_variant_seeds.setdefault(row["variant"], set()).add(row["seed"])
    seed_sets = list(by_variant_seeds.values())
    if len(set(frozenset(s) for s in seed_sets)) > 1:
        raise RuntimeError(
            f"All variants must share the same seed set. Got: "
            f"{ {v: sorted(s) for v, s in by_variant_seeds.items()} }"
        )

    best_variant = write_outputs(selected, alpha_artifact, output_dir, args.benchmark_id)
    print(f"Best variant: {best_variant}")
    print(f"Best alpha: {alpha_artifact['best_alpha']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
