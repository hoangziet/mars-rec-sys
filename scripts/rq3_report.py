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

EXPECTED_VARIANTS = ["M0", "M1", "M2", "M3"]
EXPECTED_SEEDS = [42, 123, 2024]


def _validate_rq3_grid(selected: list[dict]) -> None:
    actual_variants = sorted({str(r["variant"]) for r in selected})
    if actual_variants != EXPECTED_VARIANTS:
        raise RuntimeError(f"Expected variants {EXPECTED_VARIANTS}, got {actual_variants}")

    actual_pairs = {(str(r["variant"]), int(r["seed"])) for r in selected}
    expected_pairs = {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(f"RQ3 grid mismatch. Missing={missing}, Extra={extra}")


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
    _validate_rq3_grid(selected)

    # Validate seed consistency
    by_variant_pair: dict[str, set[int]] = {}
    pair_counts: dict[tuple[str, int], int] = {}
    for r in selected:
        by_variant_pair.setdefault(r["variant"], set()).add(r["seed"])
        key = (r["variant"], r["seed"])
        pair_counts[key] = pair_counts.get(key, 0) + 1

    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    if dupes:
        raise RuntimeError(f"Duplicate (variant, seed) runs: {dupes}")

    seed_sets = list(by_variant_pair.values())
    if len(set(frozenset(s) for s in seed_sets)) > 1:
        details = {v: sorted(s) for v, s in by_variant_pair.items()}
        raise RuntimeError(f"All variants must have the same seed set. Got: {details}")

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
