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

All selected runs are required to share the same:
    - backbone (model)
    - benchmark_id
    - preprocessing_version
    - data_source
    - git_commit
Missing or mismatched provenance fails the report — there is no silent
fallback to "mars-preprocess-v1" or "data/processed".

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

REQUIRED_PROVENANCE_FIELDS = (
    "data_source",
    "preprocessing_version",
    "git_commit",
    "benchmark_id",
    "backbone",
)


def _validate_rq2_grid(selected: list[dict]) -> None:
    """Validate that every (alpha, seed) appears exactly once and seeds are
    consistent across alphas. The expected grid is derived from the runs,
    not hardcoded — a custom sweep is valid as long as it is complete.
    """
    if not selected:
        raise RuntimeError("No runs to validate")
    actual_alpha_seeds = {(float(r["alpha"]), int(r["seed"])) for r in selected}
    if len(actual_alpha_seeds) != len(selected):
        pair_counts: dict[tuple[float, int], int] = {}
        for r in selected:
            key = (float(r["alpha"]), int(r["seed"]))
            pair_counts[key] = pair_counts.get(key, 0) + 1
        dupes = {k: v for k, v in pair_counts.items() if v > 1}
        raise RuntimeError(f"Duplicate (alpha, seed) runs: {dupes}")

    by_alpha: dict[float, set[int]] = {}
    for r in selected:
        by_alpha.setdefault(float(r["alpha"]), set()).add(int(r["seed"]))
    seed_sets = list(by_alpha.values())
    if len(set(frozenset(s) for s in seed_sets)) > 1:
        details = {alpha: sorted(seeds) for alpha, seeds in by_alpha.items()}
        raise RuntimeError(
            f"All alphas must have the same seed set. Got: {details}"
        )


def _parse_seed(run, run_id: str) -> int:
    """Strict seed parsing — never silently coerce to 0.

    Raises RuntimeError naming the run that has a malformed seed.
    """
    raw = run.data.params.get("seed")
    if raw is None:
        raise RuntimeError(
            f"Run {run_id} ({run.info.run_name}) has no 'seed' MLflow param. "
            "Every RQ2 run must log seed as an int param."
        )
    try:
        return int(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Run {run_id} ({run.info.run_name}) has malformed seed={raw!r}; "
            f"could not parse as int: {exc}. Fix the run or remove it from "
            "the experiment before re-running rq2-report."
        ) from exc


def _validate_provenance(selected: list[dict]) -> dict:
    """All selected runs must share the same provenance.

    Required fields: backbone, benchmark_id, preprocessing_version,
    data_source, git_commit. Missing any field on any run, or any
    disagreement across runs, fails the report.
    """
    expected: dict[str, str] = {}
    for r in selected:
        for key in REQUIRED_PROVENANCE_FIELDS:
            value = r.get("provenance", {}).get(key)
            if value is None or value == "":
                raise RuntimeError(
                    f"{r['run_id']} ({r['run_name']}): missing provenance "
                    f"field {key!r}. All RQ2 runs must carry: "
                    f"{REQUIRED_PROVENANCE_FIELDS}. Refusing to write a "
                    "winner artifact with silent defaults."
                )
            if key in expected and expected[key] != value:
                raise RuntimeError(
                    f"Provenance mismatch for {key!r}: "
                    f"expected {expected[key]!r} (from first run), got {value!r} "
                    f"on run {r['run_id']} ({r['run_name']}). All RQ2 runs "
                    "must share the same provenance."
                )
            expected[key] = value
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2: report alpha tuning results.")
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
        alpha_str = tags.get("confidence_alpha")
        if alpha_str is None:
            continue
        try:
            alpha = float(alpha_str)
        except (ValueError, TypeError):
            continue
        try:
            seed = _parse_seed(run, run.info.run_id)
        except RuntimeError:
            raise
        val_ndcg = run.data.metrics.get(PRIMARY_METRIC)
        if val_ndcg is None:
            continue
        provenance = {
            "backbone": tags.get("model"),
            "benchmark_id": tags.get("benchmark_id"),
            "preprocessing_version": tags.get("preprocessing_version"),
            "data_source": tags.get("data_source"),
            "git_commit": tags.get("git_commit"),
        }
        selected.append({
            "alpha": alpha,
            "seed": seed,
            "val_ndcg_at_10": val_ndcg,
            "provenance": provenance,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
        })

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")
    _validate_rq2_grid(selected)
    provenance = _validate_provenance(selected)
    if provenance["backbone"] != "gsasrec":
        raise RuntimeError(
            f"RQ2 is gSASRec-only, but selected runs report backbone={provenance['backbone']!r}"
        )

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
            f.write(f"| {row['rank']} | {row['alpha']:.6f} | {row['n_seeds']} | {row['val_ndcg_at_10_mean']:.4f} ± {row['val_ndcg_at_10_std']:.4f} |\n")

    observed_alphas = sorted({float(r["alpha"]) for r in selected})
    observed_seeds = sorted({int(r["seed"]) for r in selected})

    with open(output_dir / "rq2_best_alpha.json", "w") as f:
        winner = {
            "best_alpha": best_alpha,
            "benchmark_id": args.benchmark_id,
            "backbone": "gsasrec",
            "candidate_grid": observed_alphas,
            "seeds": observed_seeds,
            "selection_metric": PRIMARY_METRIC,
            "preprocessing_version": provenance["preprocessing_version"],
            "data_source": provenance["data_source"],
            "git_commit": provenance["git_commit"],
        }
        json.dump(winner, f, indent=2)

    print(f"Best alpha: {best_alpha}")
    print(f"Backbone: {provenance['backbone']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
