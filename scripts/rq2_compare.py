"""
scripts/rq2_compare.py
=======================
RQ2: Statistical comparison of watch variants on test set.
Runs paired t-test and Holm correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.stat_tests import apply_holm_correction, compute_seed_paired_t_test

PRIMARY_METRIC = "test_NDCG_at_10"
METRIC_LABEL = "Test NDCG@10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2: statistical comparison of watch variants.")
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    runs_file = Path(args.runs_file)
    summary_file = Path(args.summary_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(runs_file, newline="") as f:
        run_rows = list(csv.DictReader(f))

    summary = json.loads(summary_file.read_text())
    winner = summary[0]["variant"]

    by_key: dict[tuple[str, str], float] = {}
    for row in run_rows:
        variant = row["variant"]
        seed = row["seed"]
        val = float(row.get(PRIMARY_METRIC, 0))
        by_key[(variant, seed)] = val

    all_seeds = sorted({int(row["seed"]) for row in run_rows})

    baselines = [v for v in ["baseline", "wl", "we", "wlwe"] if v != winner]

    results = []
    for baseline in baselines:
        winner_vals = np.array([by_key.get((winner, str(s)), 0.0) for s in all_seeds])
        baseline_vals = np.array([by_key.get((baseline, str(s)), 0.0) for s in all_seeds])

        stats = compute_seed_paired_t_test(winner_vals, baseline_vals)
        winner_mean = float(winner_vals.mean())
        baseline_mean = float(baseline_vals.mean())

        results.append({
            "comparison": f"{winner} vs {baseline}",
            "comparison_type": "seed_paired_t_test",
            "winner": winner,
            "baseline": baseline,
            "winner_mean": winner_mean,
            "baseline_mean": baseline_mean,
            "relative_improvement": (winner_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else None,
            **stats,
        })

    apply_holm_correction(results, p_key="raw_p_value")

    all_fields = [
        "comparison", "comparison_type", "winner", "baseline",
        "winner_mean", "baseline_mean", "mean_difference", "relative_improvement",
        "std_difference", "ci95_low", "ci95_high",
        "t_statistic", "raw_p_value", "holm_adjusted_p_value",
        "significant_after_holm", "wins", "ties", "losses",
    ]
    with open(output_dir / "rq2_statistical_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(results)

    sig = [r for r in results if r["significant_after_holm"]]
    print(f"Winner: {winner}, comparisons: {len(results)}, Holm-significant: {len(sig)}")


if __name__ == "__main__":
    main()
