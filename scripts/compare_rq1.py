from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare top-two RQ1 models with paired statistical tests.")
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-pairs", type=int, default=5)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def select_top_two_models(summary_rows: list[dict]) -> tuple[str, str]:
    if len(summary_rows) < 2:
        raise RuntimeError(f"Need at least 2 ranked models, got {len(summary_rows)}")
    ranked = sorted(summary_rows, key=lambda row: row["validation_rank"])
    return ranked[0]["model"], ranked[1]["model"]


def _index_by_metric(rows: list[dict], model: str, metric: str) -> dict[int, float]:
    by_seed: dict[int, float] = {}
    for row in rows:
        if row["model"] != model:
            continue
        seed = int(row["seed"])
        if seed in by_seed:
            raise RuntimeError(f"Duplicate seed {seed} for model {model}")
        by_seed[seed] = float(row[metric])
    return by_seed


def pair_runs_by_seed(rows: list[dict], winner: str, runner_up: str, expected_pairs: int) -> list[tuple[int, float, float]]:
    metric = "test_NDCG_at_10"
    winner_by_seed = _index_by_metric(rows, winner, metric)
    runner_by_seed = _index_by_metric(rows, runner_up, metric)
    if set(winner_by_seed) != set(runner_by_seed):
        raise RuntimeError(
            f"Seed mismatch: winner={sorted(winner_by_seed)}, runner_up={sorted(runner_by_seed)}"
        )
    common_seeds = sorted(set(winner_by_seed) & set(runner_by_seed))
    if len(common_seeds) != expected_pairs:
        raise RuntimeError(
            f"Expected exactly {expected_pairs} paired seeds, got {len(common_seeds)}"
        )
    return [(seed, winner_by_seed[seed], runner_by_seed[seed]) for seed in common_seeds]


def relative_improvement(winner_mean: float, runner_up_mean: float) -> float:
    if runner_up_mean == 0:
        raise ValueError("Runner-up mean must be non-zero")
    return (winner_mean - runner_up_mean) / runner_up_mean


def _summarize_paired_differences(pairs: list[tuple[int, float, float]]) -> dict[str, float]:
    winner_arr = np.array([w for _, w, _ in pairs])
    runner_arr = np.array([r for _, _, r in pairs])
    if np.all(winner_arr == runner_arr):
        return {
            "t_stat": 0.0,
            "t_p": 1.0,
            "w_stat": 0.0,
            "w_p": 1.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    diffs = winner_arr - runner_arr
    t_result = scipy.stats.ttest_rel(winner_arr, runner_arr)
    w_result = scipy.stats.wilcoxon(winner_arr, runner_arr, alternative="greater")
    diff_stats = DescrStatsW(diffs)
    ci_low, ci_high = diff_stats.tconfint_mean(alpha=0.05)
    return {
        "t_stat": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "w_stat": float(w_result.statistic),
        "w_p": float(w_result.pvalue),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def main() -> None:
    args = parse_args()

    summary_rows = json.loads(Path(args.summary_file).read_text())
    winner, runner_up = select_top_two_models(summary_rows)

    with open(args.runs_file, newline="") as f:
        run_rows = list(csv.DictReader(f))

    pairs = pair_runs_by_seed(run_rows, winner, runner_up, args.expected_pairs)
    winner_arr = np.array([w for _, w, _ in pairs])
    runner_arr = np.array([r for _, _, r in pairs])
    winner_mean = float(winner_arr.mean())
    runner_mean = float(runner_arr.mean())
    improvement = relative_improvement(winner_mean, runner_mean)
    stats = _summarize_paired_differences(pairs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rq1_pairwise.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "test_NDCG_at_10", "paired_metric"])
        for seed, w, r in pairs:
            writer.writerow([winner, seed, f"{w:.6f}", "winner"])
            writer.writerow([runner_up, seed, f"{r:.6f}", "runner_up"])

    with open(output_dir / "rq1_significance.md", "w") as f:
        f.write(f"# RQ1 Statistical Comparison: {winner} vs {runner_up}\n\n")
        f.write(f"Paired seeds: {len(pairs)}\n\n")
        f.write(f"Winner ({winner}) mean test NDCG@10: {winner_mean:.4f}\n")
        f.write(f"Runner-up ({runner_up}) mean test NDCG@10: {runner_mean:.4f}\n")
        f.write(f"Relative improvement: {improvement:.4f} ({improvement * 100:.2f}%)\n\n")
        f.write("## 95% CI for paired differences\n\n")
        f.write(f"[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]\n\n")
        f.write("## Statistical tests\n\n")
        f.write(f"- Paired t-test (two-sided) p-value: {stats['t_p']:.6f} (t = {stats['t_stat']:.4f})\n")
        f.write(f"- Wilcoxon signed-rank test (one-sided, winner > runner-up) p-value: {stats['w_p']:.6f} (W = {stats['w_stat']:.4f})\n")


if __name__ == "__main__":
    main()
