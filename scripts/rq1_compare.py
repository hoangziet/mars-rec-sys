from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_contract import HEURISTIC_MODELS
from training.stat_tests import apply_holm_correction, compute_seed_paired_t_test


PRIMARY_METRIC = "test_NDCG_at_10"
METRIC_LABEL = "Test NDCG@10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Winner-versus-baseline RQ1 comparison."
    )
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def select_winner_and_baselines(
    summary_rows: list[dict],
) -> tuple[str, list[str]]:
    if len(summary_rows) < 2:
        raise RuntimeError(
            f"At least two models are required, got {len(summary_rows)}"
        )
    ranked = sorted(summary_rows, key=lambda row: int(row["validation_rank"]))
    winner = str(ranked[0]["model"])
    baselines = [str(row["model"]) for row in ranked[1:]]
    return winner, baselines


def _index_by_seed(
    rows: list[dict], model: str, metric: str
) -> dict[int, float]:
    by_seed: dict[int, float] = {}
    for row in rows:
        if row["model"] != model:
            continue
        seed = int(row["seed"])
        if seed in by_seed:
            raise RuntimeError(f"Duplicate seed {seed} for model {model}")
        by_seed[seed] = float(row[metric])
    return by_seed


def pair_models_by_seed(
    rows: list[dict],
    model_a: str,
    model_b: str,
    metric: str,
    expected_seeds: set[int],
) -> list[tuple[int, float, float]]:
    a_by_seed = _index_by_seed(rows, model_a, metric)
    b_by_seed = _index_by_seed(rows, model_b, metric)

    if set(a_by_seed) != expected_seeds:
        raise RuntimeError(
            f"{model_a}: expected seeds {sorted(expected_seeds)}, "
            f"got {sorted(a_by_seed)}"
        )
    if set(b_by_seed) != expected_seeds:
        raise RuntimeError(
            f"{model_b}: expected seeds {sorted(expected_seeds)}, "
            f"got {sorted(b_by_seed)}"
        )

    return [
        (seed, a_by_seed[seed], b_by_seed[seed])
        for seed in sorted(expected_seeds)
    ]


def relative_improvement(winner_mean: float, baseline_mean: float) -> float | None:
    if baseline_mean == 0:
        return None
    return (winner_mean - baseline_mean) / baseline_mean


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-6:
        return f"{value:.2e}"
    return f"{value:.6f}"


def _compute_paired_stats(
    winner_values: np.ndarray,
    baseline_values: np.ndarray,
) -> dict:
    return compute_seed_paired_t_test(winner_values, baseline_values)


def _run_comparisons(
    run_rows: list[dict],
    winner: str,
    baselines: list[str],
    neural_seeds: set[int],
) -> tuple[list[dict], list[dict]]:
    if winner in HEURISTIC_MODELS:
        raise RuntimeError(
            f"Winner '{winner}' is a deterministic model. "
            "Winner-versus-neural seed-paired comparison requires "
            "a neural winner."
        )

    neural_baselines = [m for m in baselines if m not in HEURISTIC_MODELS]
    heuristic_baselines = [m for m in baselines if m in HEURISTIC_MODELS]

    neural_results: list[dict] = []
    heuristic_results: list[dict] = []
    all_seed_pairs: list[dict] = []

    winner_by_seed = _index_by_seed(run_rows, winner, PRIMARY_METRIC)
    winner_values = np.array(
        [winner_by_seed[seed] for seed in sorted(neural_seeds)]
    )
    winner_mean = float(winner_values.mean())

    for baseline in neural_baselines:
        pairs = pair_models_by_seed(
            run_rows, winner, baseline, PRIMARY_METRIC, neural_seeds
        )
        baseline_values = np.array([b for _, _, b in pairs])
        baseline_mean = float(baseline_values.mean())
        stats = _compute_paired_stats(winner_values, baseline_values)

        result = {
            "winner_model": winner,
            "baseline_model": baseline,
            "comparison_type": "seed_paired_t_test",
            "metric": METRIC_LABEL,
            "winner_mean": winner_mean,
            "baseline_mean": baseline_mean,
            "mean_difference": stats["mean_difference"],
            "relative_improvement": relative_improvement(
                winner_mean, baseline_mean
            ),
            "n_seed_pairs": len(pairs),
            "wins": stats["wins"],
            "ties": stats["ties"],
            "losses": stats["losses"],
            "std_difference": stats["std_difference"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "t_statistic": stats["t_statistic"],
            "raw_p_value": stats["raw_p_value"],
            "holm_adjusted_p_value": None,
            "significant_after_holm": None,
            "note": None,
        }
        neural_results.append(result)

        for seed, w_val, b_val in pairs:
            all_seed_pairs.append(
                {
                    "seed": seed,
                    "winner_model": winner,
                    "winner_value": w_val,
                    "baseline_model": baseline,
                    "baseline_value": b_val,
                    "difference": w_val - b_val,
                }
            )

    if neural_results:
        apply_holm_correction(neural_results, p_key="raw_p_value")

    for baseline in heuristic_baselines:
        baseline_rows = [r for r in run_rows if r["model"] == baseline]
        if len(baseline_rows) != 1:
            raise RuntimeError(
                f"{baseline}: expected exactly one deterministic run, "
                f"got {len(baseline_rows)}"
            )
        baseline_val = float(baseline_rows[0][PRIMARY_METRIC])
        diff = winner_mean - baseline_val
        improvement = relative_improvement(winner_mean, baseline_val)

        heuristic_results.append(
            {
                "winner_model": winner,
                "baseline_model": baseline,
                "comparison_type": "descriptive",
                "metric": METRIC_LABEL,
                "winner_mean": winner_mean,
                "baseline_mean": baseline_val,
                "mean_difference": diff,
                "relative_improvement": improvement,
                "n_seed_pairs": None,
                "wins": None,
                "ties": None,
                "losses": None,
                "std_difference": None,
                "ci95_low": None,
                "ci95_high": None,
                "t_statistic": None,
                "raw_p_value": None,
                "holm_adjusted_p_value": None,
                "significant_after_holm": None,
                "note": "Deterministic baseline with one run",
            }
        )

    return neural_results + heuristic_results, all_seed_pairs


def _write_outputs(
    output_dir: Path,
    winner: str,
    summary_rows: list[dict],
    seed_pairs: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_file = output_dir / "rq1_pairwise.csv"
    legacy_file.unlink(missing_ok=True)

    all_fields = [
        "winner_model",
        "baseline_model",
        "comparison_type",
        "metric",
        "winner_mean",
        "baseline_mean",
        "mean_difference",
        "relative_improvement",
        "n_seed_pairs",
        "wins",
        "ties",
        "losses",
        "std_difference",
        "ci95_low",
        "ci95_high",
        "t_statistic",
        "raw_p_value",
        "holm_adjusted_p_value",
        "significant_after_holm",
        "note",
    ]

    with open(output_dir / "rq1_winner_vs_all.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if seed_pairs:
        seed_fields = [
            "seed",
            "winner_model",
            "winner_value",
            "baseline_model",
            "baseline_value",
            "difference",
        ]
        with open(output_dir / "rq1_seed_pairs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=seed_fields)
            writer.writeheader()
            writer.writerows(seed_pairs)

    with open(output_dir / "rq1_significance.md", "w") as f:
        f.write("# RQ1 Winner-versus-Baseline Comparison\n\n")
        f.write(
            f"Winner selected by mean validation NDCG@10: **{winner}**\n\n"
        )
        f.write(f"Primary metric: {METRIC_LABEL}\n")
        f.write("Statistical test: Two-sided paired t-test\n")
        f.write("Multiple-comparison correction: Holm\n")
        f.write("Family-wise significance level: α = 0.05\n")
        f.write(
            "Confidence intervals are unadjusted per-comparison intervals; "
            "family-wise control is applied to hypothesis-test p-values "
            "through Holm correction.\n\n"
        )

        neural_rows = [
            r for r in summary_rows if r["comparison_type"] != "descriptive"
        ]
        heuristic_rows = [
            r for r in summary_rows if r["comparison_type"] == "descriptive"
        ]

        if neural_rows:
            f.write("## Neural baselines\n\n")
            f.write(
                "| Baseline | Winner mean | Baseline mean | Difference"
                " | Relative gain | Per-comparison 95% CI | W/T/L"
                " | Raw p | Holm p | Significant |\n"
            )
            f.write(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
                " | ---: | --- |\n"
            )
            for r in neural_rows:
                ci = (
                    f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
                    if r["ci95_low"] is not None
                    else "-"
                )
                rel_imp = (
                    f"{r['relative_improvement'] * 100:.2f}%"
                    if r["relative_improvement"] is not None
                    else "-"
                )
                w_t_l = (
                    f"{r['wins']}/{r['ties']}/{r['losses']}"
                    if r["wins"] is not None
                    else "-"
                )
                raw_p = _format_p_value(r.get("raw_p_value"))
                holm_p = _format_p_value(r.get("holm_adjusted_p_value"))
                sig = "✅" if r.get("significant_after_holm") else "-"
                f.write(
                    f"| {r['baseline_model']} "
                    f"| {r['winner_mean']:.4f} "
                    f"| {r['baseline_mean']:.4f} "
                    f"| {r['mean_difference']:.6f} "
                    f"| {rel_imp} "
                    f"| {ci} "
                    f"| {w_t_l} "
                    f"| {raw_p} "
                    f"| {holm_p} "
                    f"| {sig} |\n"
                )

        if heuristic_rows:
            f.write("\n## Deterministic baselines\n\n")
            f.write(
                "| Baseline | Winner mean | Baseline value | Difference"
                " | Relative gain | Analysis |\n"
            )
            f.write(
                "| --- | ---: | ---: | ---: | ---: | --- |\n"
            )
            for r in heuristic_rows:
                rel_imp = (
                    f"{r['relative_improvement'] * 100:.2f}%"
                    if r["relative_improvement"] is not None
                    else "-"
                )
                f.write(
                    f"| {r['baseline_model']} "
                    f"| {r['winner_mean']:.4f} "
                    f"| {r['baseline_mean']:.4f} "
                    f"| {r['mean_difference']:.6f} "
                    f"| {rel_imp} "
                    f"| {r['note']} |\n"
                )


def main() -> None:
    args = parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    neural_seeds = {int(s) for s in manifest["neural_seeds"]}
    if len(neural_seeds) < 2:
        raise RuntimeError(
            "At least two neural seeds are required "
            f"for paired comparison, got {len(neural_seeds)}"
        )

    summary_rows = json.loads(Path(args.summary_file).read_text())
    winner, baselines = select_winner_and_baselines(summary_rows)

    with open(args.runs_file, newline="") as f:
        run_rows = list(csv.DictReader(f))

    results, seed_pairs = _run_comparisons(
        run_rows, winner, baselines, neural_seeds
    )

    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, winner, results, seed_pairs)

    n_neural = len([r for r in results if r["comparison_type"] != "descriptive"])
    n_heuristic = len([r for r in results if r["comparison_type"] == "descriptive"])
    n_sig = len([r for r in results if r.get("significant_after_holm")])

    print(
        f"Winner: {winner}  "
        f"Neural comparisons: {n_neural}  "
        f"Significant after Holm: {n_sig}  "
        f"Descriptive: {n_heuristic}  "
        f"Seed pairs: {len(seed_pairs)}"
    )


if __name__ == "__main__":
    main()
