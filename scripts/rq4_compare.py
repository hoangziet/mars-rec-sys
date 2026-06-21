"""
scripts/rq4_compare.py
======================
RQ4: Multi-variant statistical comparison with Holm correction.

Primary comparisons (Holm-corrected):
    V1 vs V0, V2 vs V0, V3 vs V0

Secondary comparisons (descriptive only):
    V3 vs V1, V3 vs V2

Usage:
    uv run python scripts/rq4_compare.py --runs-file ... --summary-file ... --manifest ... --output-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRIMARY_METRIC = "test_NDCG_at_10"
METRIC_LABEL = "Test NDCG@10"
PRIMARY_COMPARISONS = [("V1", "V0"), ("V2", "V0"), ("V3", "V0")]
SECONDARY_COMPARISONS = [("V3", "V1"), ("V3", "V2")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: multi-variant comparison.")
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _index_by_seed(rows: list[dict], variant: str, metric: str) -> dict[int, float]:
    by_seed: dict[int, float] = {}
    for row in rows:
        if row["variant"] != variant:
            continue
        seed = int(row["seed"])
        if seed in by_seed:
            raise RuntimeError(f"Duplicate seed {seed} for variant {variant}")
        by_seed[seed] = float(row[metric])
    return by_seed


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-6:
        return f"{value:.2e}"
    return f"{value:.6f}"


def _compute_paired_stats(winner_vals: np.ndarray, baseline_vals: np.ndarray) -> dict:
    diffs = winner_vals - baseline_vals
    if np.all(diffs == 0):
        return {"mean_diff": 0.0, "std_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "t_stat": 0.0, "p_value": 1.0, "wins": 0, "ties": len(diffs), "losses": 0}
    tie_mask = np.isclose(diffs, 0.0)
    wins = int(np.sum((diffs > 0) & ~tie_mask))
    ties = int(np.sum(tie_mask))
    losses = int(np.sum((diffs < 0) & ~tie_mask))
    t_result = scipy.stats.ttest_rel(winner_vals, baseline_vals, alternative="two-sided")
    diff_stats = DescrStatsW(diffs)
    ci_low, ci_high = diff_stats.tconfint_mean(alpha=0.05)
    return {"mean_diff": float(diffs.mean()), "std_diff": float(diffs.std(ddof=1)), "ci95_low": float(ci_low), "ci95_high": float(ci_high), "t_stat": float(t_result.statistic), "p_value": float(t_result.pvalue), "wins": wins, "ties": ties, "losses": losses}


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    expected_seeds = {int(s) for s in manifest["neural_seeds"]}
    summary_rows = json.loads(Path(args.summary_file).read_text())
    variants = sorted({r["variant"] for r in summary_rows})

    with open(args.runs_file, newline="") as f:
        run_rows = list(csv.DictReader(f))

    by_variant: dict[str, dict[int, float]] = {}
    for v in variants:
        by_variant[v] = _index_by_seed(run_rows, v, PRIMARY_METRIC)

    primary_results = []
    for comp_variant, base_variant in PRIMARY_COMPARISONS:
        if comp_variant not in by_variant or base_variant not in by_variant:
            continue
        comp_vals = np.array([by_variant[comp_variant][s] for s in sorted(expected_seeds)])
        base_vals = np.array([by_variant[base_variant][s] for s in sorted(expected_seeds)])
        stats = _compute_paired_stats(comp_vals, base_vals)
        primary_results.append({"comparison": f"{comp_variant} vs {base_variant}", "comparison_type": "primary", "comp_variant": comp_variant, "base_variant": base_variant, "comp_mean": float(comp_vals.mean()), "base_mean": float(base_vals.mean()), **stats, "holm_p": None, "significant": None})

    if primary_results:
        raw_p = [r["p_value"] for r in primary_results]
        reject, adjusted, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for r, adj, sig in zip(primary_results, adjusted, reject):
            r["holm_p"] = float(adj)
            r["significant"] = bool(sig)

    secondary_results = []
    for comp_variant, base_variant in SECONDARY_COMPARISONS:
        if comp_variant not in by_variant or base_variant not in by_variant:
            continue
        comp_vals = np.array([by_variant[comp_variant][s] for s in sorted(expected_seeds)])
        base_vals = np.array([by_variant[base_variant][s] for s in sorted(expected_seeds)])
        stats = _compute_paired_stats(comp_vals, base_vals)
        secondary_results.append({"comparison": f"{comp_variant} vs {base_variant}", "comparison_type": "secondary", "comp_variant": comp_variant, "base_variant": base_variant, "comp_mean": float(comp_vals.mean()), "base_mean": float(base_vals.mean()), **stats, "holm_p": None, "significant": None})

    all_results = primary_results + secondary_results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = ["comparison", "comparison_type", "comp_variant", "base_variant", "comp_mean", "base_mean", "mean_diff", "std_diff", "ci95_low", "ci95_high", "wins", "ties", "losses", "t_stat", "p_value", "holm_p", "significant"]
    with open(output_dir / "rq4_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    with open(output_dir / "rq4_comparison.md", "w") as f:
        f.write("# RQ4 Ablation Comparison\n\n")
        f.write(f"Primary metric: {METRIC_LABEL}\n")
        f.write("Statistical test: Two-sided paired t-test\n")
        f.write("Multiple-comparison correction: Holm (primary comparisons only)\n")
        f.write("Family-wise significance level: α = 0.05\n\n")
        f.write("## Primary comparisons (Holm-corrected)\n\n")
        f.write("| Comparison | Comp mean | Base mean | Difference | 95% CI | W/T/L | Raw p | Holm p | Sig |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for r in primary_results:
            ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
            wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
            sig = "✅" if r["significant"] else "-"
            f.write(f"| {r['comparison']} | {r['comp_mean']:.4f} | {r['base_mean']:.4f} | {r['mean_diff']:.6f} | {ci} | {wtl} | {_format_p_value(r['p_value'])} | {_format_p_value(r['holm_p'])} | {sig} |\n")
        f.write("\n## Secondary comparisons (descriptive)\n\n")
        f.write("| Comparison | Comp mean | Base mean | Difference | 95% CI | W/T/L | Raw p |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in secondary_results:
            ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
            wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
            f.write(f"| {r['comparison']} | {r['comp_mean']:.4f} | {r['base_mean']:.4f} | {r['mean_diff']:.6f} | {ci} | {wtl} | {_format_p_value(r['p_value'])} |\n")

    print(f"Primary comparisons: {len(primary_results)}")
    print(f"Secondary comparisons: {len(secondary_results)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
