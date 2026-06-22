"""
scripts/rq4_report.py
=====================
RQ4: Final report aggregation.

Usage:
    uv run python scripts/rq4_report.py --benchmark-id rq4-ablation --comparison-dir experiments/rq4/rq4-ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: final report.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    comparison_dir = Path(args.comparison_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq4" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = comparison_dir / "rq4_comparison.csv"
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Comparison results not found: {comparison_csv}")

    with open(comparison_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    primary = [r for r in rows if r["comparison_type"] == "primary"]
    secondary = [r for r in rows if r["comparison_type"] == "secondary"]

    thresholds_path = comparison_dir / "rq4_subgroup_thresholds.json"
    thresholds = None
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())

    subgroup_path = comparison_dir / "rq4_subgroup_analysis.md"
    subgroup_content = None
    if subgroup_path.exists():
        subgroup_content = subgroup_path.read_text()

    with open(output_dir / "rq4_final_report.md", "w") as f:
        f.write("# RQ4 Final Ablation Report\n\n")
        f.write(f"Benchmark: {args.benchmark_id}\n\n")

        f.write("## Primary Findings\n\n")
        f.write(
            "Each primary finding reports the effect size (Δ, relative %), the 95% bootstrap CI, "
            "the permutation p-value with Holm correction, Cohen's d, and a significance label "
            "derived from the Holm-adjusted p-value and bootstrap CI direction. Practical significance "
            "is left to the reader: inspect the 95% CI against your domain's noise floor; "
            "this report does not enforce a binary threshold.\n\n"
        )
        for r in primary:
            label = r.get("significance_label", "inconclusive")
            sig = "statistically significant" if r.get("significant") == "True" else "not significant"
            f.write(f"- **{r['comparison']}**: {sig} — {label} (Holm p = {r.get('holm_adjusted_p', '-')})\n")
            f.write(f"  - Mean difference: {float(r['mean_difference']):.6f}\n")
            f.write(f"  - Relative improvement: {r.get('relative_improvement_pct', '-')}% (Δ/|baseline| = {r.get('relative_improvement', '-')})\n")
            f.write(f"  - 95% bootstrap CI: [{r['bootstrap_ci_low']}, {r['bootstrap_ci_high']}]\n")
            f.write(f"  - Cohen's d: {float(r.get('cohens_d', 0)):.3f}\n")
            f.write(f"  - Wins / ties / losses (users): {r.get('wins', '-')} / {r.get('ties', '-')} / {r.get('losses', '-')}\n\n")

        f.write("## Secondary Findings (Descriptive)\n\n")
        for r in secondary:
            f.write(f"- **{r['comparison']}**: mean diff = {float(r['mean_difference']):.6f}, Cohen's d = {float(r.get('cohens_d', 0)):.3f}\n")

        if thresholds:
            f.write("\n## Subgroup Thresholds\n\n")
            for k, v in thresholds.items():
                f.write(f"- {k}: {v}\n")

        if subgroup_content:
            f.write(f"\n{subgroup_content}\n")

    for fname in ["rq4_comparison.csv", "rq4_comparison.md"]:
        src = comparison_dir / fname
        dst = output_dir / fname
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    print(f"Final report: {output_dir / 'rq4_final_report.md'}")


if __name__ == "__main__":
    main()
