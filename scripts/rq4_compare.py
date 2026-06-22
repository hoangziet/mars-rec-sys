"""
scripts/rq4_compare.py
======================
RQ4: User-level statistical comparison with Holm correction.

Reads per-user CSVs from rq4_ablation, joins by (seed, user_idx),
computes per-user delta, then runs:
- Paired bootstrap CI (user-level)
- Paired permutation test (sign-flip, user-level)
- Paired t-test (seed-level, supplementary)
- Cohen's d effect size
- Holm correction on 3 primary comparisons

Primary comparisons (Holm-corrected):
    V1 vs V0, V2 vs V0, V3 vs V0
Secondary comparisons (descriptive only):
    V3 vs V1, V3 vs V2

Usage:
    uv run python scripts/rq4_compare.py --per-user-dir ... --manifest ... --output-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRIMARY_METRIC = "ndcg_at_10"
METRIC_LABEL = "NDCG@10"
PRIMARY_COMPARISONS = [("V1", "V0"), ("V2", "V0"), ("V3", "V0")]
SECONDARY_COMPARISONS = [("V3", "V1"), ("V3", "V2")]
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
RNG_SEED = 42
PRACTICAL_THRESHOLD = 0.01  # absolute NDCG@10 improvement threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: user-level statistical comparison.")
    parser.add_argument("--per-user-dir", required=True, help="Directory with per-user CSV files")
    parser.add_argument("--manifest", required=True, help="rq4_manifest.json")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _load_per_user(per_user_dir: Path, variants: list[str], seeds: list[int]) -> pd.DataFrame:
    """Load and concatenate per-user CSVs for given variants and seeds."""
    frames = []
    for variant in variants:
        for seed in seeds:
            path = per_user_dir / f"{variant}_s{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing per-user file: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise RuntimeError(f"Empty per-user file: {path}")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _join_by_user(per_user: pd.DataFrame, comp_variant: str, base_variant: str,
                  metric: str, expected_seed_count: int) -> pd.DataFrame:
    """Join two variants by (seed, user_idx, target_item), then average delta per user across seeds.

    Returns one row per user with averaged delta.
    """
    comp = per_user[per_user["variant"] == comp_variant][["seed", "user_idx", "target_item", metric]].copy()
    base = per_user[per_user["variant"] == base_variant][["seed", "user_idx", "target_item", metric]].copy()
    comp = comp.rename(columns={metric: "comp_metric"})
    base = base.rename(columns={metric: "base_metric"})

    merged = comp.merge(base, on=["seed", "user_idx", "target_item"], how="inner")
    if len(merged) == 0:
        raise RuntimeError(f"No matching users between {comp_variant} and {base_variant}")

    # Validate no dropped users
    comp_users = set(zip(comp["user_idx"], comp["target_item"]))
    base_users = set(zip(base["user_idx"], base["target_item"]))
    common_users = comp_users & base_users
    if len(common_users) != len(comp_users) or len(common_users) != len(base_users):
        raise RuntimeError(
            f"User mismatch: comp has {len(comp_users)}, base has {len(base_users)}, "
            f"common={len(common_users)}"
        )

    merged["delta"] = merged["comp_metric"] - merged["base_metric"]

    # Average delta per user across seeds
    user_avg = merged.groupby(["user_idx", "target_item"]).agg(
        delta=("delta", "mean"),
        comp_metric=("comp_metric", "mean"),
        base_metric=("base_metric", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    if not (user_avg["n_seeds"] == expected_seed_count).all():
        bad = user_avg[user_avg["n_seeds"] != expected_seed_count][["user_idx", "target_item", "n_seeds"]]
        raise RuntimeError(
            "expected all users to have complete seed coverage; got "
            f"{bad.to_dict(orient='records')[:10]}"
        )

    return user_avg


def _bootstrap_ci(deltas: np.ndarray, n_bootstrap: int = BOOTSTRAP_N,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Paired bootstrap 95% CI for mean delta."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(deltas)
    boot_means = np.array([
        deltas[rng.integers(0, n, n)].mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _permutation_test(deltas: np.ndarray, n_perm: int = PERMUTATION_N,
                      rng: np.random.Generator | None = None) -> float:
    """Paired permutation (sign-flip) test for H0: mean(delta) = 0.

    Returns two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    observed_mean = abs(deltas.mean())
    n = len(deltas)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = abs((deltas * signs).mean())
        if perm_mean >= observed_mean:
            count += 1
    return (count + 1) / (n_perm + 1)


def _cohens_d(deltas: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(delta) / std(delta)."""
    std = deltas.std(ddof=1)
    if std == 0:
        return 0.0
    return float(deltas.mean() / std)


def _seed_level_ttest(per_user: pd.DataFrame, comp_variant: str, base_variant: str,
                      metric: str, expected_seeds: set[int]) -> dict:
    """Supplementary seed-level paired t-test on per-seed means."""
    comp_seeds = per_user[per_user["variant"] == comp_variant].groupby("seed")[metric].mean()
    base_seeds = per_user[per_user["variant"] == base_variant].groupby("seed")[metric].mean()

    comp_vals = np.array([comp_seeds[s] for s in sorted(expected_seeds) if s in comp_seeds.index])
    base_vals = np.array([base_seeds[s] for s in sorted(expected_seeds) if s in base_seeds.index])

    if len(comp_vals) < 2 or len(base_vals) < 2:
        return {"t_stat": None, "t_p": None, "n_seeds": min(len(comp_vals), len(base_vals))}

    t_result = scipy.stats.ttest_rel(comp_vals, base_vals, alternative="two-sided")
    return {
        "t_stat": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "n_seeds": len(comp_vals),
    }


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-6:
        return f"{value:.2e}"
    return f"{value:.6f}"


def _run_comparison(per_user: pd.DataFrame, comp_variant: str, base_variant: str,
                    expected_seeds: set[int], rng: np.random.Generator) -> dict:
    """Run full comparison between two variants."""
    merged = _join_by_user(per_user, comp_variant, base_variant, PRIMARY_METRIC, len(expected_seeds))
    deltas = merged["delta"].values
    n_users = len(deltas)

    # User-level bootstrap CI
    ci_low, ci_high = _bootstrap_ci(deltas, rng=rng)

    # User-level permutation test
    perm_p = _permutation_test(deltas, rng=rng)

    # Effect size
    d = _cohens_d(deltas)

    # Seed-level t-test (supplementary)
    seed_test = _seed_level_ttest(per_user, comp_variant, base_variant, PRIMARY_METRIC, expected_seeds)

    # Wins/ties/losses
    tie_mask = np.isclose(deltas, 0.0)
    wins = int(np.sum((deltas > 0) & ~tie_mask))
    ties = int(np.sum(tie_mask))
    losses = int(np.sum((deltas < 0) & ~tie_mask))

    comp_mean = float(merged["comp_metric"].mean())
    base_mean = float(merged["base_metric"].mean())
    mean_diff = float(deltas.mean())
    relative_imp = (mean_diff / base_mean) if base_mean != 0 else None

    return {
        "comparison": f"{comp_variant} vs {base_variant}",
        "comp_variant": comp_variant,
        "base_variant": base_variant,
        "n_users": n_users,
        "comp_mean": comp_mean,
        "base_mean": base_mean,
        "mean_difference": mean_diff,
        "relative_improvement": relative_imp,
        "practically_significant": bool(mean_diff >= PRACTICAL_THRESHOLD),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p": perm_p,
        "cohens_d": d,
        "seed_t_stat": seed_test.get("t_stat"),
        "seed_t_p": seed_test.get("t_p"),
        "n_seeds": seed_test.get("n_seeds"),
        "holm_adjusted_p": None,
        "significant": None,
    }


def main() -> None:
    args = parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    expected_seeds = {int(s) for s in manifest["neural_seeds"]}
    variants = manifest["variants"]

    per_user_dir = Path(args.per_user_dir)
    per_user = _load_per_user(per_user_dir, variants, sorted(expected_seeds))

    rng = np.random.default_rng(RNG_SEED)

    # Primary comparisons
    primary_results = []
    for comp_variant, base_variant in PRIMARY_COMPARISONS:
        result = _run_comparison(per_user, comp_variant, base_variant, expected_seeds, rng)
        result["comparison_type"] = "primary"
        primary_results.append(result)

    # Holm correction on permutation p-values
    if primary_results:
        raw_p = [r["permutation_p"] for r in primary_results]
        reject, adjusted, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for r, adj, sig in zip(primary_results, adjusted, reject):
            r["holm_adjusted_p"] = float(adj)
            r["significant"] = bool(sig)

    # Secondary comparisons
    secondary_results = []
    for comp_variant, base_variant in SECONDARY_COMPARISONS:
        result = _run_comparison(per_user, comp_variant, base_variant, expected_seeds, rng)
        result["comparison_type"] = "secondary"
        secondary_results.append(result)

    all_results = primary_results + secondary_results

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fields = [
        "comparison", "comparison_type", "comp_variant", "base_variant",
        "n_users", "comp_mean", "base_mean", "mean_difference", "relative_improvement",
        "practically_significant",
        "wins", "ties", "losses",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "permutation_p", "cohens_d",
        "seed_t_stat", "seed_t_p", "n_seeds",
        "holm_adjusted_p", "significant",
    ]
    with open(output_dir / "rq4_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Write Markdown
    with open(output_dir / "rq4_comparison.md", "w") as f:
        f.write("# RQ4 Ablation Comparison\n\n")
        f.write(f"Primary metric: {METRIC_LABEL}\n")
        f.write(f"User-level paired bootstrap CI ({BOOTSTRAP_N} resamples)\n")
        f.write(f"User-level paired permutation test ({PERMUTATION_N} sign-flips)\n")
        f.write("Multiple-comparison correction: Holm (primary comparisons only)\n")
        f.write("Family-wise significance level: α = 0.05\n")
        f.write(f"Practical significance threshold: Δ ≥ {PRACTICAL_THRESHOLD}\n\n")

        f.write("## Primary comparisons (Holm-corrected)\n\n")
        f.write("| Comparison | Comp | Base | Δ | Rel | Practical | 95% CI (bootstrap) | W/T/L | Perm p | Holm p | Cohen's d | Sig |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for r in primary_results:
            ci = f"[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}]"
            wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
            rel = f"{r['relative_improvement']*100:.2f}%" if r["relative_improvement"] is not None else "-"
            pract = "✅" if r.get("practically_significant") else "-"
            sig = "✅" if r["significant"] else "-"
            f.write(
                f"| {r['comparison']} "
                f"| {r['comp_mean']:.4f} "
                f"| {r['base_mean']:.4f} "
                f"| {r['mean_difference']:.6f} "
                f"| {rel} "
                f"| {pract} "
                f"| {ci} "
                f"| {wtl} "
                f"| {_format_p_value(r['permutation_p'])} "
                f"| {_format_p_value(r['holm_adjusted_p'])} "
                f"| {r['cohens_d']:.3f} "
                f"| {sig} |\n"
            )

        f.write("\n## Secondary comparisons (descriptive)\n\n")
        f.write("| Comparison | Comp | Base | Δ | Rel | 95% CI (bootstrap) | W/T/L | Perm p | Cohen's d |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in secondary_results:
            ci = f"[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}]"
            wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
            rel = f"{r['relative_improvement']*100:.2f}%" if r["relative_improvement"] is not None else "-"
            f.write(
                f"| {r['comparison']} "
                f"| {r['comp_mean']:.4f} "
                f"| {r['base_mean']:.4f} "
                f"| {r['mean_difference']:.6f} "
                f"| {rel} "
                f"| {ci} "
                f"| {wtl} "
                f"| {_format_p_value(r['permutation_p'])} "
                f"| {r['cohens_d']:.3f} |\n"
            )

        f.write("\n## Seed-level paired t-test (supplementary)\n\n")
        f.write("| Comparison | Seeds | t | p |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for r in all_results:
            t = f"{r['seed_t_stat']:.4f}" if r['seed_t_stat'] is not None else "-"
            p = _format_p_value(r['seed_t_p'])
            f.write(f"| {r['comparison']} | {r['n_seeds']} | {t} | {p} |\n")

    n_sig = len([r for r in primary_results if r["significant"]])
    print(f"Primary comparisons: {len(primary_results)}, significant after Holm: {n_sig}")
    print(f"Secondary comparisons: {len(secondary_results)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
