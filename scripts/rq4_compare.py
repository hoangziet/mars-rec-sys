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
- Holm correction on primary comparisons

Comparisons are derived from the protocol manifest, not hardcoded:

Primary (Holm-corrected):
    every non-baseline variant vs the explicit ``baseline_variant``
    declared in the protocol manifest. There is no implicit fallback
    to ``variants[0]``; reordering or a custom variant list will not
    silently change the baseline.

Secondary (descriptive only):
    best-performing non-baseline variant vs every other non-baseline
    variant. No multiple-comparison correction; significance is always
    False for these.

The result manifest must declare ``baseline_variant`` and a unique list
of variants. Otherwise ``_resolve_baseline`` raises and the comparison
fails fast.

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
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
RNG_SEED = 42
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: user-level statistical comparison.")
    parser.add_argument("--per-user-dir", required=True, help="Directory with per-user CSV files")
    parser.add_argument("--manifest", required=True, help="rq4_result_manifest.json (produced by rq4_collect)")
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


def _check_duplicates(per_user: pd.DataFrame) -> None:
    """Fail-fast: same (variant, seed, user, target) twice inflates join counts."""
    dup_mask = per_user.duplicated(
        subset=["variant", "seed", "user_idx", "target_item"], keep=False
    )
    if dup_mask.any():
        sample = per_user[dup_mask].head(10)
        raise RuntimeError(
            f"Duplicate (variant, seed, user_idx, target_item) rows found in "
            f"per-user CSVs. {len(per_user[dup_mask])} duplicate rows. "
            f"First few: {sample.to_dict(orient='records')}"
        )


def _check_key_set_equality(per_user: pd.DataFrame, variants: list[str]) -> None:
    """Fail-fast: every variant must have identical (user_idx, target_item) keys.

    A paired join on (variant, seed, user_idx, target_item) silently drops rows
    when the key sets diverge, biasing the comparison.
    """
    key_sets: dict[str, set[tuple[int, int]]] = {}
    for variant in variants:
        sub = per_user[per_user["variant"] == variant]
        key_sets[variant] = set(zip(sub["user_idx"], sub["target_item"]))

    ref_variant = variants[0]
    ref_keys = key_sets[ref_variant]
    for variant in variants[1:]:
        other = key_sets[variant]
        if other != ref_keys:
            missing = ref_keys - other
            extra = other - ref_keys
            raise RuntimeError(
                f"Key-set mismatch: {ref_variant} has {len(ref_keys)} keys, "
                f"{variant} has {len(other)} keys. "
                f"Missing from {variant}: {len(missing)}, extra: {len(extra)}. "
                f"This would silently bias the paired comparison."
            )


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
    """Cohen's d for paired samples: mean(delta) / std(delta).

    Returns ``float('inf')`` when every user has the same non-zero delta
    (maximum effect).  Returns 0.0 when the mean delta is exactly 0.
    """
    mean = deltas.mean()
    std = deltas.std(ddof=1)
    if std == 0:
        if mean == 0:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / std)


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
        "relative_improvement_pct": (
            round(100.0 * mean_diff / base_mean, 2) if base_mean != 0 else None
        ),
        "abs_mean_difference": abs(mean_diff),
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


def _assign_significance_label(
    result: dict,
    holm_adjusted_p: float | None,
    is_primary: bool = True,
) -> None:
    """Assign ``significant`` as the sole public conclusion.

    Primary: ``significant`` is True only when the Holm-adjusted p < 0.05
    AND the bootstrap CI does not cross zero (same direction as mean_diff).

    Secondary: ``significant`` is always False (no multiple-comparison
    correction), and the result carries no ``significance_label`` in public
    output.
    """
    mean_diff = float(result["mean_difference"])
    ci_low = float(result["bootstrap_ci_low"])
    ci_high = float(result["bootstrap_ci_high"])

    if is_primary and holm_adjusted_p is not None and holm_adjusted_p < 0.05:
        if (mean_diff > 0 and ci_low > 0) or (mean_diff < 0 and ci_high < 0):
            result["significant"] = True
        else:
            result["significant"] = False
    else:
        result["significant"] = False


def _resolve_baseline(manifest: dict, variants: list[str]) -> str:
    """Pick the baseline variant for statistical comparison.

    The protocol manifest MUST declare ``baseline_variant`` explicitly.
    Variants must be unique. We never silently pick variants[0] because
    reordering or a custom variant list would silently change the baseline.
    """
    if not variants:
        raise RuntimeError("Cannot resolve baseline: variants list is empty")

    seen: dict[str, int] = {}
    for v in variants:
        seen[v] = seen.get(v, 0) + 1
    dupes = {k: c for k, c in seen.items() if c > 1}
    if dupes:
        raise RuntimeError(
            f"Variant IDs must be unique, found duplicates: {sorted(dupes)}. "
            "Re-init the protocol with unique variant IDs."
        )

    baseline = manifest.get("baseline_variant")
    if not baseline:
        raise RuntimeError(
            "Protocol manifest is missing 'baseline_variant'. "
            "Re-run rq4-init with --baseline-variant V0 (or another baseline). "
            "There is no implicit fallback to variants[0]."
        )
    if baseline not in variants:
        raise RuntimeError(
            f"baseline_variant={baseline!r} not in variants={variants}. "
            "Re-init the protocol or fix the manifest."
        )
    return baseline


def main() -> None:
    args = parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    expected_seeds = {int(s) for s in manifest["neural_seeds"]}
    variants = manifest["variants"]

    baseline = _resolve_baseline(manifest, variants)

    per_user_dir = Path(args.per_user_dir)
    per_user = _load_per_user(per_user_dir, variants, sorted(expected_seeds))

    _check_duplicates(per_user)
    _check_key_set_equality(per_user, variants)

    rng = np.random.default_rng(RNG_SEED)

    non_baseline = [v for v in variants if v != baseline]
    if not non_baseline:
        raise RuntimeError(f"No non-baseline variants to compare against {baseline}")

    primary_pairs = [(v, baseline) for v in non_baseline]

    # Primary comparisons
    primary_results = []
    for comp_variant, base_variant in primary_pairs:
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
            # assign label AFTER Holm + CI check
            _assign_significance_label(r, holm_adjusted_p=float(adj), is_primary=True)

    # Secondary comparisons: best non-baseline variant vs the rest
    secondary_results = []
    if len(non_baseline) > 1:
        best = max(non_baseline, key=lambda v: next(
            r["comp_mean"] for r in primary_results if r["comp_variant"] == v
        ))
        for other in non_baseline:
            if other != best:
                result = _run_comparison(per_user, best, other, expected_seeds, rng)
                result["comparison_type"] = "secondary"
                _assign_significance_label(result, holm_adjusted_p=None, is_primary=False)
                secondary_results.append(result)

    all_results = primary_results + secondary_results

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fields = [
        "comparison", "comparison_type", "comp_variant", "base_variant",
        "n_users", "comp_mean", "base_mean", "mean_difference", "relative_improvement",
        "relative_improvement_pct", "abs_mean_difference",
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
        f.write("Practical-significance threshold: not enforced. Report Δ, rel%, [95% CI].\n")
        f.write("``significant`` is True when Holm-adjusted p < 0.05 and the bootstrap CI does not cross zero.\n\n")

        f.write("## Primary comparisons (Holm-corrected)\n\n")
        f.write("| Comparison | Comp | Base | Δ | Rel % | 95% CI (bootstrap) | W/T/L | Perm p | Holm p | Cohen's d | Sig |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for r in primary_results:
            ci = f"[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}]"
            wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
            rel = f"{r['relative_improvement_pct']:.2f}%" if r["relative_improvement_pct"] is not None else "-"
            sig = "✅" if r["significant"] else "-"
            f.write(
                f"| {r['comparison']} "
                f"| {r['comp_mean']:.4f} "
                f"| {r['base_mean']:.4f} "
                f"| {r['mean_difference']:.6f} "
                f"| {rel} "
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
