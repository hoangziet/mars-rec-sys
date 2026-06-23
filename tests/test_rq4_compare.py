import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_compare


def _make_per_user(variant: str, seed: int, n_users: int = 5,
                   base_score: float = 0.1, delta: float = 0.0) -> pd.DataFrame:
    """Build a per-user DataFrame for one (variant, seed) pair."""
    rows = []
    for user in range(n_users):
        target = 100 + user
        rows.append({
            "variant": variant,
            "seed": seed,
            "user_idx": user,
            "target_item": target,
            "rank": 1,
            "hit_at_10": 1.0,
            "ndcg_at_10": base_score + delta,
            "hit_at_20": 1.0,
            "ndcg_at_20": base_score + delta,
        })
    return pd.DataFrame(rows)


# ---- _check_duplicates ----

def test_check_duplicates_raises_on_duplicate_keys():
    per_user = pd.DataFrame({
        "variant": ["V0", "V0"],
        "seed": [42, 42],
        "user_idx": [1, 1],
        "target_item": [100, 100],
        "ndcg_at_10": [0.1, 0.2],
    })
    with pytest.raises(RuntimeError, match="Duplicate"):
        rq4_compare._check_duplicates(per_user)


def test_check_duplicates_passes_on_unique_keys():
    per_user = pd.DataFrame({
        "variant": ["V0", "V0", "V0"],
        "seed": [42, 42, 42],
        "user_idx": [1, 2, 3],
        "target_item": [100, 101, 102],
        "ndcg_at_10": [0.1, 0.2, 0.3],
    })
    rq4_compare._check_duplicates(per_user)


def test_check_duplicates_allows_different_seeds_same_user():
    """Same (variant, user, target) across different seeds is OK."""
    per_user = pd.DataFrame({
        "variant": ["V0", "V0"],
        "seed": [42, 43],
        "user_idx": [1, 1],
        "target_item": [100, 100],
        "ndcg_at_10": [0.1, 0.2],
    })
    rq4_compare._check_duplicates(per_user)


# ---- _check_key_set_equality ----

def test_check_key_set_equality_raises_on_mismatch():
    per_user = pd.DataFrame({
        "variant": ["V0", "V0", "V1"],
        "seed": [42, 42, 42],
        "user_idx": [1, 2, 1],
        "target_item": [100, 101, 100],
        "ndcg_at_10": [0.1, 0.2, 0.1],
    })
    with pytest.raises(RuntimeError, match="Key-set mismatch"):
        rq4_compare._check_key_set_equality(per_user, ["V0", "V1"])


def test_check_key_set_equality_passes_on_match():
    per_user = pd.DataFrame({
        "variant": ["V0", "V0", "V1", "V1"],
        "seed": [42, 42, 42, 42],
        "user_idx": [1, 2, 1, 2],
        "target_item": [100, 101, 100, 101],
        "ndcg_at_10": [0.1, 0.2, 0.1, 0.2],
    })
    rq4_compare._check_key_set_equality(per_user, ["V0", "V1"])


# ---- _run_comparison: raw stats only, no label ----

def test_run_comparison_does_not_return_significance_label():
    """_run_comparison returns raw stats; significant is assigned later."""
    rows = []
    for user in range(10):
        target = 100 + user
        for seed in (42, 43):
            rows.append({"variant": "V0", "seed": seed, "user_idx": user, "target_item": target, "ndcg_at_10": 0.1})
            rows.append({"variant": "V1", "seed": seed, "user_idx": user, "target_item": target, "ndcg_at_10": 0.5})
    per_user = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    result = rq4_compare._run_comparison(per_user, "V1", "V0", {42, 43}, rng)
    assert "significance_label" not in result
    assert "significant" in result
    assert result["significant"] is None  # not assigned yet
    assert "permutation_p" in result
    assert "mean_difference" in result
    assert "bootstrap_ci_low" in result
    assert "bootstrap_ci_high" in result


# ---- _assign_significance_label: sets only significant boolean ----

def test_assign_primary_significant_true():
    """Holm-adjusted p < 0.05 + mean_diff > 0 + CI_low > 0 → significant=True."""
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=True)
    assert result["significant"] is True
    assert "significance_label" not in result


def test_assign_primary_significant_false_ci_crosses_zero():
    """Holm p < 0.05 but CI crosses zero → significant=False."""
    result = {"mean_difference": 0.01, "bootstrap_ci_low": -0.001, "bootstrap_ci_high": 0.021}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=True)
    assert result["significant"] is False


def test_assign_primary_significant_false_weak_p():
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.10, is_primary=True)
    assert result["significant"] is False


def test_assign_secondary_significant_always_false():
    """Secondary comparisons never get significant=True."""
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=False)
    assert result["significant"] is False

    result2 = {"mean_difference": -0.001, "bootstrap_ci_low": -0.008, "bootstrap_ci_high": 0.006}
    rq4_compare._assign_significance_label(result2, holm_adjusted_p=None, is_primary=False)
    assert result2["significant"] is False


def test_run_comparison_drops_practically_significant():
    rows = []
    for user in range(10):
        target = 100 + user
        for seed in (42, 43):
            rows.append({"variant": "V0", "seed": seed, "user_idx": user, "target_item": target, "ndcg_at_10": 0.1})
            rows.append({"variant": "V1", "seed": seed, "user_idx": user, "target_item": target, "ndcg_at_10": 0.5})
    per_user = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    result = rq4_compare._run_comparison(per_user, "V1", "V0", {42, 43}, rng)
    assert "practically_significant" not in result
    assert "relative_improvement_pct" in result
    assert "abs_mean_difference" in result
    assert result["relative_improvement_pct"] is not None
    assert result["abs_mean_difference"] >= 0


# ---- PRACTICAL_THRESHOLD removed ----

def test_practical_threshold_constant_removed():
    assert not hasattr(rq4_compare, "PRACTICAL_THRESHOLD")
