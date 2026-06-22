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
    """_run_comparison returns raw stats; label is assigned later (after Holm)."""
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
    assert "permutation_p" in result
    assert "mean_difference" in result
    assert "bootstrap_ci_low" in result
    assert "bootstrap_ci_high" in result


# ---- _assign_significance_label (called in main after Holm) ----

def test_assign_primary_label_improvement():
    """Holm-adjusted p < 0.05 + mean_diff > 0 + CI_low > 0 → reliable_improvement."""
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=True)
    assert result["significance_label"] == "reliable_improvement"


def test_assign_primary_label_degradation():
    result = {"mean_difference": -0.01, "bootstrap_ci_low": -0.018, "bootstrap_ci_high": -0.001}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.02, is_primary=True)
    assert result["significance_label"] == "reliable_degradation"


def test_assign_primary_label_inconclusive_p_value():
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.10, is_primary=True)
    assert result["significance_label"] == "inconclusive"


def test_assign_primary_label_inconclusive_ci_crosses_zero():
    result = {"mean_difference": 0.01, "bootstrap_ci_low": -0.001, "bootstrap_ci_high": 0.021}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=True)
    assert result["significance_label"] == "inconclusive"


def test_assign_secondary_label_descriptive_only():
    """Secondary comparisons never get 'reliable_*' — they are always descriptive."""
    result = {"mean_difference": 0.01, "bootstrap_ci_low": 0.002, "bootstrap_ci_high": 0.018}
    rq4_compare._assign_significance_label(result, holm_adjusted_p=0.03, is_primary=False)
    assert result["significance_label"] == "descriptive_improvement"

    result2 = {"mean_difference": -0.001, "bootstrap_ci_low": -0.008, "bootstrap_ci_high": 0.006}
    rq4_compare._assign_significance_label(result2, holm_adjusted_p=None, is_primary=False)
    assert result2["significance_label"] == "descriptive_inconclusive"


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
