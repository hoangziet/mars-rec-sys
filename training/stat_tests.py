from __future__ import annotations

import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW


def count_wins_ties_losses(differences) -> tuple[int, int, int]:
    values = np.asarray(differences, dtype=float)
    tie_mask = np.isclose(values, 0.0)
    wins = int(np.sum((values > 0) & ~tie_mask))
    ties = int(np.sum(tie_mask))
    losses = int(np.sum((values < 0) & ~tie_mask))
    return wins, ties, losses


def compute_seed_paired_t_test(left_values, right_values) -> dict:
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    differences = left - right
    if np.all(differences == 0):
        wins, ties, losses = 0, len(differences), 0
        return {
            "mean_difference": 0.0,
            "std_difference": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "t_statistic": 0.0,
            "raw_p_value": 1.0,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    wins, ties, losses = count_wins_ties_losses(differences)
    t_result = scipy.stats.ttest_rel(left, right, alternative="two-sided")
    ci_low, ci_high = DescrStatsW(differences).tconfint_mean(alpha=0.05)
    return {
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t_statistic": float(t_result.statistic),
        "raw_p_value": float(t_result.pvalue),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def apply_holm_correction(rows: list[dict], p_key: str = "raw_p_value") -> list[dict]:
    if not rows:
        return rows
    raw_p_values = [row[p_key] for row in rows]
    reject, adjusted_p, _, _ = multipletests(raw_p_values, alpha=0.05, method="holm")
    for row, adjusted, significant in zip(rows, adjusted_p, reject, strict=True):
        row["holm_adjusted_p_value"] = float(adjusted)
        row["significant_after_holm"] = bool(significant)
    return rows
