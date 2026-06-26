import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.stat_tests import (
    apply_holm_correction,
    compute_seed_paired_t_test,
    count_wins_ties_losses,
)


def test_compute_seed_paired_t_test_returns_expected_keys():
    result = compute_seed_paired_t_test(
        left_values=[0.31, 0.29, 0.33, 0.30, 0.32],
        right_values=[0.28, 0.27, 0.29, 0.28, 0.30],
    )
    assert set(result) == {
        "mean_difference",
        "std_difference",
        "ci95_low",
        "ci95_high",
        "t_statistic",
        "raw_p_value",
        "wins",
        "ties",
        "losses",
    }
    assert result["wins"] == 5
    assert result["ties"] == 0
    assert result["losses"] == 0


def test_apply_holm_correction_keeps_pair_order():
    rows = [
        {"baseline": "a", "raw_p_value": 0.01},
        {"baseline": "b", "raw_p_value": 0.03},
        {"baseline": "c", "raw_p_value": 0.20},
    ]
    updated = apply_holm_correction(rows, p_key="raw_p_value")
    assert [row["baseline"] for row in updated] == ["a", "b", "c"]
    assert all("holm_adjusted_p_value" in row for row in updated)
    assert all("significant_after_holm" in row for row in updated)


def test_count_wins_ties_losses_handles_exact_ties():
    wins, ties, losses = count_wins_ties_losses([0.02, 0.0, -0.01, 0.0])
    assert (wins, ties, losses) == (1, 2, 1)
