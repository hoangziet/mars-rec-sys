import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq1_compare import (
    _compute_paired_stats,
    _format_p_value,
    _index_by_seed,
    _run_comparisons,
    _write_outputs,
    pair_models_by_seed,
    relative_improvement,
    select_winner_and_baselines,
)
from training.stat_tests import count_wins_ties_losses


# ---- select_winner_and_baselines ----

def test_selects_winner_by_validation_rank():
    rows = [
        {"model": "GRU4Rec", "validation_rank": 2},
        {"model": "gSASRec", "validation_rank": 1},
        {"model": "SASRec", "validation_rank": 3},
    ]
    winner, baselines = select_winner_and_baselines(rows)
    assert winner == "gSASRec"
    assert baselines == ["GRU4Rec", "SASRec"]


def test_returns_all_other_models_as_baselines():
    rows = [
        {"model": "A", "validation_rank": 1},
        {"model": "B", "validation_rank": 2},
        {"model": "C", "validation_rank": 3},
        {"model": "D", "validation_rank": 4},
    ]
    winner, baselines = select_winner_and_baselines(rows)
    assert winner == "A"
    assert baselines == ["B", "C", "D"]


def test_fails_when_less_than_two_models():
    rows = [{"model": "X", "validation_rank": 1}]
    with pytest.raises(RuntimeError, match="At least two"):
        select_winner_and_baselines(rows)


# ---- pair_models_by_seed ----

def test_pairs_models_using_manifest_seeds():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
        {"model": "gSASRec", "seed": "123", "test_NDCG_at_10": "0.31"},
        {"model": "GRU4Rec", "seed": "123", "test_NDCG_at_10": "0.27"},
    ]
    pairs = pair_models_by_seed(
        rows, "gSASRec", "GRU4Rec", "test_NDCG_at_10",
        expected_seeds={42, 123},
    )
    assert pairs == [(42, 0.30, 0.28), (123, 0.31, 0.27)]


def test_rejects_missing_seed():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
    ]
    with pytest.raises(RuntimeError, match="expected seeds"):
        pair_models_by_seed(
            rows, "gSASRec", "GRU4Rec", "test_NDCG_at_10",
            expected_seeds={42, 123},
        )


def test_rejects_extra_seed():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "gSASRec", "seed": "123", "test_NDCG_at_10": "0.31"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
        {"model": "GRU4Rec", "seed": "123", "test_NDCG_at_10": "0.27"},
    ]
    with pytest.raises(RuntimeError, match="expected seeds"):
        pair_models_by_seed(
            rows, "gSASRec", "GRU4Rec", "test_NDCG_at_10",
            expected_seeds={42},
        )


def test_rejects_duplicate_seed():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.31"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
    ]
    with pytest.raises(RuntimeError, match="Duplicate seed"):
        pair_models_by_seed(
            rows, "gSASRec", "GRU4Rec", "test_NDCG_at_10",
            expected_seeds={42},
        )


# ---- _compute_paired_stats ----

def test_computes_two_sided_paired_t_test():
    winner_vals = np.array([0.30, 0.31, 0.29, 0.32, 0.28])
    baseline_vals = np.array([0.28, 0.27, 0.26, 0.29, 0.25])
    stats = _compute_paired_stats(winner_vals, baseline_vals)
    assert stats["raw_p_value"] < 0.05
    assert stats["wins"] == 5
    assert stats["ties"] == 0
    assert stats["losses"] == 0


def test_handles_zero_differences():
    vals = np.array([0.30, 0.28, 0.29, 0.31, 0.27])
    stats = _compute_paired_stats(vals, vals.copy())
    assert stats["raw_p_value"] == 1.0
    assert stats["t_statistic"] == 0.0
    assert stats["ci95_low"] == 0.0
    assert stats["ci95_high"] == 0.0
    assert stats["wins"] == 0
    assert stats["ties"] == 5
    assert stats["losses"] == 0


def test_counts_wins_ties_losses():
    winner_vals = np.array([0.30, 0.30, 0.30, 0.29, 0.27])
    baseline_vals = np.array([0.25, 0.28, 0.32, 0.27, 0.27])
    stats = _compute_paired_stats(winner_vals, baseline_vals)
    assert stats["wins"] == 3
    assert stats["ties"] == 1
    assert stats["losses"] == 1


def test_mutually_exclusive_wins_ties_losses():
    differences = np.array([0.05, 1e-10, -1e-10, 0.0, -0.02])
    wins, ties, losses = count_wins_ties_losses(differences)
    assert wins == 1
    assert ties == 3
    assert losses == 1
    assert wins + ties + losses == len(differences)


# ---- relative_improvement ----

def test_relative_improvement_basic():
    assert relative_improvement(0.30, 0.25) == pytest.approx(0.20)


def test_relative_improvement_rejects_zero_baseline():
    assert relative_improvement(0.30, 0.0) is None


# ---- _index_by_seed ----

def test_index_by_seed_rejects_duplicate():
    rows = [
        {"model": "X", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "X", "seed": "42", "test_NDCG_at_10": "0.31"},
    ]
    with pytest.raises(RuntimeError, match="Duplicate seed"):
        _index_by_seed(rows, "X", "test_NDCG_at_10")


# ---- _format_p_value ----

def test_format_p_value_uses_scientific_for_tiny_values():
    assert _format_p_value(1.88e-08) == "1.88e-08"
    assert _format_p_value(9.02e-10) == "9.02e-10"
    assert _format_p_value(1e-7) == "1.00e-07"


def test_format_p_value_uses_six_decimals_for_normal_values():
    assert _format_p_value(0.05) == "0.050000"
    assert _format_p_value(0.000232) == "0.000232"


def test_format_p_value_returns_dash_for_none():
    assert _format_p_value(None) == "-"


# ---- _run_comparisons ----

def _make_run_rows(model_seed_pairs):
    rows = []
    for model, seed, val in model_seed_pairs:
        rows.append({
            "model": model,
            "seed": str(seed),
            "test_NDCG_at_10": str(val),
        })
    return rows


def test_run_comparisons_returns_all_baselines():
    rows = _make_run_rows([
        ("W", 42, 0.30), ("W", 123, 0.31),
        ("A", 42, 0.28), ("A", 123, 0.27),
        ("B", 42, 0.26), ("B", 123, 0.25),
        ("popularity", 42, 0.10),
        ("itemcf", 42, 0.05),
    ])
    results, seed_pairs = _run_comparisons(
        rows, winner="W", baselines=["A", "B", "popularity", "itemcf"],
        neural_seeds={42, 123},
    )
    assert len(results) == 4
    comparisons = {r["baseline_model"]: r for r in results}
    assert comparisons["A"]["comparison_type"] == "seed_paired_t_test"
    assert comparisons["B"]["comparison_type"] == "seed_paired_t_test"
    assert comparisons["popularity"]["comparison_type"] == "descriptive"
    assert comparisons["itemcf"]["comparison_type"] == "descriptive"
    assert comparisons["popularity"]["raw_p_value"] is None


def test_holm_correction_is_applied():
    rows = _make_run_rows([
        ("W", 42, 0.30), ("W", 123, 0.31),
        ("A", 42, 0.28), ("A", 123, 0.29),
        ("B", 42, 0.20), ("B", 123, 0.21),
    ])
    results, _ = _run_comparisons(
        rows, winner="W", baselines=["A", "B"],
        neural_seeds={42, 123},
    )
    for r in results:
        assert r["holm_adjusted_p_value"] is not None
        assert r["holm_adjusted_p_value"] >= r["raw_p_value"]
        assert isinstance(r["significant_after_holm"], bool)


def test_heuristic_baseline_requires_exactly_one_run():
    rows = _make_run_rows([
        ("W", 42, 0.30), ("W", 123, 0.31),
        ("popularity", 42, 0.10), ("popularity", 123, 0.11),
    ])
    with pytest.raises(RuntimeError, match="expected exactly one deterministic run"):
        _run_comparisons(
            rows, winner="W", baselines=["popularity"],
            neural_seeds={42, 123},
        )


def test_rejects_heuristic_winner():
    rows = _make_run_rows([
        ("popularity", 42, 0.10),
        ("A", 42, 0.28), ("A", 123, 0.27),
    ])
    with pytest.raises(RuntimeError, match="deterministic model"):
        _run_comparisons(
            rows, winner="popularity", baselines=["A"],
            neural_seeds={42, 123},
        )


def test_produces_twenty_neural_seed_pairs():
    rows = _make_run_rows([
        ("W", s, 0.30) for s in range(5)
    ] + [
        ("A", s, 0.28) for s in range(5)
    ] + [
        ("B", s, 0.25) for s in range(5)
    ] + [
        ("popularity", 0, 0.10),
    ])
    _, seed_pairs = _run_comparisons(
        rows, winner="W", baselines=["A", "B", "popularity"],
        neural_seeds={0, 1, 2, 3, 4},
    )
    assert len(seed_pairs) == 10


# ---- _write_outputs ----

def test_write_outputs_removes_legacy_pairwise_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        legacy = out / "rq1_pairwise.csv"
        legacy.write_text("old")
        assert legacy.exists()

        results = [
            {
                "winner_model": "W", "baseline_model": "A",
                "comparison_type": "seed_paired_t_test",
                "metric": "X", "winner_mean": 0.3, "baseline_mean": 0.28,
                "mean_difference": 0.02, "relative_improvement": 0.071,
                "n_seed_pairs": 2, "wins": 2, "ties": 0, "losses": 0,
                "std_difference": 0.01, "ci95_low": 0.01, "ci95_high": 0.03,
                "t_statistic": 12.5, "raw_p_value": 0.1,
                "holm_adjusted_p_value": 0.1, "significant_after_holm": False,
                "note": None,
            },
        ]
        _write_outputs(out, "W", results, [])
        assert not legacy.exists()
        assert (out / "rq1_winner_vs_all.csv").exists()
        assert (out / "rq1_significance.md").exists()
        assert not (out / "rq1_seed_pairs.csv").exists()


def test_markdown_formats_small_p_values_in_scientific_notation():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        results = [
            {
                "winner_model": "W", "baseline_model": "A",
                "comparison_type": "seed_paired_t_test",
                "metric": "X", "winner_mean": 0.3, "baseline_mean": 0.28,
                "mean_difference": 0.02, "relative_improvement": 0.071,
                "n_seed_pairs": 5, "wins": 5, "ties": 0, "losses": 0,
                "std_difference": 0.01, "ci95_low": 0.01, "ci95_high": 0.03,
                "t_statistic": 12.5, "raw_p_value": 1.88e-08,
                "holm_adjusted_p_value": 9.02e-10,
                "significant_after_holm": True,
                "note": None,
            },
        ]
        _write_outputs(out, "W", results, [])
        md = (out / "rq1_significance.md").read_text()
        assert "1.88e-08" in md
        assert "9.02e-10" in md
        assert "0.000000" not in md
