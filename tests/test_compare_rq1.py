import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_rq1 import (
    _compute_paired_stats,
    _index_by_seed,
    pair_models_by_seed,
    relative_improvement,
    select_winner_and_baselines,
)


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


def test_pairs_models_using_manifest_seeds():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
        {"model": "gSASRec", "seed": "123", "test_NDCG_at_10": "0.31"},
        {"model": "GRU4Rec", "seed": "123", "test_NDCG_at_10": "0.27"},
    ]
    pairs = pair_models_by_seed(
        rows,
        "gSASRec",
        "GRU4Rec",
        "test_NDCG_at_10",
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
            rows,
            "gSASRec",
            "GRU4Rec",
            "test_NDCG_at_10",
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
            rows,
            "gSASRec",
            "GRU4Rec",
            "test_NDCG_at_10",
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
            rows,
            "gSASRec",
            "GRU4Rec",
            "test_NDCG_at_10",
            expected_seeds={42},
        )


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


def test_relative_improvement_basic():
    assert relative_improvement(0.30, 0.25) == pytest.approx(0.20)


def test_relative_improvement_rejects_zero_baseline():
    assert relative_improvement(0.30, 0.0) is None


def test_index_by_seed_rejects_duplicate():
    rows = [
        {"model": "X", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "X", "seed": "42", "test_NDCG_at_10": "0.31"},
    ]
    with pytest.raises(RuntimeError, match="Duplicate seed"):
        _index_by_seed(rows, "X", "test_NDCG_at_10")
