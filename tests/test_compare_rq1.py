import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_rq1 import pair_runs_by_seed, relative_improvement, select_top_two_models, _require_min_pairs


def test_pair_runs_by_seed_matches_winner_and_runner_up():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
    ]
    assert pair_runs_by_seed(rows, winner="gSASRec", runner_up="GRU4Rec", expected_pairs=1) == [(42, 0.30, 0.28)]


def test_pair_runs_by_seed_rejects_missing_seed():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "123", "test_NDCG_at_10": "0.28"},
    ]
    with pytest.raises(RuntimeError, match="Seed mismatch"):
        pair_runs_by_seed(rows, winner="gSASRec", runner_up="GRU4Rec", expected_pairs=1)


def test_pair_runs_by_seed_rejects_duplicate_seed():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.31"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
    ]
    with pytest.raises(RuntimeError, match="Duplicate seed"):
        pair_runs_by_seed(rows, winner="gSASRec", runner_up="GRU4Rec", expected_pairs=1)


def test_pair_runs_by_seed_rejects_single_pair_when_expected_two():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.28"},
    ]
    with pytest.raises(RuntimeError, match="Expected exactly 2 paired seeds"):
        pair_runs_by_seed(rows, winner="gSASRec", runner_up="GRU4Rec", expected_pairs=2)


def test_relative_improvement_rejects_zero_runner_mean():
    with pytest.raises(ValueError, match="Runner-up mean must be non-zero"):
        relative_improvement(0.30, 0.0)


def test_ranked_models_are_selected_by_validation_rank():
    summary_rows = [
        {"model": "GRU4Rec", "validation_rank": 2},
        {"model": "gSASRec", "validation_rank": 1},
    ]
    assert select_top_two_models(summary_rows) == ("gSASRec", "GRU4Rec")


def test_zero_differences_are_handled_without_nan():
    rows = [
        {"model": "gSASRec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "GRU4Rec", "seed": "42", "test_NDCG_at_10": "0.30"},
        {"model": "gSASRec", "seed": "123", "test_NDCG_at_10": "0.28"},
        {"model": "GRU4Rec", "seed": "123", "test_NDCG_at_10": "0.28"},
    ]
    assert pair_runs_by_seed(rows, winner="gSASRec", runner_up="GRU4Rec", expected_pairs=2) == [
        (42, 0.30, 0.30),
        (123, 0.28, 0.28),
    ]


def test_require_min_pairs_rejects_single_pair():
    with pytest.raises(ValueError, match="at least two paired seeds"):
        _require_min_pairs(1)
