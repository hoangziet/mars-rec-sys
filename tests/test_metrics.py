"""
tests/test_metrics.py
=====================
Unit tests for pipeline/metrics.py.
"""

import math

import numpy as np
import pytest
import torch

from pipeline.metrics import compute_metrics_from_ranks, _ranks_from_logits


class TestComputeMetricsFromRanks:
    def test_perfect_rank_hr(self):
        assert compute_metrics_from_ranks([1] * 50)["Recall@10"] == 1.0

    def test_perfect_rank_ndcg(self):
        assert compute_metrics_from_ranks([1] * 50)["NDCG@10"] == 1.0

    def test_worst_rank_hr(self):
        assert compute_metrics_from_ranks([9999] * 50)["Recall@10"] == 0.0

    def test_worst_rank_ndcg(self):
        assert compute_metrics_from_ranks([9999] * 50)["NDCG@10"] == 0.0

    def test_rank_at_k_boundary_hr(self):
        # Rank 10 = exactly at boundary → should hit Recall@10
        r = compute_metrics_from_ranks([10])
        assert r["Recall@10"] == 1.0
        assert r["Recall@20"] == 1.0

    def test_rank_just_outside_k_hr(self):
        r = compute_metrics_from_ranks([11])
        assert r["Recall@10"] == 0.0
        assert r["Recall@20"] == 1.0

    def test_ndcg_formula(self):
        rank = 3
        expected = 1.0 / math.log2(rank + 1)
        r = compute_metrics_from_ranks([rank])
        assert abs(r["NDCG@10"] - expected) < 1e-6

    def test_both_k_values_present(self):
        r = compute_metrics_from_ranks([5])
        assert "Recall@10" in r and "Recall@20" in r
        assert "NDCG@10" in r and "NDCG@20" in r

    def test_average_over_users(self):
        # 50% hit at rank 1, 50% miss at rank 9999
        r = compute_metrics_from_ranks([1, 9999])
        assert r["Recall@10"] == pytest.approx(0.5)

    def test_custom_k_list(self):
        r = compute_metrics_from_ranks([5], k_list=(5, 50))
        assert "Recall@5" in r and "Recall@50" in r
        assert r["Recall@5"] == 1.0


class TestRanksFromLogits:
    def test_target_ranked_first(self):
        logits       = torch.zeros(2, 10)
        logits[0, 3] = 100.0   # target
        logits[1, 7] = 100.0

        history_mask = torch.zeros(2, 10, dtype=torch.bool)
        history_mask[:, 0] = True   # mask padding

        target = torch.tensor([3, 7])
        ranks  = _ranks_from_logits(logits, history_mask, target)
        assert ranks == [1, 1]

    def test_target_ranked_last(self):
        logits = torch.zeros(2, 5)
        logits[:, 1] = 10.0
        logits[:, 2] = 10.0
        logits[:, 3] = 10.0

        history_mask = torch.zeros(2, 5, dtype=torch.bool)
        history_mask[:, 0] = True

        target = torch.tensor([4, 4])   # lowest score among unmasked
        ranks  = _ranks_from_logits(logits, history_mask, target)
        assert ranks[0] == 4  # items 1,2,3 score higher; rank = 4

    def test_history_mask_applied(self):
        logits = torch.zeros(1, 5)
        for i in range(1, 4):
            logits[0, i] = float(i) * 10   # items 1,2,3 all score higher

        history_mask = torch.zeros(1, 5, dtype=torch.bool)
        history_mask[0, 0] = True   # padding
        history_mask[0, 1] = True   # mask item 1 (highest non-target)
        history_mask[0, 2] = True   # mask item 2

        target = torch.tensor([4])  # target has score 0; item 3 scores 30
        ranks  = _ranks_from_logits(logits, history_mask, target)
        # Only item 3 (score=30) and item 4 (score=0) are unmasked.
        # target (item 4) rank = 2.
        assert ranks == [2]

    def test_ties_are_ranked_conservatively(self):
        logits = torch.tensor([[0.0, 5.0, 5.0, 1.0]])
        history_mask = torch.zeros(1, 4, dtype=torch.bool)
        target = torch.tensor([1])

        ranks = _ranks_from_logits(logits, history_mask, target)
        assert ranks == [2]
