import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metrics import compute_metrics_from_ranks


def test_compute_metrics_from_ranks_does_not_round_results():
    metrics = compute_metrics_from_ranks([1, 10, 11], k_list=(10,))

    assert metrics["Recall@10"] == pytest.approx(2 / 3)
    expected_ndcg = (1.0 + (1.0 / 3.4594316186372973) + 0.0) / 3
    assert metrics["NDCG@10"] == pytest.approx(expected_ndcg)
