import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq4_compare_rejects_user_missing_some_seeds():
    import pandas as pd
    from scripts import rq4_compare

    per_user = pd.DataFrame(
        [
            {"variant": "V0", "seed": 42, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.2},
            {"variant": "V0", "seed": 123, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.3},
            {"variant": "V1", "seed": 42, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.25},
            # V1 missing seed 123 for same user
        ]
    )
    with pytest.raises(RuntimeError, match="expected all users to have"):
        rq4_compare._join_by_user(per_user, "V1", "V0", "ndcg_at_10", expected_seed_count=2)


def test_evaluate_bert4rec_detailed_returns_per_user_rows():
    import torch
    from pipeline.metrics import evaluate_bert4rec_detailed

    class _DummyBERT:
        n_items = 5
        mask_token = 6
        watch_mode = "none"

        def eval(self):
            return self

        def __call__(self, input_seq, watch_input_ids=None):
            batch, seq_len = input_seq.shape
            logits = torch.zeros(batch, seq_len, self.n_items + 1)
            # make target item 3 rank first at the last position
            logits[:, -1, 3] = 1.0
            return logits

    batch = {
        "input_seq": torch.tensor([[1, 2, 3, 4]]),
        "history_mask": torch.tensor([[True, True, True, False, False, False]]),
        "target": torch.tensor([3]),
        "user": torch.tensor([7]),
        "watch_input_ids": torch.tensor([[0, 0, 0, 0]]),
    }

    metrics, rows = evaluate_bert4rec_detailed(_DummyBERT(), [batch], device=torch.device("cpu"))
    assert rows[0]["user_idx"] == 7
    assert rows[0]["target_item"] == 3
    assert "ndcg_at_10" in rows[0]
    assert metrics["Recall@10"] == 1.0
