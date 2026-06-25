import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gru4rec import GRU4Rec
from pipeline.builder import build_criterion_fn


def test_gru4rec_criterion_uses_requested_loss_type_without_stale_cache(monkeypatch):
    model = GRU4Rec(n_items=10)
    top1_calls = []
    bpr_max_calls = []

    def fake_top1(input_seq, pos_items, neg_items):
        top1_calls.append((input_seq.shape, pos_items.shape, neg_items.shape))
        return torch.tensor(1.0)

    def fake_bpr_max(input_seq, pos_items, neg_items):
        bpr_max_calls.append((input_seq.shape, pos_items.shape, neg_items.shape))
        return torch.tensor(2.0)

    monkeypatch.setattr(model, "top1_loss", fake_top1)
    monkeypatch.setattr(model, "bpr_max_loss", fake_bpr_max)

    batch = {
        "input_seq": torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long),
        "pos_items": torch.tensor([3, 5], dtype=torch.long),
        "neg_items": torch.tensor([[6, 7], [8, 9]], dtype=torch.long),
    }

    top1_fn = build_criterion_fn("gru4rec", {"loss_type": "top1"})
    bpr_max_fn = build_criterion_fn("gru4rec", {"loss_type": "bpr_max"})

    assert top1_fn(model, batch, "cpu").item() == 1.0
    assert bpr_max_fn(model, batch, "cpu").item() == 2.0
    assert len(top1_calls) == 1
    assert len(bpr_max_calls) == 1
