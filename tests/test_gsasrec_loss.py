import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gsasrec import GSASRec


def _make_model():
    return GSASRec(n_items=100, max_len=10, hidden_dim=32, num_heads=2, num_layers=1, t=0.5, num_neg=4)


def test_loss_default_returns_scalar():
    model = _make_model()
    input_seq = torch.randint(1, 101, (4, 10))
    pos_items = torch.randint(1, 101, (4,))
    neg_items = torch.randint(1, 101, (4, 4))
    loss = model.loss(input_seq, pos_items, neg_items)
    assert loss.dim() == 0


def test_loss_reduction_none_returns_per_sample():
    model = _make_model()
    input_seq = torch.randint(1, 101, (4, 10))
    pos_items = torch.randint(1, 101, (4,))
    neg_items = torch.randint(1, 101, (4, 4))
    loss = model.loss(input_seq, pos_items, neg_items, reduction="none")
    assert loss.shape == (4,)


def test_loss_reduction_mean_matches_default():
    model = _make_model().eval()
    torch.manual_seed(42)
    input_seq = torch.randint(1, 101, (4, 10))
    pos_items = torch.randint(1, 101, (4,))
    neg_items = torch.randint(1, 101, (4, 4))
    default_loss = model.loss(input_seq, pos_items, neg_items)
    per_sample = model.loss(input_seq, pos_items, neg_items, reduction="none")
    assert torch.allclose(default_loss, per_sample.mean())


def test_pos_smoothing_does_not_overflow_position_embeddings():
    model = GSASRec(
        n_items=50,
        max_len=5,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        t=0.5,
        num_neg=4,
        pos_smoothing=100.0,
    ).train()
    input_seq = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    out = model(input_seq)

    assert out.shape == (1, 51)
