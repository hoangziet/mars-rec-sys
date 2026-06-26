import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.gsasrec import GSASRec


def _make_model(*, n_items: int = 100, t: float = 0.5) -> GSASRec:
    torch.manual_seed(0)
    return GSASRec(n_items=n_items, max_len=4, hidden_dim=16, num_heads=2, num_layers=1, dropout=0.0, t=t, num_neg=4).eval()


def test_gbce_sampling_rate_matches_k_over_n_minus_one():
    model = _make_model(n_items=101)
    assert model._negative_sampling_rate(5) == 5 / 100


def test_gbce_t_zero_preserves_positive_logits():
    model = _make_model(t=0.0)
    scores = torch.tensor([-2.0, 0.0, 2.0])
    transformed = model._gbce_transform_pos(scores, num_negatives=4)
    assert torch.allclose(transformed, scores, atol=1e-6, rtol=1e-6)


def test_gsasrec_returns_per_position_loss():
    model = _make_model()
    input_seq = torch.tensor([[0, 1, 2, 3]])
    positives = torch.tensor([[0, 2, 3, 4]])
    negatives = torch.tensor([[[0, 0, 0, 0], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]]])
    loss_mask = torch.tensor([[False, True, True, True]])

    per_position = model.sequence_loss(input_seq, positives, negatives, loss_mask, reduction="none")
    assert per_position.shape == (1, 4)
    assert per_position[0, 0].item() == 0.0
    assert torch.all(per_position[0, 1:] > 0)
