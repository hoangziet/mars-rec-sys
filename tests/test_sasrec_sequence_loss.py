import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from models.sasrec import SASRec


def _make_model() -> SASRec:
    torch.manual_seed(0)
    return SASRec(n_items=20, max_len=4, hidden_dim=8, num_heads=2, num_layers=1, dropout=0.0).eval()


def test_sasrec_returns_per_position_loss():
    model = _make_model()
    input_seq = torch.tensor([[0, 1, 2, 3]])
    positives = torch.tensor([[0, 2, 3, 4]])
    negatives = torch.tensor([[[0], [5], [6], [7]]])
    loss_mask = torch.tensor([[False, True, True, True]])

    per_position = model.sequence_loss(input_seq, positives, negatives, loss_mask, reduction="none")

    assert per_position.shape == (1, 4)
    assert per_position[0, 0].item() == 0.0
    assert torch.all(per_position[0, 1:] > 0)


def test_sasrec_mean_matches_reference_formula():
    model = _make_model()
    input_seq = torch.tensor([[0, 1, 2, 3]])
    positives = torch.tensor([[0, 2, 3, 4]])
    negatives = torch.tensor([[[0], [5], [6], [7]]])
    loss_mask = torch.tensor([[False, True, True, True]])

    with torch.no_grad():
        hidden = model._encode(input_seq)
        positive_logits = (hidden * model.item_emb(positives)).sum(dim=-1)
        negative_logits = (hidden.unsqueeze(2) * model.item_emb(negatives)).sum(dim=-1).squeeze(-1)
        expected = (
            F.binary_cross_entropy_with_logits(positive_logits[loss_mask], torch.ones_like(positive_logits[loss_mask]))
            + F.binary_cross_entropy_with_logits(negative_logits[loss_mask], torch.zeros_like(negative_logits[loss_mask]))
        )

    actual = model.sequence_loss(input_seq, positives, negatives, loss_mask)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
