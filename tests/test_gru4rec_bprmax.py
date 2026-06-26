import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from models.gru4rec import GRU4Rec


def _make_model() -> GRU4Rec:
    torch.manual_seed(0)
    return GRU4Rec(n_items=20, emb_dim=8, hidden_dim=8, num_layers=1, dropout=0.0).eval()


def test_gru_returns_hidden_for_every_timestep():
    model = _make_model()
    input_seq = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])
    hidden = model._encode_sequence(input_seq)
    assert hidden.shape == (2, 4, 8)
    assert torch.all(hidden[0, 0] == 0)
    assert torch.all(hidden[1, :2] == 0)


def test_bpr_max_matches_reference_formula():
    model = _make_model()
    input_seq = torch.tensor([[0, 1, 2, 3]])
    positives = torch.tensor([[0, 2, 3, 4]])
    negatives = torch.tensor([[[0, 0], [5, 6], [6, 7], [7, 8]]])
    loss_mask = positives.ne(0)
    bpreg = 0.5
    elu_param = 0.5

    with torch.no_grad():
        hidden = model._encode_sequence(input_seq)
        positive_score = (hidden * model.item_emb(positives)).sum(dim=-1) + model.item_bias(positives).squeeze(-1)
        negative_score = torch.einsum("bld,blkd->blk", hidden, model.item_emb(negatives)) + model.item_bias(negatives).squeeze(-1)
        positive_score = F.elu(positive_score, alpha=elu_param)
        negative_score = F.elu(negative_score, alpha=elu_param)
        weights = torch.softmax(negative_score, dim=-1)
        pair_probability = torch.sigmoid(positive_score.unsqueeze(-1) - negative_score)
        ranking = -torch.log((weights * pair_probability).sum(dim=-1) + 1e-24)
        regularization = bpreg * (weights * negative_score.pow(2)).sum(dim=-1)
        expected = (ranking + regularization)[loss_mask].mean()

    actual = model.sequence_bpr_max_loss(input_seq, positives, negatives, loss_mask, bpreg=bpreg, elu_param=elu_param)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
