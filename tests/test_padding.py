import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gsasrec import GSASRec
from models.sasrec import SASRec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sasrec(n_items=100, num_layers=2):
    return SASRec(n_items=n_items, max_len=10, hidden_dim=32, num_heads=2, num_layers=num_layers)


def _make_gsasrec(n_items=100, num_layers=2):
    return GSASRec(n_items=n_items, max_len=10, hidden_dim=32, num_heads=2, num_layers=num_layers, t=0.5, num_neg=4)


# ---------------------------------------------------------------------------
# Existing tests (preserve)
# ---------------------------------------------------------------------------


def test_gsasrec_padding_embedding_is_zero():
    model = GSASRec(n_items=100, max_len=10, hidden_dim=32, num_heads=2, num_layers=1)
    assert torch.all(model.item_emb.weight[0] == 0)
    assert torch.all(model.pos_emb.weight[0] == 0)


def test_sasrec_padding_embedding_is_zero():
    model = SASRec(n_items=100, max_len=10, hidden_dim=32, num_heads=2, num_layers=1)
    assert torch.all(model.item_emb.weight[0] == 0)
    assert torch.all(model.pos_emb.weight[0] == 0)


# ---------------------------------------------------------------------------
# New tests: full forward with left-padded sequences
# ---------------------------------------------------------------------------


def test_sasrec_encode_finite_with_left_padding():
    """3 valid items at positions 2,3,4: should not produce NaN."""
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
    assert torch.isfinite(enc).all().item(), f"Encode produced NaN/Inf: {enc}"


def test_sasrec_predict_finite_with_left_padding():
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        out = model.predict(seq)
    assert torch.isfinite(out).all().item(), f"Predict produced NaN/Inf: {out}"


def test_sasrec_loss_finite_with_left_padding():
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    pos = torch.tensor([4])
    neg = torch.tensor([5])
    with torch.no_grad():
        loss = model.loss(seq, pos, neg)
    assert torch.isfinite(loss).item(), f"Loss produced NaN/Inf: {loss}"


def test_sasrec_finite_with_single_valid_item():
    """Only 1 valid item at position 4 (heavy left padding)."""
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 0, 0, 1]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_sasrec_finite_with_one_block():
    """Test with num_layers=1 (degenerate case)."""
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=1).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_sasrec_finite_with_three_blocks():
    """Test with num_layers=3 (deeper than default)."""
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=3).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_sasrec_batch_finite():
    """Test with a batch of mixed-length sequences."""
    torch.manual_seed(0)
    model = _make_sasrec(num_layers=2).eval()
    seq = torch.tensor([
        [0, 0, 1, 2, 3],   # 3 valid
        [0, 1, 2, 3, 4],   # 4 valid
        [0, 0, 0, 0, 1],   # 1 valid
    ])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


# gSASRec versions of the same tests


def test_gsasrec_encode_finite_with_left_padding():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
    assert torch.isfinite(enc).all().item()


def test_gsasrec_predict_finite_with_left_padding():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        out = model.predict(seq)
    assert torch.isfinite(out).all().item()


def test_gsasrec_loss_finite_with_left_padding():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    pos = torch.tensor([4])
    neg = torch.tensor([[5, 6, 7, 8]])  # (B, K)
    with torch.no_grad():
        loss = model.loss(seq, pos, neg)
    assert torch.isfinite(loss).item()


def test_gsasrec_finite_with_single_valid_item():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=2).eval()
    seq = torch.tensor([[0, 0, 0, 0, 1]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_gsasrec_finite_with_one_block():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=1).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_gsasrec_finite_with_three_blocks():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=3).eval()
    seq = torch.tensor([[0, 0, 1, 2, 3]])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()


def test_gsasrec_batch_finite():
    torch.manual_seed(0)
    model = _make_gsasrec(num_layers=2).eval()
    seq = torch.tensor([
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 1],
    ])
    with torch.no_grad():
        enc = model._encode(seq)
        h = model._last_hidden(seq)
        out = model.predict(seq)
    assert torch.isfinite(enc).all().item()
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(out).all().item()
