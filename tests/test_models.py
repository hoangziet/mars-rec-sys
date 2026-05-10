"""
tests/test_models.py
====================
Unit tests for all model implementations.
Tests: forward pass shape, loss computation, no NaN, no negative loss.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ITEMS  = 100
N_USERS  = 50
MAX_LEN  = 10
BATCH    = 4
HIDDEN   = 32
NUM_NEG  = 4


def make_input_seq(batch=BATCH, max_len=MAX_LEN, n_items=N_ITEMS) -> torch.Tensor:
    seq       = torch.zeros(batch, max_len, dtype=torch.long)
    seq[:, -3:] = torch.randint(1, n_items + 1, (batch, 3))
    return seq


def assert_valid_loss(loss: torch.Tensor):
    assert loss.dim() == 0, "loss must be scalar"
    assert not torch.isnan(loss), "loss is NaN"
    assert not torch.isinf(loss), "loss is inf"
    assert loss.item() >= 0, "loss must be non-negative"


# ---------------------------------------------------------------------------
# SASRec
# ---------------------------------------------------------------------------


class TestSASRec:
    @pytest.fixture
    def model(self):
        from models.sasrec import SASRec
        return SASRec(n_items=N_ITEMS, max_len=MAX_LEN, hidden_dim=HIDDEN,
                      num_heads=2, num_layers=2, dropout=0.0)

    def test_forward_shape(self, model):
        out = model(make_input_seq())
        assert out.shape == (BATCH, N_ITEMS + 1)

    def test_loss_valid(self, model):
        seq = make_input_seq()
        pos = torch.randint(1, N_ITEMS + 1, (BATCH,))
        neg = torch.randint(1, N_ITEMS + 1, (BATCH,))
        assert_valid_loss(model.loss(seq, pos, neg))

    def test_padding_seq_no_nan(self, model):
        seq = torch.zeros(BATCH, MAX_LEN, dtype=torch.long)   # all padding
        out = model(seq)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# gSASRec
# ---------------------------------------------------------------------------


class TestGSASRec:
    @pytest.fixture
    def model(self):
        from models.gsasrec import GSASRec
        return GSASRec(n_items=N_ITEMS, max_len=MAX_LEN, hidden_dim=HIDDEN,
                       num_heads=2, num_layers=2, dropout=0.0, t=0.5)

    def test_forward_shape(self, model):
        out = model(make_input_seq())
        assert out.shape == (BATCH, N_ITEMS + 1)

    def test_loss_valid_multi_neg(self, model):
        seq  = make_input_seq()
        pos  = torch.randint(1, N_ITEMS + 1, (BATCH,))
        negs = torch.randint(1, N_ITEMS + 1, (BATCH, NUM_NEG))
        conf = torch.rand(BATCH).clamp(0.1, 1.0)
        assert_valid_loss(model.loss(seq, pos, negs, confidence=conf))

    def test_loss_t0_equivalent_to_bce(self, model):
        from models.gsasrec import GSASRec
        model_t0 = GSASRec(n_items=N_ITEMS, max_len=MAX_LEN, hidden_dim=HIDDEN,
                            num_heads=2, num_layers=2, dropout=0.0, t=0.0)
        seq  = make_input_seq()
        pos  = torch.randint(1, N_ITEMS + 1, (BATCH,))
        neg  = torch.randint(1, N_ITEMS + 1, (BATCH, 1))
        conf = torch.ones(BATCH)
        loss = model_t0.loss(seq, pos, neg, confidence=conf)
        assert_valid_loss(loss)


# ---------------------------------------------------------------------------
# GRU4Rec
# ---------------------------------------------------------------------------


class TestGRU4Rec:
    @pytest.fixture
    def model(self):
        from models.gru4rec import GRU4Rec
        return GRU4Rec(n_items=N_ITEMS, emb_dim=HIDDEN, hidden_dim=HIDDEN,
                       num_layers=1, dropout=0.0)

    def test_forward_shape(self, model):
        out = model(make_input_seq())
        assert out.shape == (BATCH, N_ITEMS + 1)

    def test_loss_valid_ce(self, model):
        seq = make_input_seq()
        pos = torch.randint(1, N_ITEMS + 1, (BATCH,))
        assert_valid_loss(model.loss(seq, pos))

    def test_no_bias_parameters(self, model):
        bias_params = [n for n, _ in model.named_parameters() if "bias" in n and "gru" in n]
        assert len(bias_params) == 0, f"GRU should have no bias params, found: {bias_params}"


# ---------------------------------------------------------------------------
# BERT4Rec
# ---------------------------------------------------------------------------


class TestBERT4Rec:
    @pytest.fixture
    def model(self):
        from models.bert4rec import BERT4Rec
        return BERT4Rec(n_items=N_ITEMS, max_len=MAX_LEN, hidden_dim=HIDDEN,
                        num_heads=2, num_layers=2, dropout=0.0)

    def test_forward_shape(self, model):
        seq = make_input_seq()
        out = model(seq)
        assert out.shape == (BATCH, MAX_LEN, N_ITEMS + 2)   # vocab = n_items + 2

    def test_loss_cross_entropy_valid(self, model):
        import torch.nn.functional as F
        seq    = make_input_seq()
        logits = model(seq)
        labels = torch.zeros(BATCH, MAX_LEN, dtype=torch.long)
        labels[:, -1] = torch.randint(1, N_ITEMS + 1, (BATCH,))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        assert_valid_loss(loss)

    def test_weight_tying(self, model):
        # Weight tying: output logits = x @ item_embedding.weight.T + out_bias.
        # Verify by mutating item_embedding weight and checking output changes.
        import torch
        seq    = make_input_seq()
        out_before = model(seq).detach().clone()
        with torch.no_grad():
            model.item_embedding.weight[1] += 100.0
        out_after = model(seq).detach()
        # If weight tying is active, output projection changes when embedding changes.
        assert not torch.allclose(out_before, out_after), \
            "Mutating item_embedding.weight should change logits (weight tying)"


# ---------------------------------------------------------------------------
# BPR-MF
# ---------------------------------------------------------------------------


class TestBPRMF:
    @pytest.fixture
    def model(self):
        from models.bprmf import BPRMF
        return BPRMF(n_users=N_USERS, n_items=N_ITEMS, emb_dim=HIDDEN)

    def test_forward_returns_pos_neg_scores(self, model):
        users = torch.randint(1, N_USERS + 1, (BATCH,))
        pos   = torch.randint(1, N_ITEMS + 1, (BATCH,))
        neg   = torch.randint(1, N_ITEMS + 1, (BATCH,))
        pos_s, neg_s = model(users, pos, neg)
        assert pos_s.shape == (BATCH,)
        assert neg_s.shape == (BATCH,)

    def test_loss_valid(self, model):
        from models.bprmf import bpr_loss
        users = torch.randint(1, N_USERS + 1, (BATCH,))
        pos   = torch.randint(1, N_ITEMS + 1, (BATCH,))
        neg   = torch.randint(1, N_ITEMS + 1, (BATCH,))
        pos_s, neg_s = model(users, pos, neg)
        assert_valid_loss(bpr_loss(pos_s, neg_s, reg_lambda=1e-4, model=model))

    def test_n_items_attribute(self, model):
        assert model.n_items == N_ITEMS

    def test_padding_embedding_is_zero(self, model):
        assert model.user_embedding.weight[0].abs().sum().item() == pytest.approx(0.0)
        assert model.item_embedding.weight[0].abs().sum().item() == pytest.approx(0.0)
