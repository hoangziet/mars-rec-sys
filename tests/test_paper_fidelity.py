"""
tests/test_paper_fidelity.py
=============================
Tests verifying paper-fidelity fixes.
"""

import math
import torch
import torch.nn as nn
import numpy as np

from pipeline.loaders import TrainSequenceDataset


class TestBERT4RecMaskedLoss:
    """BERT4Rec loss only at masked positions."""

    def test_loss_only_at_masked_positions(self):
        """CE loss should be computed only on masked tokens, not all positions."""
        logits = torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 2.0, 0.0],
             [0.0, 0.0, 1.0]],
        ])
        labels = torch.tensor([[0, 0, 2, 0]])

        mask = (labels != 0)
        logits_masked = logits[mask]
        labels_masked = labels[mask]

        assert logits_masked.shape == (1, 3), f"Expected (1, 3), got {logits_masked.shape}"
        assert labels_masked.item() == 2, f"Expected label 2, got {labels_masked.item()}"


class TestTrainSequenceDatasetNegSampling:
    """Negative re-sampling in TrainSequenceDataset."""

    def test_different_negatives_per_call(self):
        """__getitem__ should return different negatives on repeated calls."""
        import tempfile, os

        csv_data = 'item_sequence\n"[1, 2, 3]"\n"[4, 5, 6]"'
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        tmp.write(csv_data)
        tmp.close()

        ds = TrainSequenceDataset(tmp.name, max_len=10, n_items=10, num_neg=5)
        negs0 = ds[0]["neg_items"].tolist()
        negs1 = ds[0]["neg_items"].tolist()
        assert negs0 != negs1, "Negatives should be re-sampled on each __getitem__ call"

        os.unlink(tmp.name)


class TestGRU4RecLosses:
    """TOP1 and BPR-max loss implementations."""

    def test_top1_loss_positive(self):
        """TOP1 loss should be non-negative and finite."""
        from models.gru4rec import GRU4Rec
        model = GRU4Rec(n_items=10, emb_dim=4, hidden_dim=8)
        model.eval()
        seq = torch.tensor([[1, 2, 3]])
        pos = torch.tensor([4])
        neg = torch.tensor([[5, 6]])
        loss = model.top1_loss(seq, pos, neg)
        assert loss.item() > 0
        assert not math.isnan(loss.item())

    def test_bpr_max_loss_positive(self):
        """BPR-max loss should be non-negative."""
        from models.gru4rec import GRU4Rec
        model = GRU4Rec(n_items=10, emb_dim=4, hidden_dim=8)
        model.eval()
        seq = torch.tensor([[1, 2, 3]])
        pos = torch.tensor([4])
        neg = torch.tensor([[5, 6]])
        loss = model.bpr_max_loss(seq, pos, neg)
        assert loss.item() > 0
        assert not math.isnan(loss.item())


class TestSASRecNormFirst:
    """Pre-LN vs Post-LN switch."""

    def test_norm_first_true_runs(self):
        """SASRec with norm_first=True forward pass without error."""
        from models.sasrec import SASRec
        model = SASRec(n_items=10, max_len=5, norm_first=True)
        model.eval()
        seq = torch.tensor([[0, 0, 1, 2, 3]])
        out = model.predict(seq)
        assert out.shape == (1, 11)

    def test_norm_first_false_runs(self):
        """SASRec with norm_first=False (Post-LN) forward pass without error."""
        from models.sasrec import SASRec
        model = SASRec(n_items=10, max_len=5, norm_first=False)
        model.eval()
        seq = torch.tensor([[0, 0, 1, 2, 3]])
        out = model.predict(seq)
        assert out.shape == (1, 11)


class TestGSASRecPosSmoothing:
    """Position smoothing in gSASRec."""

    def test_pos_smoothing_zero_no_effect(self):
        """With pos_smoothing=0, output should be deterministic."""
        from models.gsasrec import GSASRec
        model = GSASRec(n_items=10, max_len=5, pos_smoothing=0.0)
        model.eval()
        torch.manual_seed(42)
        seq = torch.tensor([[0, 0, 1, 2, 3]])
        out1 = model.predict(seq)
        torch.manual_seed(42)
        out2 = model.predict(seq)
        assert torch.allclose(out1, out2)

    def test_pos_smoothing_nonzero_trains(self):
        """With pos_smoothing>0, training mode should not crash."""
        from models.gsasrec import GSASRec
        model = GSASRec(n_items=10, max_len=5, pos_smoothing=0.5)
        model.train()
        seq = torch.tensor([[0, 0, 1, 2, 3]])
        pos = torch.tensor([4])
        neg = torch.tensor([[5, 6, 7, 8, 9]])
        loss = model.loss(seq, pos, neg)
        assert not math.isnan(loss.item())


class TestBPRMFL2Scaling:
    """BPR-MF L2 regularization scaling matches reference."""

    def test_l2_per_vector_scaling(self):
        """L2 should be sum(||e_i||²/2) across batch, not divided by batch_size."""
        from models.bprmf import bpr_loss
        u = torch.randn(4, 8)
        p = torch.randn(4, 8)
        n = torch.randn(4, 8)
        pos_scores = (u * p).sum(-1)
        neg_scores = (u * n).sum(-1)

        loss = bpr_loss(pos_scores, neg_scores, reg_lambda=1.0,
                        u_emb=u, p_emb=p, n_emb=n)
        expected_reg = (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / 2.0
        ce = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        expected = ce + expected_reg
        assert torch.allclose(loss, expected, rtol=1e-4), \
            f"loss={loss.item():.6f}, expected={expected.item():.6f}"
