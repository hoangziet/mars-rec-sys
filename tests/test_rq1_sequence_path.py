import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from pipeline.builder import build_criterion_fn, build_rq1_train_criterion_fn
from pipeline.confidence import WeightedCriterionFn


class _DummySASRec:
    def sequence_loss(self, input_seq, pos_items, neg_items, loss_mask, reduction="mean"):
        assert input_seq.shape == (1, 4)
        assert pos_items.shape == (1, 4)
        assert neg_items.shape == (1, 4, 1)
        assert loss_mask.shape == (1, 4)
        return torch.tensor(1.0)


class _DummyGSASRec:
    def loss(self, input_seq, pos_items, neg_items, reduction="none"):
        assert input_seq.shape == (1, 4)
        assert pos_items.shape == (1,)
        assert neg_items.shape == (1,)
        return torch.tensor([2.0])

    def sequence_loss(self, input_seq, pos_items, neg_items, loss_mask, reduction="mean"):
        assert input_seq.shape == (1, 4)
        assert pos_items.shape == (1, 4)
        assert neg_items.shape == (1, 4, 4)
        assert loss_mask.shape == (1, 4)
        return torch.tensor(3.0)


def test_rq1_sequence_criterion_accepts_shifted_batch():
    fn = build_rq1_train_criterion_fn("sasrec", {})
    batch = {
        "input_seq": torch.ones(1, 4, dtype=torch.long),
        "pos_items": torch.ones(1, 4, dtype=torch.long),
        "neg_items": torch.ones(1, 4, 1, dtype=torch.long),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    loss = fn(_DummySASRec(), batch, "cpu")
    assert loss.item() == 1.0


def test_val_criterion_accepts_scalar_batch_after_sequence_rewrite():
    fn = build_criterion_fn("gsasrec", {"confidence_alpha": 0.0})
    batch = {
        "input_seq": torch.ones(1, 4, dtype=torch.long),
        "pos_items": torch.ones(1, dtype=torch.long),
        "neg_items": torch.ones(1, dtype=torch.long),
    }
    loss = fn(_DummyGSASRec(), batch, "cpu")
    assert loss.item() == 2.0


def test_gsasrec_confidence_weighting_remains_per_sample():
    fn = build_criterion_fn("gsasrec", {"confidence_alpha": 2.0})
    assert isinstance(fn, WeightedCriterionFn)

    batch = {
        "input_seq": torch.ones(1, 4, dtype=torch.long),
        "pos_items": torch.ones(1, dtype=torch.long),
        "neg_items": torch.ones(1, dtype=torch.long),
        "engagement": torch.tensor([0.5]),
    }
    loss = fn(_DummyGSASRec(), batch, "cpu")
    assert torch.isfinite(loss)
    assert loss.dim() == 0
