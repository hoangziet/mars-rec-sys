import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from models.bert4rec import BERT4Rec


def test_bert4rec_weighted_loss_uses_engagement_weights():
    model = BERT4Rec(
        n_items=20,
        max_len=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        watch_mode="loss",
        watch_alpha=1.0,
        watch_num_bins=5,
        dropout=0.0,
    )
    input_seq = torch.tensor([[1, 2, 21, 4]])
    labels = torch.tensor([[0, 0, 3, 0]])
    engagement = torch.tensor([[0.0, 0.0, 0.8, 0.0]])
    watch_input_ids = torch.tensor([[2, 3, 1, 6]])
    loss = model.loss(input_seq, labels, engagement=engagement, watch_input_ids=watch_input_ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bert4rec_accepts_watch_embedding_inputs():
    model = BERT4Rec(
        n_items=20,
        max_len=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        watch_mode="embedding",
        watch_num_bins=5,
        dropout=0.0,
    )
    logits = model(
        torch.tensor([[1, 2, 21, 4]]),
        watch_input_ids=torch.tensor([[2, 3, 1, 6]]),
    )
    assert logits.shape == (1, 4, 21)


def test_bert4rec_forward_with_no_watch_mode_unchanged():
    model = BERT4Rec(n_items=20, max_len=4, hidden_dim=8, num_heads=2, num_layers=1, dropout=0.0)
    logits = model(torch.tensor([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 21)
