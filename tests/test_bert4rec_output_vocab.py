import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from models.bert4rec import BERT4Rec


def _make_model() -> BERT4Rec:
    torch.manual_seed(0)
    return BERT4Rec(n_items=10, max_len=4, hidden_dim=8, num_heads=2, num_layers=1, dropout=0.0).eval()


def test_mask_is_input_only_not_output_class():
    model = _make_model()
    input_seq = torch.tensor([[1, 2, model.mask_token, 0]])
    logits = model(input_seq)
    assert model.mask_token == 11
    assert logits.shape == (1, 4, 11)


def test_padding_embedding_is_zero():
    model = _make_model()
    assert torch.all(model.item_embedding.weight[0] == 0)


def test_bert_loss_uses_real_item_classes_only():
    model = _make_model()
    input_seq = torch.tensor([[1, 2, model.mask_token, 0]])
    labels = torch.tensor([[0, 0, 10, 0]])
    loss = model.loss(input_seq, labels)
    assert torch.isfinite(loss)


def test_mask_token_label_is_rejected():
    model = _make_model()
    input_seq = torch.tensor([[1, 2, model.mask_token, 0]])
    labels = torch.tensor([[0, 0, model.mask_token, 0]])
    with pytest.raises(ValueError, match="real item IDs"):
        model.loss(input_seq, labels)
