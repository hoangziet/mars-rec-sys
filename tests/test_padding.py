import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gsasrec import GSASRec
from models.sasrec import SASRec


def test_gsasrec_padding_embedding_is_zero():
    model = GSASRec(n_items=100, max_len=10, hidden_dim=32, num_heads=2, num_layers=1)
    assert torch.all(model.item_emb.weight[0] == 0)
    assert torch.all(model.pos_emb.weight[0] == 0)


def test_sasrec_padding_embedding_is_zero():
    model = SASRec(n_items=100, max_len=10, hidden_dim=32, num_heads=2, num_layers=1)
    assert torch.all(model.item_emb.weight[0] == 0)
    assert torch.all(model.pos_emb.weight[0] == 0)
