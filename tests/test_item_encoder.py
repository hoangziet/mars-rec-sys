import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.item_encoder import ItemEncoder


def _make_metadata_tensors(n_items=50, hidden_dim=32):
    return {
        "language": torch.randint(0, 5, (n_items + 1,)),
        "difficulty": torch.randint(0, 4, (n_items + 1,)),
        "theme": torch.randint(0, 8, (n_items + 1, 3)),
        "software": torch.randint(0, 6, (n_items + 1, 2)),
        "job": torch.randint(0, 3, (n_items + 1, 2)),
        "type": torch.randint(0, 4, (n_items + 1, 2)),
        "duration": torch.randn(n_items + 1),
    }


def test_item_encoder_output_shape():
    meta = _make_metadata_tensors()
    text_emb = torch.randn(51, 768)
    enc = ItemEncoder(n_items=50, hidden_dim=32, metadata_tensors=meta,
                      text_embeddings=text_emb, use_structured=True, use_text=True)
    item_ids = torch.randint(1, 51, (4,))
    out = enc(item_ids)
    assert out.shape == (4, 32)


def test_item_encoder_id_only():
    meta = _make_metadata_tensors()
    enc = ItemEncoder(n_items=50, hidden_dim=32, metadata_tensors=meta,
                      text_embeddings=None, use_structured=False, use_text=False)
    item_ids = torch.randint(1, 51, (4,))
    out = enc(item_ids)
    assert out.shape == (4, 32)


def test_item_encoder_padding_returns_zero():
    meta = _make_metadata_tensors()
    enc = ItemEncoder(n_items=50, hidden_dim=32, metadata_tensors=meta,
                      text_embeddings=None, use_structured=False, use_text=False)
    out = enc(torch.tensor([0]))
    assert torch.all(out == 0)
