import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import builder


def test_build_model_does_not_mutate_model_kwargs():
    """build_model() should not pop or alter the caller-supplied dict."""
    n_items = 100
    n_users = 10
    model_kwargs = {
        "hidden_dim": 32,
        "item_encoder": {
            "use_structured": True,
            "use_text": False,
            "metadata_vocab_path": "data/processed/item_features/metadata_vocab.json",
            "metadata_csv_path": "data/processed/item_features/item_metadata.csv",
            "text_emb_path": "data/processed/item_features/text_embeddings.pt",
        },
    }
    snapshot = copy.deepcopy(model_kwargs)

    try:
        builder.build_model("gsasrec", n_items, n_users, model_kwargs, max_len=50)
    except Exception:
        # We don't care if the build itself succeeds (missing files, etc.) —
        # only that the caller's dict is untouched.
        pass

    assert model_kwargs == snapshot
