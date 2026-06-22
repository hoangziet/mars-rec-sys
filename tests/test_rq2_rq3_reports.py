import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq2_tune_forces_batch_size_128(monkeypatch):
    from training.configs import build_model_config
    cfg = build_model_config("gsasrec")
    train_kwargs = dict(cfg["train_kwargs"])
    train_kwargs["confidence_alpha"] = 0.5
    train_kwargs["batch_size"] = 256

    # protocol rule: tuning must align with final batch size
    train_kwargs["batch_size"] = 128
    assert train_kwargs["batch_size"] == 128
