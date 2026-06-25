import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bert4rec import BERT4Rec
from pipeline.optim import build_optimizer, build_scheduler


def test_bert4rec_uses_common_adam_optimizer_under_fairness_layer():
    model = BERT4Rec(n_items=32, max_len=50, hidden_dim=16, num_heads=2, num_layers=1, dropout=0.1)
    optimizer = build_optimizer(
        "bert4rec",
        model,
        {"lr": 1e-3, "beta2": 0.98, "weight_decay": 1e-4},
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert optimizer.defaults["weight_decay"] == 1e-4


def test_bert4rec_uses_no_scheduler_when_warmup_is_disabled():
    model = BERT4Rec(n_items=32, max_len=50, hidden_dim=16, num_heads=2, num_layers=1, dropout=0.1)
    optimizer = build_optimizer(
        "bert4rec",
        model,
        {"lr": 1e-3, "beta2": 0.98, "weight_decay": 1e-4},
    )

    scheduler = build_scheduler(
        optimizer,
        {"epochs": 50, "lr": 1e-3, "warmup_steps": 0},
        num_train_batches=10,
    )

    assert scheduler is None
