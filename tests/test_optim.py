"""
tests/test_optim.py
===================
Unit tests for optimizer and scheduler helpers.
"""

import pytest
import torch

from pipeline.optim import LinearWarmupDecayScheduler, build_optimizer, build_scheduler


class TestBuildOptimizer:
    def test_bert4rec_uses_adamw(self):
        from models.bert4rec import BERT4Rec

        model = BERT4Rec(n_items=50, max_len=10, hidden_dim=16, num_heads=2, num_layers=1)
        optimizer = build_optimizer(
            "bert4rec",
            model,
            {"lr": 1e-4, "weight_decay": 1e-2, "beta2": 0.999},
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2

    def test_bert4rec_excludes_layer_norm_and_bias_from_weight_decay(self):
        from models.bert4rec import BERT4Rec

        model = BERT4Rec(n_items=50, max_len=10, hidden_dim=16, num_heads=2, num_layers=1)
        optimizer = build_optimizer(
            "bert4rec",
            model,
            {"lr": 1e-4, "weight_decay": 1e-2, "beta2": 0.999},
        )

        weight_decay_by_param_id = {
            id(param): group["weight_decay"]
            for group in optimizer.param_groups
            for param in group["params"]
        }

        named_params = dict(model.named_parameters())
        assert weight_decay_by_param_id[id(named_params["input_ln.weight"])] == pytest.approx(0.0)
        assert weight_decay_by_param_id[id(named_params["input_ln.bias"])] == pytest.approx(0.0)
        assert weight_decay_by_param_id[id(named_params["pred_ln.weight"])] == pytest.approx(0.0)
        assert weight_decay_by_param_id[id(named_params["pred_ffn.bias"])] == pytest.approx(0.0)
        assert weight_decay_by_param_id[id(named_params["pred_ffn.weight"])] == pytest.approx(1e-2)


class TestLinearWarmupDecayScheduler:
    def test_reference_warmup_and_decay_sequence(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([param], lr=0.1)
        scheduler = LinearWarmupDecayScheduler(
            optimizer=optimizer,
            init_lr=0.1,
            num_train_steps=10,
            num_warmup_steps=4,
        )

        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)

        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.075)

        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.06)

    def test_build_scheduler_returns_none_without_warmup(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([param], lr=0.1)

        scheduler = build_scheduler(
            optimizer,
            {"epochs": 5, "lr": 0.1, "warmup_steps": 0},
            num_train_batches=10,
        )

        assert scheduler is None
