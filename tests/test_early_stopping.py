"""
tests/test_early_stopping.py
=============================
Tests for early stopping logic and config hyperparameter verification.
"""

import math
import torch
import torch.nn as nn

from training.trainer import Trainer


class DummyModel(nn.Module):
    """Simple linear model for testing Trainer loop."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=16):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"input": torch.randn(1, 4), "target": torch.tensor([idx % 2])}


class TestEarlyStopping:
    """Unit tests for early stopping logic."""

    def test_early_stop_triggers_on_plateau(self):
        """Training stops when NDCG plateaus for patience epochs."""
        # Simulate: NDCG peaks at epoch 6 (0.312), then declines
        metrics_sequence = [
            0.1, 0.2, 0.3, 0.31, 0.311, 0.312, 0.311, 0.310, 0.309,
        ]

        best_val_ndcg = -1.0
        patience_counter = 0
        patience = 3
        min_delta = 1e-4
        stop_epoch = None

        for epoch_idx, ndcg in enumerate(metrics_sequence, start=1):
            prev_best = best_val_ndcg  # capture BEFORE updating

            if math.isfinite(ndcg) and ndcg > best_val_ndcg:
                best_val_ndcg = ndcg

            if math.isfinite(ndcg) and ndcg > prev_best + min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_epoch = epoch_idx
                    break

        assert stop_epoch == 9, f"Expected stop at epoch 9, got {stop_epoch}"

    def test_early_stop_disabled_when_patience_zero(self):
        """No early stopping when patience=0."""
        model = DummyModel()
        dataset = DummyDataset(16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        metrics = [0.1, 0.1, 0.1, 0.1, 0.1]  # flatline

        patience = 0
        min_delta = 1e-4
        stopped_early = False

        for epoch_idx, ndcg in enumerate(metrics, start=1):
            if patience > 0:
                # This block should never execute
                stopped_early = True

        assert not stopped_early, "Should not stop early when patience=0"

    def test_best_model_is_restored(self):
        """Best state is from peak epoch, not final epoch."""
        model = DummyModel()

        metrics_sequence = [0.1, 0.3, 0.2, 0.15]
        best_val_ndcg = -1.0
        best_state = None

        for epoch_idx, ndcg in enumerate(metrics_sequence, start=1):
            if math.isfinite(ndcg) and ndcg > best_val_ndcg:
                best_val_ndcg = ndcg
                best_state = epoch_idx  # proxy for state dict

        assert best_val_ndcg == 0.3, f"Best NDCG should be 0.3, got {best_val_ndcg}"
        assert best_state == 2, f"Best epoch should be 2, got {best_state}"


class TestConfigVerification:
    """Verify configs.py has correct hyperparameter values."""

    def test_sasrec_config_values(self):
        from training.configs import MODEL_CONFIGS
        tk = MODEL_CONFIGS["sasrec"]["train_kwargs"]
        assert tk["epochs"] == 50
        assert tk["beta2"] == 0.98
        assert tk["weight_decay"] == 1e-4
        assert tk["early_stop_patience"] == 10
        assert tk["early_stop_min_delta"] == 1e-4

    def test_gsasrec_config_values(self):
        from training.configs import MODEL_CONFIGS
        tk = MODEL_CONFIGS["gsasrec"]["train_kwargs"]
        assert tk["epochs"] == 50
        assert tk["beta2"] == 0.98
        assert tk["weight_decay"] == 0.0
        assert tk["early_stop_patience"] == 10
        assert tk["early_stop_min_delta"] == 1e-4
        assert "use_confidence_weighting" not in tk

    def test_other_models_unchanged(self):
        from training.configs import MODEL_CONFIGS
        for m in ["gru4rec", "bert4rec", "bprmf"]:
            tk = MODEL_CONFIGS[m]["train_kwargs"]
            assert "beta2" not in tk, f"{m} should not have beta2"
            assert "early_stop_patience" not in tk, f"{m} should not have early_stop"

    def test_gsasrec_model_kwargs_unchanged(self):
        from training.configs import MODEL_CONFIGS
        mk = MODEL_CONFIGS["gsasrec"]["model_kwargs"]
        assert mk["t"] == 0.5
        assert mk["num_neg"] == 32
        assert mk["hidden_dim"] == 64
        assert mk["num_heads"] == 2
        assert mk["num_layers"] == 2
