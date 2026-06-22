import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.trainer import NoValidCheckpointError, Trainer


class _NaNModel(nn.Module):
    """Model whose loss is always NaN — to trigger the FAILED branch."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)


class _FakeLoader:
    """Minimal loader stub: iterable of dict batches + a .dataset with __len__."""
    def __init__(self, n=2, input_dim=4):
        self.dataset = list(range(n))  # only used for len()
        self._batches = [
            {"input_seq": torch.zeros(1, input_dim),
             "pos_items": torch.tensor([1]),
             "neg_items": torch.tensor([2])}
            for _ in range(n)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _make_loader(n=2, input_dim=4):
    return _FakeLoader(n=n, input_dim=input_dim)


def test_trainer_raises_when_no_valid_checkpoint(tmp_path):
    """When all training is non-finite, trainer must raise NoValidCheckpointError.

    The caller (rq4_ablation etc.) must catch this specifically and skip
    per-user export / downstream reporting for that seed.
    """
    model = _NaNModel()
    trainer = Trainer("nan_test", "cpu", run_dir=tmp_path, use_mlflow=False)

    def nan_criterion(model, batch, device):
        return torch.tensor(float("nan"))

    def nan_eval(model, loader, device):
        return {"NDCG@10": float("nan"), "Recall@10": float("nan")}

    loader = _make_loader()
    with pytest.raises(NoValidCheckpointError, match="No valid checkpoint"):
        trainer.train(
            model=model, train_loader=loader, val_loader=loader,
            test_loader=loader,
            optimizer=torch.optim.SGD(model.parameters(), 0.01),
            epochs=2, criterion_fn=nan_criterion, eval_fn=nan_eval,
        )

    # No checkpoint saved
    assert not (tmp_path / "best_model.pt").exists()
    # Metrics still saved (for traceability)
    assert (tmp_path / "metrics.json").exists()


def test_trainer_saves_checkpoint_when_loss_finite(tmp_path):
    """Sanity check: with finite loss, checkpoint IS saved."""
    torch.manual_seed(0)
    model = _NaNModel()
    trainer = Trainer("ok_test", "cpu", run_dir=tmp_path, use_mlflow=False)

    def finite_criterion(model, batch, device):
        out = model(batch["input_seq"])
        return out.sum()

    def finite_eval(model, loader, device):
        return {"NDCG@10": 0.1, "Recall@10": 0.1}

    loader = _make_loader()
    tracker = trainer.train(
        model=model, train_loader=loader, val_loader=loader,
        test_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), 0.01),
        epochs=2, criterion_fn=finite_criterion, eval_fn=finite_eval,
    )

    assert (tmp_path / "best_model.pt").exists()
    assert tracker.test_results  # not empty


def test_train_one_epoch_allows_less_than_half_nan_batches(tmp_path):
    """1/3 non-finite batches should not trip the divergence guard."""
    torch.manual_seed(0)
    model = _NaNModel()
    trainer = Trainer("threshold_test", "cpu", run_dir=tmp_path, use_mlflow=False)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loader = _make_loader(n=3)

    calls = {"count": 0}

    def criterion(model, batch, device):
        calls["count"] += 1
        if calls["count"] == 1:
            return torch.tensor(float("nan"))
        return model(batch["input_seq"]).sum()

    train_loss = trainer._train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion_fn=criterion,
        gradient_clip=5.0,
        scheduler=None,
    )

    assert torch.isfinite(torch.tensor(train_loss)).item()
