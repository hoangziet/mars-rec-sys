import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.confidence import WeightedCriterionFn


def _fake_criterion(model, batch, device):
    """Returns per-sample loss of shape (B,)."""
    return batch["per_sample_loss"].to(device)


def test_weighted_criterion_applies_confidence():
    fn = WeightedCriterionFn(_fake_criterion, alpha=1.0)
    batch = {
        "per_sample_loss": torch.tensor([0.5, 0.3, 0.8]),
        "engagement": torch.tensor([1.0, 0.0, 0.5]),
    }
    # confidence = [1 + 1.0*1.0, 1 + 1.0*0.0, 1 + 1.0*0.5] = [2.0, 1.0, 1.5]
    # weighted_loss = (2.0*0.5 + 1.0*0.3 + 1.5*0.8) / (2.0 + 1.0 + 1.5)
    # = (1.0 + 0.3 + 1.2) / 4.5 = 2.5 / 4.5 ≈ 0.5556
    result = fn(None, batch, torch.device("cpu"))
    assert abs(result.item() - 2.5 / 4.5) < 1e-4


def test_weighted_criterion_alpha_zero_is_weighted_mean():
    fn = WeightedCriterionFn(_fake_criterion, alpha=0.0)
    batch = {
        "per_sample_loss": torch.tensor([0.5, 0.3, 0.8]),
        "engagement": torch.tensor([1.0, 0.0, 0.5]),
    }
    # confidence = [1.0, 1.0, 1.0] → weighted = regular mean
    result = fn(None, batch, torch.device("cpu"))
    assert abs(result.item() - 0.5333) < 1e-3


def test_weighted_criterion_missing_engagement_defaults_to_one():
    fn = WeightedCriterionFn(_fake_criterion, alpha=2.0)
    batch = {"per_sample_loss": torch.tensor([0.4, 0.6])}
    # No engagement key → confidence = [1.0, 1.0]
    result = fn(None, batch, torch.device("cpu"))
    assert abs(result.item() - 0.5) < 1e-4


def test_weighted_criterion_clips_engagement():
    fn = WeightedCriterionFn(_fake_criterion, alpha=1.0)
    batch = {
        "per_sample_loss": torch.tensor([0.5]),
        "engagement": torch.tensor([150.0]),  # > 100, should be clipped to 1.0
    }
    result = fn(None, batch, torch.device("cpu"))
    expected_confidence = 1.0 + 1.0 * 1.0  # engagement clipped to 1.0
    assert abs(result.item() - 0.5) < 1e-4  # (2.0*0.5)/2.0 = 0.5
