"""
pipeline/confidence.py
======================
Confidence weighting for sequential recommendation loss.

Classes:
    WeightedCriterionFn — wraps a per-sample loss with engagement-based confidence weighting
"""

import torch


class WeightedCriterionFn:
    """Wraps a base criterion_fn with confidence weighting.

    The base fn must return per-sample loss (B,) when reduction='none'.
    Computes: weighted_loss = sum(confidence * loss) / sum(confidence)

    Confidence formula:
        watch_ratio = clip(watch_percentage, 0, 100) / 100
        confidence  = 1 + alpha * watch_ratio

    If engagement is missing from batch, confidence = 1.0 for all samples.
    """

    def __init__(self, base_criterion_fn, alpha: float):
        self.base_fn = base_criterion_fn
        self.alpha = alpha

    def __call__(self, model, batch, device):
        per_sample_loss = self.base_fn(model, batch, device)  # (B,)

        if "engagement" in batch:
            engagement = batch["engagement"].to(device)  # (B,)
            engagement = engagement.clamp(0.0, 1.0)      # clip to [0, 1]
        else:
            engagement = torch.zeros(per_sample_loss.size(0), device=device)

        confidence = 1.0 + self.alpha * engagement  # (B,)
        return (confidence * per_sample_loss).sum() / confidence.sum()
