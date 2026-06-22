import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metrics import _ranks_from_logits


def test_ranks_from_logits_raises_on_nan():
    """If logits have NaN, _ranks_from_logits must raise — not silently produce garbage ranks."""
    logits = torch.zeros(2, 10)
    logits[0, 5] = float("nan")
    history_mask = torch.zeros(2, 10, dtype=torch.bool)
    target = torch.tensor([1, 1])
    with pytest.raises(FloatingPointError, match="Non-finite"):
        _ranks_from_logits(logits, history_mask, target)


def test_ranks_from_logits_raises_on_inf():
    logits = torch.zeros(2, 10)
    logits[1, 3] = float("inf")
    history_mask = torch.zeros(2, 10, dtype=torch.bool)
    target = torch.tensor([1, 1])
    with pytest.raises(FloatingPointError, match="Non-finite"):
        _ranks_from_logits(logits, history_mask, target)


def test_ranks_from_logits_passes_on_finite():
    """Smoke test: finite logits work normally."""
    logits = torch.randn(2, 10)
    history_mask = torch.zeros(2, 10, dtype=torch.bool)
    target = torch.tensor([1, 1])
    ranks = _ranks_from_logits(logits, history_mask, target)
    assert len(ranks) == 2
    assert all(1 <= r <= 10 for r in ranks)
