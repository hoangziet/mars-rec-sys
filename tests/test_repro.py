import sys
from pathlib import Path

import random

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_seed_everything_enables_deterministic_cudnn():
    from training.repro import seed_everything

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    seed_everything(42)

    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    a1 = random.random()
    b1 = np.random.rand()
    c1 = torch.rand(1).item()

    seed_everything(42)

    assert random.random() == a1
    assert np.random.rand() == b1
    assert torch.rand(1).item() == c1
