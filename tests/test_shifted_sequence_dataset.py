import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from pipeline.negative_sampling import (
    PopularityNegativeSampler,
    sample_uniform_excluding_targets,
    sample_unseen_user,
)


def test_uniform_sampler_excludes_each_position_target():
    rng = np.random.default_rng(42)
    targets = np.array([2, 5, 9], dtype=np.int64)

    negatives = sample_uniform_excluding_targets(
        rng=rng,
        targets=targets,
        n_items=10,
        num_negatives=100,
    )

    assert negatives.shape == (3, 100)
    assert np.all((negatives >= 1) & (negatives <= 10))
    for row, target in zip(negatives, targets, strict=True):
        assert target not in row


def test_unseen_user_sampler_excludes_full_user_history():
    rng = np.random.default_rng(42)

    negatives = sample_unseen_user(
        rng=rng,
        seen_items={1, 2, 3, 4},
        n_items=10,
        valid_positions=3,
        num_negatives=50,
    )

    assert negatives.shape == (3, 50)
    assert {1, 2, 3, 4}.isdisjoint(set(negatives.flatten().tolist()))


def test_unseen_user_sampler_fails_when_pool_is_empty():
    rng = np.random.default_rng(42)

    with pytest.raises(RuntimeError, match="no unseen item remains"):
        sample_unseen_user(
            rng=rng,
            seen_items={1, 2, 3},
            n_items=3,
            valid_positions=1,
            num_negatives=1,
        )


def test_popularity_sampler_excludes_target():
    sampler = PopularityNegativeSampler(
        item_counts=np.array([0.0, 100.0, 10.0, 1.0, 1.0], dtype=np.float64),
        sample_alpha=0.5,
    )
    rng = np.random.default_rng(42)
    targets = np.array([1, 2, 3], dtype=np.int64)

    negatives = sampler.sample(rng=rng, targets=targets, num_negatives=100)

    assert negatives.shape == (3, 100)
    for row, target in zip(negatives, targets, strict=True):
        assert target not in row
