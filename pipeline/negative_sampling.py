from __future__ import annotations

import numpy as np


def sample_uniform_excluding_targets(*, rng: np.random.Generator, targets: np.ndarray, n_items: int, num_negatives: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    if n_items < 2:
        raise ValueError("n_items must be >= 2")
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if np.any((targets < 1) | (targets > n_items)):
        raise ValueError(f"targets must be in [1, {n_items}]")

    draws = rng.integers(low=1, high=n_items, size=(targets.size, num_negatives), endpoint=False, dtype=np.int64)
    return draws + (draws >= targets[:, None])


def sample_unseen_user(*, rng: np.random.Generator, seen_items: set[int], n_items: int, valid_positions: int, num_negatives: int) -> np.ndarray:
    if valid_positions < 0:
        raise ValueError("valid_positions must be >= 0")
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    all_items = np.arange(1, n_items + 1, dtype=np.int64)
    seen_array = np.fromiter(sorted(seen_items), dtype=np.int64)
    pool = np.setdiff1d(all_items, seen_array, assume_unique=False)
    if pool.size == 0:
        raise RuntimeError("Cannot sample SASRec negative: no unseen item remains for this user.")

    return rng.choice(pool, size=(valid_positions, num_negatives), replace=True).astype(np.int64)


class PopularityNegativeSampler:
    def __init__(self, *, item_counts: np.ndarray, sample_alpha: float) -> None:
        counts = np.asarray(item_counts, dtype=np.float64)
        if counts.ndim != 1:
            raise ValueError("item_counts must be one-dimensional")
        if counts.size < 3:
            raise ValueError("item_counts must contain padding plus at least two items")
        if sample_alpha < 0:
            raise ValueError("sample_alpha must be >= 0")

        counts = counts.copy()
        counts[0] = 0.0
        weights = counts ** sample_alpha
        weights[0] = 0.0
        if weights.sum() <= 0:
            raise ValueError("Popularity sampler has zero total weight")

        self.n_items = counts.size - 1
        self.probabilities = weights / weights.sum()
        self.item_ids = np.arange(counts.size, dtype=np.int64)

    def sample(self, *, rng: np.random.Generator, targets: np.ndarray, num_negatives: int) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.int64)
        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        if np.any((targets < 1) | (targets > self.n_items)):
            raise ValueError(f"targets must be in [1, {self.n_items}]")
        if num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")

        result = rng.choice(self.item_ids, size=(targets.size, num_negatives), replace=True, p=self.probabilities).astype(np.int64)
        collisions = result == targets[:, None]
        while collisions.any():
            result[collisions] = rng.choice(self.item_ids, size=int(collisions.sum()), replace=True, p=self.probabilities)
            collisions = result == targets[:, None]
        return result
