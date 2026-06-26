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


# ---------------------------------------------------------------------------
# ShiftedSequenceDataset + RQ1 loader contract tests
# ---------------------------------------------------------------------------

import pandas as pd
import torch

from pipeline.loaders import ShiftedSequenceDataset, get_rq1_train_loader, get_val_loss_loader


def _write_train_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["1 2 3 4 5"],
            "engagement_sequence": ["0.1 0.2 0.3 0.4 0.5"],
        }
    ).to_csv(path, index=False)


def test_shifted_dataset_builds_one_sample_per_user(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_train_csv(csv_path)

    dataset = ShiftedSequenceDataset(
        str(csv_path),
        n_items=10,
        max_len=4,
        num_negatives=1,
        negative_sampling="unseen_user",
        seed=42,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["input_seq"].tolist() == [1, 2, 3, 4]
    assert sample["pos_items"].tolist() == [2, 3, 4, 5]
    assert sample["loss_mask"].tolist() == [True, True, True, True]
    assert sample["engagement"].tolist() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert sample["neg_items"].shape == (4, 1)


def test_shifted_dataset_left_pads_short_sequence(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["3 4 5"],
            "engagement_sequence": ["0.3 0.4 0.5"],
        }
    ).to_csv(csv_path, index=False)

    dataset = ShiftedSequenceDataset(
        str(csv_path),
        n_items=10,
        max_len=5,
        num_negatives=2,
        negative_sampling="catalog_except_positive",
        seed=42,
    )

    sample = dataset[0]
    assert sample["input_seq"].tolist() == [0, 0, 0, 3, 4]
    assert sample["pos_items"].tolist() == [0, 0, 0, 4, 5]
    assert sample["loss_mask"].tolist() == [False, False, False, True, True]
    assert torch.all(sample["neg_items"][:3] == 0)


def test_dataset_epoch_changes_negatives_reproducibly(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_train_csv(csv_path)

    first = ShiftedSequenceDataset(
        str(csv_path),
        n_items=30,
        max_len=4,
        num_negatives=16,
        negative_sampling="catalog_except_positive",
        seed=42,
    )
    second = ShiftedSequenceDataset(
        str(csv_path),
        n_items=30,
        max_len=4,
        num_negatives=16,
        negative_sampling="catalog_except_positive",
        seed=42,
    )

    first.set_epoch(0)
    second.set_epoch(0)
    epoch_zero_a = first[0]["neg_items"]
    epoch_zero_b = second[0]["neg_items"]

    first.set_epoch(1)
    second.set_epoch(1)
    epoch_one_a = first[0]["neg_items"]
    epoch_one_b = second[0]["neg_items"]

    assert torch.equal(epoch_zero_a, epoch_zero_b)
    assert torch.equal(epoch_one_a, epoch_one_b)
    assert not torch.equal(epoch_zero_a, epoch_one_a)


def test_rq1_train_loader_emits_shifted_batch(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_train_csv(csv_path)
    stats = {"n_items": 10}

    loader = get_rq1_train_loader(
        model_type="sasrec",
        train_csv=str(csv_path),
        stats=stats,
        batch_size=1,
        max_len=4,
        num_neg=1,
        seed=42,
    )
    batch = next(iter(loader))

    assert batch["input_seq"].shape == (1, 4)
    assert batch["pos_items"].shape == (1, 4)
    assert batch["neg_items"].shape == (1, 4, 1)
    assert batch["loss_mask"].shape == (1, 4)


def test_val_loss_loader_keeps_scalar_batch(tmp_path: Path):
    csv_path = tmp_path / "val.csv"
    pd.DataFrame(
        {
            "user_idx": [1],
            "item_sequence": ["1 2 3 4"],
            "target_item": [5],
        }
    ).to_csv(csv_path, index=False)
    stats = {"n_items": 10}

    loader = get_val_loss_loader(
        model_type="sasrec",
        val_csv=str(csv_path),
        stats=stats,
        batch_size=1,
        max_len=4,
        num_neg=1,
        seed=42,
    )
    batch = next(iter(loader))

    assert batch["input_seq"].shape == (1, 4)
    assert batch["pos_items"].shape == (1,)
    assert batch["neg_items"].shape == (1,)
