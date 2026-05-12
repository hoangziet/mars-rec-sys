"""
tests/test_dataloader.py
========================
Unit tests for pipeline/loaders.py.
"""

import os
import tempfile

import pandas as pd
import pytest
import torch

from pipeline.loaders import (
    FullSortEvalDataset,
    MaskedSequenceDataset,
    TrainSequenceDataset,
    pad_sequence,
    parse_seq,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_train_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df  = pd.DataFrame(rows)
    df.to_csv(tmp.name, index=False)
    return tmp.name


def make_eval_csv(rows: list[dict]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df  = pd.DataFrame(rows)
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ---------------------------------------------------------------------------
# pad_sequence
# ---------------------------------------------------------------------------


class TestPadSequence:
    def test_short_sequence_left_padded(self):
        assert pad_sequence([1, 2, 3], 5) == [0, 0, 1, 2, 3]

    def test_exact_length_unchanged(self):
        assert pad_sequence([1, 2, 3], 3) == [1, 2, 3]

    def test_long_sequence_truncated_from_left(self):
        assert pad_sequence([1, 2, 3, 4, 5, 6], 4) == [3, 4, 5, 6]

    def test_custom_pad_token(self):
        assert pad_sequence([1, 2], 4, pad_token=99) == [99, 99, 1, 2]


# ---------------------------------------------------------------------------
# parse_seq
# ---------------------------------------------------------------------------


class TestParseSeq:
    def test_string_list(self):
        assert parse_seq("[1, 2, 3]") == [1, 2, 3]

    def test_already_list(self):
        assert parse_seq([1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# TrainSequenceDataset
# ---------------------------------------------------------------------------


class TestTrainSequenceDataset:
    def _make_dataset(self, seqs, n_items=20, max_len=5, num_neg=1):
        rows = [{"item_sequence": str(seq)} for seq in seqs]
        path = make_train_csv(rows)
        ds   = TrainSequenceDataset(path, max_len=max_len, n_items=n_items, num_neg=num_neg)
        os.unlink(path)
        return ds

    def test_length_equals_sum_of_seq_positions(self):
        ds = self._make_dataset([[1, 2, 3, 4], [5, 6, 7]])
        # seq [1,2,3,4] → 3 positions; seq [5,6,7] → 2 positions
        assert len(ds) == 5

    def test_seq_shorter_than_2_skipped(self):
        ds = self._make_dataset([[1], [2, 3, 4]])
        # [1] skipped; [2,3,4] → 2 positions
        assert len(ds) == 2

    def test_output_keys(self):
        ds    = self._make_dataset([[1, 2, 3]])
        batch = ds[0]
        assert "input_seq" in batch
        assert "pos_items" in batch
        assert "neg_items" in batch
        assert "confidence" not in batch

    def test_neg_tensor_scalar_when_num_neg_1(self):
        ds    = self._make_dataset([[1, 2, 3]], num_neg=1)
        batch = ds[0]
        assert batch["neg_items"].dim() == 0  # scalar

    def test_neg_tensor_vector_when_num_neg_k(self):
        ds    = self._make_dataset([[1, 2, 3, 4, 5]], num_neg=4)
        batch = ds[0]
        assert batch["neg_items"].shape == (4,)

    def test_input_seq_has_correct_length(self):
        ds    = self._make_dataset([[1, 2, 3]], max_len=5)
        batch = ds[0]
        assert batch["input_seq"].shape == (5,)


# ---------------------------------------------------------------------------
# FullSortEvalDataset
# ---------------------------------------------------------------------------


class TestFullSortEvalDataset:
    N_ITEMS = 10

    def _make_dataset(self, rows):
        path = make_eval_csv(rows)
        ds   = FullSortEvalDataset(path, n_items=self.N_ITEMS, max_len=5)
        os.unlink(path)
        return ds

    def _row(self, uid=1, train_seq="[1, 2, 3]", target=4):
        return {"user_idx": uid, "train_seq": train_seq, "target": target}

    def test_output_keys(self):
        ds    = self._make_dataset([self._row()])
        batch = ds[0]
        for key in ("user", "input_seq", "mask", "target", "history_mask"):
            assert key in batch

    def test_history_mask_shape(self):
        ds    = self._make_dataset([self._row()])
        batch = ds[0]
        assert batch["history_mask"].shape == (self.N_ITEMS + 1,)

    def test_padding_token_is_masked(self):
        ds    = self._make_dataset([self._row()])
        batch = ds[0]
        assert batch["history_mask"][0].item() is True

    def test_seen_items_are_masked(self):
        ds    = self._make_dataset([self._row(train_seq="[1, 2, 3]", target=4)])
        batch = ds[0]
        for item in [1, 2, 3]:
            assert batch["history_mask"][item].item() is True

    def test_target_is_NOT_masked(self):
        ds    = self._make_dataset([self._row(train_seq="[1, 2, 3]", target=4)])
        batch = ds[0]
        # Target must remain unmasked so model can rank it.
        assert batch["history_mask"][4].item() is False

    def test_unseen_items_are_not_masked(self):
        ds    = self._make_dataset([self._row(train_seq="[1, 2]", target=3)])
        batch = ds[0]
        # Item 5 is neither in history nor target → should be rankable
        assert batch["history_mask"][5].item() is False

    def test_target_value_correct(self):
        ds    = self._make_dataset([self._row(target=7)])
        batch = ds[0]
        assert batch["target"].item() == 7

    def test_multiple_users(self):
        rows = [self._row(uid=1, target=4), self._row(uid=2, target=5)]
        ds   = self._make_dataset(rows)
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# MaskedSequenceDataset
# ---------------------------------------------------------------------------


class TestMaskedSequenceDataset:
    def _make_dataset(self, seqs, n_items=20, max_len=5, **kwargs):
        rows = [{"item_sequence": str(seq)} for seq in seqs]
        path = make_train_csv(rows)
        ds = MaskedSequenceDataset(
            path,
            n_items=n_items,
            max_len=max_len,
            is_train=True,
            **kwargs,
        )
        os.unlink(path)
        return ds

    def test_sliding_window_expands_long_sequences(self):
        ds = self._make_dataset(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            max_len=4,
            dupe_factor=1,
            prop_sliding_window=0.5,
            force_last_item_mask=False,
        )
        assert len(ds) == 3

    def test_force_last_item_mask_adds_extra_training_instance(self):
        ds = self._make_dataset(
            [[1, 2, 3, 4]],
            max_len=4,
            dupe_factor=1,
            force_last_item_mask=True,
        )
        assert len(ds) == 2

        last_item_batch = ds[1]
        assert last_item_batch["input_seq"][-1].item() == ds.mask_token
        assert last_item_batch["labels"][-1].item() == 4
