import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metadata_utils import (
    MISSING,
    UNK,
    MetadataVocab,
    build_metadata_tensors,
    load_item_metadata,
)


def test_metadata_vocab_categorical(tmp_path):
    """Test categorical field encoding with PAD/MISSING/UNK."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "item_idx,language,difficulty,theme,software,job,type,duration\n"
        "1,fr,beginner,design;office,excel,data analyst,video,30\n"
        "2,en,advanced,design,powerpoint,,article,\n"
    )
    df = load_item_metadata(str(csv_path), n_items=2)
    vocab = MetadataVocab.build(df)

    # language: fr → some index > 2, en → some index > 2
    lang_fr = vocab.encode_categorical("language", "fr")
    lang_en = vocab.encode_categorical("language", "en")
    assert lang_fr > 2  # not PAD/MISSING/UNK
    assert lang_en > 2
    assert lang_fr != lang_en

    # Missing value → MISSING token (1)
    assert vocab.encode_categorical("language", None) == 1

    # Unknown value → UNK token (2)
    assert vocab.encode_categorical("language", "zh") == 2


def test_metadata_vocab_multilabel(tmp_path):
    """Test multi-label field encoding."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "item_idx,language,difficulty,theme,software,job,type,duration\n"
        "1,fr,beginner,design;office,excel,data analyst,video,30\n"
        "2,en,advanced,design,powerpoint,,article,\n"
    )
    df = load_item_metadata(str(csv_path), n_items=2)
    vocab = MetadataVocab.build(df)

    theme_1 = vocab.encode_multilabel("theme", "design;office")
    assert len(theme_1) == 2
    assert all(idx > 2 for idx in theme_1)

    # Empty → [MISSING]
    theme_empty = vocab.encode_multilabel("theme", None)
    assert theme_empty == [1]


def test_build_metadata_tensors(tmp_path):
    """Test tensor building for all items."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "item_idx,language,difficulty,theme,software,job,type,duration\n"
        "1,fr,beginner,design;office,excel,data analyst,video,30\n"
        "2,en,advanced,design,powerpoint,,article,\n"
    )
    df = load_item_metadata(str(csv_path), n_items=2)
    vocab = MetadataVocab.build(df)
    tensors = build_metadata_tensors(vocab, df, n_items=2)

    assert "language" in tensors
    assert "difficulty" in tensors
    assert tensors["language"].shape == (3,)  # n_items + 1 (padding idx 0)
    assert tensors["language"][0] == 0  # padding
    assert tensors["language"][1] > 2   # item 1


# ---- train-item subset vocab ----

def _make_meta_df() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame({
        "item_idx": [1, 2, 3, 4],
        "language": ["EN", "FR", "DE", "JP"],
        "difficulty": ["easy", "hard", "easy", "hard"],
        "theme": ["t1", "t2", "t3;t1", "t4"],
        "software": ["s1", "s2", "s3", "s4"],
        "job": ["j1", "j2", "j3", "j4"],
        "type": ["x", "y", "x", "z"],
        "duration": [60.0, 30.0, 90.0, 45.0],
    })


def test_vocab_built_on_train_items_excludes_test_categories():
    """Categories that only appear in test items should NOT be in vocab."""
    df = _make_meta_df()
    train_items = {1, 2}  # items 3, 4 are test-only
    vocab = MetadataVocab.build(df, train_item_idx=train_items)
    assert "JP" not in vocab.categorical["language"]
    assert "t4" not in vocab.multilabel["theme"]
    # Categories that exist in train should be present
    assert "EN" in vocab.categorical["language"]
    assert "t1" in vocab.multilabel["theme"]


def test_vocab_test_items_map_to_unk():
    """Items not in train should map to UNK through encode_categorical/encode_multilabel."""
    df = _make_meta_df()
    train_items = {1, 2}
    vocab = MetadataVocab.build(df, train_item_idx=train_items)
    # Item 3 has language "DE" and theme "t3;t1" — "DE" not in train vocab → UNK
    assert vocab.encode_categorical("language", "DE") == UNK
    # "t3" not in train vocab → UNK; "t1" in train vocab → real id
    encoded = vocab.encode_multilabel("theme", "t3;t1")
    assert UNK in encoded
    assert any(idx > 2 for idx in encoded)


def test_vocab_without_filter_keeps_all_categories():
    """Default behavior (no filter) should still include all categories."""
    df = _make_meta_df()
    vocab = MetadataVocab.build(df)
    assert "JP" in vocab.categorical["language"]
    assert "t4" in vocab.multilabel["theme"]


# ---- encode_duration: missing mask ----

def test_encode_duration_missing_mask_for_none():
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    val, missing = vocab.encode_duration(None)
    assert val == 0.0
    assert missing is True


def test_encode_duration_missing_mask_for_nan():
    import math
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    val, missing = vocab.encode_duration(float("nan"))
    assert val == 0.0
    assert missing is True


def test_encode_duration_missing_mask_for_empty_string():
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    val, missing = vocab.encode_duration("")
    assert val == 0.0
    assert missing is True


def test_encode_duration_zero_is_not_missing():
    """A real duration of 0 is valid; only None/NaN/empty is missing."""
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    val, missing = vocab.encode_duration(0.0)
    assert missing is False
    assert val == 0.0  # log1p(0) = 0, normalized by 0-mean 1-std → 0


def test_encode_duration_valid_value_returns_normalized():
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    val, missing = vocab.encode_duration(60.0)
    assert missing is False
    assert val != 0.0  # log1p(60) ≈ 4.11, not zero


def test_encode_duration_rejects_negative():
    vocab = MetadataVocab(
        categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        vocab.encode_duration(-5.0)


# ---- build_metadata_tensors: duration_missing ----

def test_build_metadata_tensors_emits_duration_missing():
    """build_metadata_tensors should emit both duration and duration_missing."""
    import pandas as pd
    df = pd.DataFrame({
        "item_idx": [1, 2, 3],
        "language": ["EN", "FR", "DE"],
        "difficulty": ["easy", "hard", "easy"],
        "theme": ["t1", "t2", "t3"],
        "software": ["s1", "s2", "s3"],
        "job": ["j1", "j2", "j3"],
        "type": ["x", "y", "x"],
        "duration": [60.0, None, 30.0],
    })
    df = df.sort_values("item_idx").reset_index(drop=True)
    vocab = MetadataVocab.build(df)
    tensors = build_metadata_tensors(vocab, df, n_items=3)

    assert "duration" in tensors
    assert "duration_missing" in tensors
    assert tensors["duration"].dtype == torch.float32
    assert tensors["duration_missing"].dtype == torch.bool
    # Padding row at index 0: missing=1
    assert tensors["duration_missing"][0].item() is True
    # Item 1: 60.0 → not missing
    assert tensors["duration_missing"][1].item() is False
    # Item 2: None → missing
    assert tensors["duration_missing"][2].item() is True
    # Item 3: 30.0 → not missing
    assert tensors["duration_missing"][3].item() is False


# ---- MetadataVocab.save with train_item_sha256 ----

def test_metadata_vocab_save_includes_train_item_sha256(tmp_path):
    import hashlib
    vocab = MetadataVocab(
        categorical={"language": {"EN": 3}}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    out = tmp_path / "vocab.json"
    train_items = {1, 2, 3}
    expected_sha = hashlib.sha256(str(sorted(train_items)).encode()).hexdigest()
    vocab.save(out, train_item_sha256=expected_sha)
    data = json.loads(out.read_text())
    assert data["train_item_sha256"] == expected_sha


def test_metadata_vocab_save_omits_sha_when_none(tmp_path):
    vocab = MetadataVocab(
        categorical={"language": {"EN": 3}}, multilabel={}, duration_mean=0.0, duration_std=1.0
    )
    out = tmp_path / "vocab.json"
    vocab.save(out)
    data = json.loads(out.read_text())
    assert "train_item_sha256" not in data
