import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metadata_utils import MetadataVocab, build_metadata_tensors, load_item_metadata


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
