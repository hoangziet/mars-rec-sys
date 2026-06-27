import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gsasrec import GSASRec
from pipeline.item_encoder import ItemEncoder


def test_no_legacy_csv_outputs(tmp_path):
    """After preprocess.py runs, no legacy CSVs at output_dir/ root."""
    import pandas as pd

    from data.preprocess import build_dataset_stats, save_processed_outputs

    interactions = pd.DataFrame({
        "user_idx": [1, 1, 2],
        "item_idx": [10, 20, 30],
        "user_id": ["u1", "u1", "u2"],
        "item_id": ["a", "b", "c"],
        "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
        "engagement_score": [0.0, 0.0, 0.0],
        "has_watch_signal": [False, False, False],
        "sequence_order": [0, 1, 0],
    })
    train_df = pd.DataFrame({
        "user_idx": [1, 2],
        "item_sequence": [[10], [30]],
        "engagement_sequence": [[0.0], [0.0]],
        "watch_signal_sequence": [[False], [False]],
        "sequence_length": [1, 1],
    })
    val_df = train_df.copy()
    val_df["target_item"] = [20, 30]
    val_df["target_engagement"] = [0.0, 0.0]
    val_df["target_has_watch_signal"] = [False, False]
    test_df = val_df.copy()
    item_metadata = pd.DataFrame({
        "item_idx": [10, 20, 30],
        "item_id": ["a", "b", "c"],
        "title": ["t1", "t2", "t3"],
        "description": ["d1", "d2", "d3"],
        "text": ["t1 [SEP] d1", "t2 [SEP] d2", "t3 [SEP] d3"],
        "language": ["EN", "EN", "EN"],
        "difficulty": ["easy", "easy", "easy"],
        "theme": ["", "", ""],
        "software": ["", "", ""],
        "job": ["", "", ""],
        "type": ["", "", ""],
        "duration": [60.0, 30.0, 45.0],
    })
    user_id_map = pd.DataFrame({"user_id": ["u1", "u2"], "user_idx": [1, 2]})
    item_id_map = pd.DataFrame({"item_id": ["a", "b", "c"], "item_idx": [10, 20, 30]})

    save_processed_outputs(
        output_dir=tmp_path,
        interactions=interactions,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        item_metadata=item_metadata,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        dataset_stats=build_dataset_stats(2, 3, 3, 4, 3),
        preprocessing_report={},
    )

    assert not (tmp_path / "interactions.csv").exists()
    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "val.csv").exists()
    assert not (tmp_path / "test.csv").exists()
    assert not (tmp_path / "item_meta.csv").exists()
    assert not (tmp_path / "dataset_stats.json").exists()

    assert (tmp_path / "interactions" / "interactions.csv").exists()
    assert (tmp_path / "splits" / "train_sequences.csv").exists()
    assert (tmp_path / "splits" / "val_sequences.csv").exists()
    assert (tmp_path / "splits" / "test_sequences.csv").exists()
    assert (tmp_path / "item_features" / "item_metadata.csv").exists()
    assert (tmp_path / "reports" / "dataset_stats.json").exists()


def test_gsasrec_docstring_has_paper_default_caveat():
    docstring = GSASRec.__doc__
    assert docstring is not None
    assert "K=256" in docstring
    assert "memory budget" in docstring or "project default" in docstring.lower()


def test_num_neg_single_source():
    """gsasrec.yaml has num_neg in only ONE of model_kwargs / train_kwargs."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "model" / "gsasrec.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_neg = "num_neg" in cfg.get("model_kwargs", {})
    train_neg = "num_neg" in cfg.get("train_kwargs", {})
    assert model_neg != train_neg, (
        f"num_neg must be in exactly one of model_kwargs / train_kwargs. "
        f"Found in model_kwargs={model_neg}, train_kwargs={train_neg}"
    )


def test_item_encoder_buffers_not_persistent():
    """Metadata/text tensors must not be in state_dict (they're loaded fresh)."""
    from pipeline.metadata_utils import MetadataVocab

    vocab = MetadataVocab(categorical={}, multilabel={}, duration_mean=0.0, duration_std=1.0)
    n_items = 5
    meta_tensors = {
        "language": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        "duration": torch.zeros(n_items + 1, dtype=torch.float32),
        "duration_missing": torch.ones(n_items + 1, dtype=torch.bool),
    }
    text_embeddings = torch.zeros(n_items + 1, 16)
    encoder = ItemEncoder(
        n_items=n_items, hidden_dim=8,
        metadata_tensors=meta_tensors,
        text_embeddings=text_embeddings,
        use_structured=True, use_text=True,
    )
    state = encoder.state_dict()
    bad = [k for k in state if k.startswith("meta_") or k == "text_emb"]
    assert not bad, f"Persistent buffers leaked into state_dict: {bad}"

