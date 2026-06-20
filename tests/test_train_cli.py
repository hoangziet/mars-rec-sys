import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import TRAINABLE_MODELS, resolve_dataset_context, validate_processed_layout


def test_trainable_models_exclude_heuristics():
    assert "itemcf" not in TRAINABLE_MODELS
    assert "popularity" not in TRAINABLE_MODELS


def test_validate_processed_layout_passes_with_canonical_structure(tmp_path):
    data_dir = tmp_path / "processed"
    (data_dir / "reports").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)
    (data_dir / "reports" / "dataset_stats.json").write_text('{"n_items": 5, "n_users": 3}')
    (data_dir / "splits" / "train_sequences.csv").write_text("user_idx,item_sequence\n")
    (data_dir / "splits" / "val_sequences.csv").write_text("user_idx,item_sequence\n")
    (data_dir / "splits" / "test_sequences.csv").write_text("user_idx,item_sequence\n")

    validate_processed_layout(data_dir)


def test_validate_processed_layout_raises_on_missing_artifacts(tmp_path):
    data_dir = tmp_path / "processed"
    (data_dir / "reports").mkdir(parents=True)

    import pytest
    with pytest.raises(FileNotFoundError, match="Missing processed artifacts"):
        validate_processed_layout(data_dir)


def test_resolve_dataset_context_uses_actual_data_dir(tmp_path):
    data_dir = tmp_path / "processed"
    context = resolve_dataset_context(data_dir)

    assert context == {"data_source": str(data_dir.resolve())}
