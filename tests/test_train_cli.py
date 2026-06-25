import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import train, train_all
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


def test_train_main_passes_model_kwargs_num_neg_to_loaders(monkeypatch, tmp_path):
    captured = {}

    data_dir = tmp_path / "processed"
    (data_dir / "reports").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)
    (data_dir / "reports" / "dataset_stats.json").write_text('{"n_items": 10, "n_users": 5}')
    for name in ("train_sequences.csv", "val_sequences.csv", "test_sequences.csv"):
        (data_dir / "splits" / name).write_text("user_idx,item_sequence,target_item\n1,1 2 3,4\n")

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "output_dir": str(tmp_path / "out"),
            "db": {"data_dir": str(data_dir), "max_len": 50},
            "model": {
                "name": "gsasrec",
                "model_kwargs": {"hidden_dim": 8, "num_neg": 32},
                "train_kwargs": {"epochs": 1, "batch_size": 2, "max_len": 5},
            },
        }
    )

    monkeypatch.setattr(train, "load_stats", lambda *_args, **_kwargs: {"n_items": 10, "n_users": 5})
    monkeypatch.setattr(train, "build_model", lambda *_args, **_kwargs: __import__("torch").nn.Linear(1, 1))

    def fake_build_train_loader(model_name, data_dir, stats, train_kwargs, model_kwargs=None):
        captured["train_loader_model_kwargs"] = model_kwargs
        return [0]

    def fake_get_val_loss_loader(*_args, **kwargs):
        captured["val_loss_num_neg"] = kwargs["num_neg"]
        return [0]

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, *args, **kwargs):
            return None

    monkeypatch.setattr(train, "build_train_loader", fake_build_train_loader)
    monkeypatch.setattr(train, "get_eval_loader", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(train, "get_val_loss_loader", fake_get_val_loss_loader)
    monkeypatch.setattr(train, "build_optimizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train, "build_scheduler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "build_criterion_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "build_eval_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train, "collect_common_run_metadata", lambda **_kwargs: {})
    monkeypatch.setattr(train, "build_training_tags", lambda **_kwargs: {})
    monkeypatch.setattr(train, "build_run_name", lambda *_args, **_kwargs: "run")
    monkeypatch.setattr(train, "get_experiment_name_for_phase", lambda *_args, **_kwargs: "exp")
    train.main.__wrapped__(cfg)

    assert captured["train_loader_model_kwargs"]["num_neg"] == 32
    assert captured["val_loss_num_neg"] == 32


def test_train_all_run_neural_model_passes_model_kwargs_num_neg_to_loaders(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(train_all, "build_model", lambda *_args, **_kwargs: __import__("torch").nn.Linear(1, 1))

    def fake_build_train_loader(model_name, data_dir, stats, train_kwargs, model_kwargs=None):
        captured["train_loader_model_kwargs"] = model_kwargs
        return [0]

    def fake_get_val_loss_loader(*_args, **kwargs):
        captured["val_loss_num_neg"] = kwargs["num_neg"]
        return [0]

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, *args, **kwargs):
            class _T:
                def summary(self):
                    return {"best_val_ndcg": 0.1, "test_results": {}}

            return _T()

    monkeypatch.setattr(train_all, "build_train_loader", fake_build_train_loader)
    monkeypatch.setattr(train_all, "get_eval_loader", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(train_all, "get_val_loss_loader", fake_get_val_loss_loader)
    monkeypatch.setattr(train_all, "build_optimizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_all, "build_scheduler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_all, "build_criterion_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_all, "build_eval_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_all, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_all, "collect_common_run_metadata", lambda **_kwargs: {})
    monkeypatch.setattr(train_all, "build_training_tags", lambda **_kwargs: {})
    monkeypatch.setattr(train_all, "build_run_name", lambda *_args, **_kwargs: "run")
    monkeypatch.setattr(train_all, "get_experiment_name_for_phase", lambda *_args, **_kwargs: "exp")
    summary = train_all.run_neural_model(
        model_name="gsasrec",
        data_dir=tmp_path / "processed",
        stats={"n_items": 10, "n_users": 5},
        device=__import__("torch").device("cpu"),
        output_dir=str(tmp_path / "out"),
        benchmark_id="rq1-smoke",
        protocol_version="rq1-v1",
        preprocessing_version="mars-preprocess-v1",
        model_kwargs={"hidden_dim": 8, "num_neg": 32},
        train_kwargs={"epochs": 1, "batch_size": 2, "max_len": 5},
        seed=42,
    )

    assert summary["best_val_ndcg"] == 0.1
    assert captured["train_loader_model_kwargs"]["num_neg"] == 32
    assert captured["val_loss_num_neg"] == 32
