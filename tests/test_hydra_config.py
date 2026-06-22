from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").absolute()


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_default_model_is_sasrec():
    cfg = _compose([])
    assert cfg.model.name == "sasrec"
    assert cfg.model.train_kwargs.batch_size == 256


def test_sasrec_config_values():
    cfg = _compose(["model=sasrec"])
    assert cfg.model.model_kwargs.hidden_dim == 64
    assert cfg.model.model_kwargs.num_heads == 2
    assert cfg.model.model_kwargs.num_layers == 2
    assert cfg.model.model_kwargs.dropout == 0.2
    assert cfg.model.model_kwargs.norm_first is True


def test_gsasrec_config_values():
    cfg = _compose(["model=gsasrec"])
    assert cfg.model.model_kwargs.num_neg == 32
    assert cfg.model.model_kwargs.t == 0.5


def test_bert4rec_config_values():
    cfg = _compose(["model=bert4rec"])
    assert cfg.model.train_kwargs.mask_ratio == 0.2
    assert cfg.model.train_kwargs.force_last_item_mask is True
    assert cfg.model.train_kwargs.warmup_steps == 100


def test_all_neural_models_have_confidence_alpha():
    for model_name in ("sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"):
        cfg = _compose([f"model={model_name}"])
        assert "confidence_alpha" in cfg.model.train_kwargs, model_name


def test_cli_override_batch_size():
    cfg = _compose(["model=sasrec", "model.train_kwargs.batch_size=128"])
    assert cfg.model.train_kwargs.batch_size == 128


def test_cli_override_epochs_and_lr():
    cfg = _compose([
        "model=gru4rec",
        "model.train_kwargs.epochs=100",
        "model.train_kwargs.lr=5e-5",
    ])
    assert cfg.model.train_kwargs.epochs == 100
    assert cfg.model.train_kwargs.lr == pytest.approx(5e-5)


def test_data_dir_default():
    cfg = _compose(["model=sasrec"])
    assert cfg.db.data_dir == "data/processed"


def test_seed_default():
    cfg = _compose(["model=sasrec"])
    assert cfg.seed == 42


def test_output_dir_default():
    cfg = _compose(["model=sasrec"])
    assert cfg.output_dir == "experiments"


def test_phase_and_reportable_defaults():
    cfg = _compose(["model=sasrec"])
    assert cfg.phase == "benchmark"
    assert cfg.reportable is True


def test_cli_override_phase_and_reportable():
    cfg = _compose(["model=sasrec", "phase=smoke", "reportable=false"])
    assert cfg.phase == "smoke"
    assert cfg.reportable is False
