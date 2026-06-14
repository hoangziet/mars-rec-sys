"""
scripts/train.py
===============
Train a single model with Hydra config + MLflow tracking.

Usage:
    uv run python scripts/train.py model=sasrec
    uv run python scripts/train.py model=gru4rec model.train_kwargs.epochs=100 model.train_kwargs.lr=5e-4
    uv run python scripts/train.py model=bprmf seed=123
"""

import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.optim import build_optimizer, build_scheduler
from training.mlflow_contract import build_run_name, build_training_tags, get_experiment_name_for_phase
from training.mlflow_utils import collect_common_run_metadata, get_dataset_version, get_git_commit
from training.trainer import Trainer

TRAINABLE_MODELS = ("sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_processed_layout(data_dir: Path) -> None:
    required = [
        data_dir / "reports" / "dataset_stats.json",
        data_dir / "splits" / "train_sequences.csv",
        data_dir / "splits" / "val_sequences.csv",
        data_dir / "splits" / "test_sequences.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed artifacts: {missing}")



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    if model_name not in TRAINABLE_MODELS:
        print(f"Model '{model_name}' is not trainable via train.py. Use train_all.py for heuristics.")
        sys.exit(1)

    seed_everything(cfg.seed)

    data_dir = Path(cfg.db.data_dir)
    validate_processed_layout(data_dir)
    stats = load_stats(data_dir / "reports" / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(cfg.model.model_kwargs)
    train_kwargs = dict(cfg.model.train_kwargs)

    max_len = train_kwargs.get("max_len", cfg.db.max_len)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model(
        model_name, stats["n_items"], stats["n_users"], model_kwargs, max_len
    ).to(device)

    train_loader = build_train_loader(model_name, data_dir, stats, train_kwargs)
    val_loader = get_eval_loader(
        data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len
    )
    test_loader = get_eval_loader(
        data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len
    )

    optimizer = build_optimizer(model_name, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))

    criterion_fn = build_criterion_fn(model_name, train_kwargs)
    eval_fn = build_eval_fn(model_name)
    val_loss_loader = get_val_loss_loader(
        model_name,
        data_dir / "splits" / "val_sequences.csv",
        stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=train_kwargs.get("num_neg", 1),
        seed=cfg.seed,
    )

    phase = "benchmark"
    experiment_name = get_experiment_name_for_phase(phase)
    stats_path = data_dir / "reports" / "dataset_stats.json"
    dataset_version = get_dataset_version(stats_path)
    run_name = build_run_name(model_name, cfg.seed, variant="base")

    trainer = Trainer(
        model_name,
        device,
        cfg.output_dir,
        use_mlflow=True,
        mlflow_config={
            "experiment_name": experiment_name,
            "run_name": run_name,
            "log_artifacts": True,
            "phase": phase,
            "variant": "base",
            "dataset_name": "mars",
            "dataset_version": dataset_version,
            "git_commit": get_git_commit(),
            "reportable": True,
        },
    )

    mlflow_cfg = collect_common_run_metadata(
        model_name=model_name,
        seed=cfg.seed,
        phase=phase,
        git_commit=get_git_commit(),
        dataset_version=dataset_version,
        extra_params={**model_kwargs, **train_kwargs},
    )
    mlflow_cfg["tags"] = build_training_tags(
        model_name=model_name,
        phase=phase,
        variant="base",
        git_commit=get_git_commit(),
        dataset_name="mars",
        dataset_version=dataset_version,
        reportable=True,
    )

    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=train_kwargs["epochs"],
        criterion_fn=criterion_fn,
        eval_fn=eval_fn,
        gradient_clip=train_kwargs.get("gradient_clip", 5.0),
        val_loss_loader=val_loss_loader,
        early_stop_patience=train_kwargs.get("early_stop_patience", 0),
        early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4),
        scheduler=scheduler,
        mlflow_params=mlflow_cfg,
    )


if __name__ == "__main__":
    main()
