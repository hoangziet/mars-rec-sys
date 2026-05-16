"""
scripts/train.py
================
Train a single model with full tracking.

Usage:
    uv run python scripts/train.py sasrec
    uv run python scripts/train.py gru4rec --epochs 50 --lr 5e-4
    uv run python scripts/train.py bprmf --seed 123
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.configs import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, MODEL_CONFIGS
from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.optim import build_optimizer, build_scheduler
from training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs",     type=int)
    parser.add_argument("--lr",         type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    stats    = load_stats(data_dir / "dataset_stats.json")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg          = MODEL_CONFIGS[args.model]
    model_kwargs = cfg["model_kwargs"].copy()
    train_kwargs = cfg["train_kwargs"].copy()

    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size

    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model        = build_model(args.model, stats["n_items"], stats["n_users"], model_kwargs, max_len).to(device)
    train_loader = build_train_loader(args.model, data_dir, stats, train_kwargs)
    val_loader   = get_eval_loader(data_dir / "val.csv",  stats, batch_size=batch_size, max_len=max_len)
    test_loader  = get_eval_loader(data_dir / "test.csv", stats, batch_size=batch_size, max_len=max_len)

    if args.model not in MODEL_CONFIGS:
        print(f"\nModel '{args.model}' not found.")
        sys.exit(1)

    optimizer = build_optimizer(args.model, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))

    criterion_fn   = build_criterion_fn(args.model, train_kwargs)
    eval_fn        = build_eval_fn(args.model)
    val_loss_loader = get_val_loss_loader(
        args.model,
        data_dir / "val.csv",
        stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=train_kwargs.get("num_neg", 1),
        seed=args.seed,
    )

    trainer = Trainer(args.model, device, args.output_dir)
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
    )


if __name__ == "__main__":
    main()
