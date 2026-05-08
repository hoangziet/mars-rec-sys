#!/usr/bin/env python3
"""
run_experiment.py
=================
Train a single model with full tracking.

Usage:
    uv run python run_experiment.py sasrec
    uv run python run_experiment.py gru4rec --epochs 50 --lr 5e-4
    uv run python run_experiment.py bprmf --seed 123
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from configs import MODEL_CONFIGS, DEFAULT_SEED, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from dataloader import get_train_loader, get_eval_loader, load_stats
from evaluate import evaluate_sequential, evaluate_bert4rec, evaluate_bprmf, evaluate_popularity, evaluate_itemcf
from trainer import Trainer


def build_model(model_name, n_items, n_users, model_kwargs, max_len):
    if model_name == "sasrec":
        from models.sasrec import SASRec
        return SASRec(n_items=n_items, max_len=max_len, **model_kwargs)

    if model_name == "gsasrec":
        from models.gsasrec import GSASRec
        return GSASRec(n_items=n_items, max_len=max_len, **model_kwargs)

    if model_name == "gru4rec":
        from models.gru4rec import GRU4Rec
        return GRU4Rec(n_items=n_items, **model_kwargs)

    if model_name == "bert4rec":
        from models.bert4rec import BERT4Rec
        return BERT4Rec(n_items=n_items, max_len=max_len, **model_kwargs)

    if model_name == "bprmf":
        from models.bprmf import BPRMF
        return BPRMF(n_users=n_users, n_items=n_items, **model_kwargs)

    raise ValueError(f"Unknown model: {model_name}")


def build_criterion_fn(model_name, train_kwargs):
    if model_name in ("sasrec", "gru4rec"):
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        return lambda model, batch, device: criterion(
            model(batch["input_seq"].to(device), mask=batch.get("mask", None).to(device) if batch.get("mask") is not None else None),
            batch["target"].to(device)
        )

    if model_name == "gsasrec":
        from models.gsasrec import weighted_ce_loss
        def fn(model, batch, device):
            logits = model(batch["input_seq"].to(device), mask=batch["mask"].to(device))
            return weighted_ce_loss(logits, batch["target"].to(device), batch["confidence"].to(device))
        return fn

    if model_name == "bert4rec":
        import torch.nn.functional as F
        def fn(model, batch, device):
            logits = model(batch["input_seq"].to(device))
            labels = batch["labels"].to(device)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
        return fn

    if model_name == "bprmf":
        from models.bprmf import bpr_loss
        reg = train_kwargs.get("reg_lambda", 1e-4)
        def fn(model, batch, device):
            pos, neg = model(
                batch["user"].to(device),
                batch["pos_item"].to(device),
                batch["neg_item"].to(device)
            )
            return bpr_loss(pos, neg, reg_lambda=reg, model=model)
        return fn

    raise ValueError(f"No criterion for: {model_name}")


def build_eval_fn(model_name):
    if model_name in ("sasrec", "gsasrec", "gru4rec"):
        return lambda model, loader, device: evaluate_sequential(model, loader, device)

    if model_name == "bert4rec":
        return lambda model, loader, device: evaluate_bert4rec(model, loader, device)

    if model_name == "bprmf":
        return lambda model, loader, device: evaluate_bprmf(model, loader, device)

    raise ValueError(f"No eval fn for: {model_name}")


def build_train_loader(model_name, data_dir, stats, train_kwargs):
    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    if model_name in ("sasrec", "gsasrec", "gru4rec", "bert4rec"):
        return get_train_loader(
            model_name, data_dir / "train.csv", stats,
            batch_size=batch_size, max_len=max_len,
            use_confidence=(model_name == "gsasrec")
        )

    if model_name == "bprmf":
        return get_train_loader(
            "bprmf", data_dir / "train.csv", stats,
            batch_size=batch_size, max_len=max_len
        )

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "dataset_stats.json")
    n_items = stats["n_items"]
    n_users = stats["n_users"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MODEL_CONFIGS[args.model]
    model_kwargs = cfg["model_kwargs"].copy()
    train_kwargs = cfg["train_kwargs"].copy()

    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size

    max_len = train_kwargs.get("max_len", 50)

    model = build_model(args.model, n_items, n_users, model_kwargs, max_len).to(device)
    train_loader = build_train_loader(args.model, data_dir, stats, train_kwargs)
    val_loader = get_eval_loader(data_dir / "val.csv", stats, batch_size=train_kwargs.get("batch_size", 256), max_len=max_len)
    test_loader = get_eval_loader(data_dir / "test.csv", stats, batch_size=train_kwargs.get("batch_size", 256), max_len=max_len)

    if args.model in ("sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"):
        optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs.get("lr", 1e-3))
        criterion_fn = build_criterion_fn(args.model, train_kwargs)
        eval_fn = build_eval_fn(args.model)

        trainer = Trainer(args.model, device, args.output_dir)
        tracker = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=train_kwargs["epochs"],
            criterion_fn=criterion_fn,
            eval_fn=eval_fn,
            gradient_clip=train_kwargs.get("gradient_clip", 5.0),
        )
    else:
        print(f"\nModel '{args.model}' is heuristic-based — run via run_all.py for comparison.")
        sys.exit(0)


if __name__ == "__main__":
    main()
