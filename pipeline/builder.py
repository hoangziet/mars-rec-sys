"""
pipeline/builder.py
===================
Factory functions shared between scripts/train.py and scripts/train_all.py.

Functions:
    build_model()        — instantiate model by name
    build_criterion_fn() — return a (model, batch, device) -> loss callable
    build_eval_fn()      — return the correct evaluate_* function
    build_train_loader() — return a training DataLoader
"""

from pathlib import Path

import torch

from pipeline.loaders import get_train_loader
from pipeline.metrics import (
    evaluate_bert4rec,
    evaluate_bprmf,
    evaluate_sequential,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(
    model_name: str,
    n_items: int,
    n_users: int,
    model_kwargs: dict,
    max_len: int,
) -> torch.nn.Module:
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


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------


def build_criterion_fn(model_name: str, train_kwargs: dict):
    """Return a callable ``(model, batch, device) -> loss scalar``."""

    if model_name == "sasrec":
        def fn(model, batch, device):
            return model.loss(
                batch["input_seq"].to(device),
                batch["pos_items"].to(device),
                batch["neg_items"].to(device),
            )
        return fn

    if model_name == "gsasrec":
        def fn(model, batch, device):
            return model.loss(
                batch["input_seq"].to(device),
                batch["pos_items"].to(device),
                batch["neg_items"].to(device),
                confidence=batch["confidence"].to(device),
            )
        return fn

    if model_name == "gru4rec":
        def fn(model, batch, device):
            return model.loss(
                batch["input_seq"].to(device),
                batch["pos_items"].to(device),
            )
        return fn

    if model_name == "bert4rec":
        import torch.nn.functional as F

        def fn(model, batch, device):
            logits = model(batch["input_seq"].to(device))
            labels = batch["labels"].to(device)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0,
            )
        return fn

    if model_name == "bprmf":
        reg = train_kwargs.get("reg_lambda", 1e-4)

        def fn(model, batch, device):
            return model.loss(
                batch["user"].to(device),
                batch["pos_item"].to(device),
                batch["neg_item"].to(device),
                reg_lambda=reg,
            )
        return fn

    raise ValueError(f"No criterion defined for model: {model_name}")


# ---------------------------------------------------------------------------
# Eval function
# ---------------------------------------------------------------------------


def build_eval_fn(model_name: str):
    """Return the correct ``evaluate_*(model, loader, device) -> dict`` fn."""

    if model_name in ("sasrec", "gsasrec", "gru4rec"):
        return lambda model, loader, device: evaluate_sequential(model, loader, device)

    if model_name == "bert4rec":
        return lambda model, loader, device: evaluate_bert4rec(model, loader, device)

    if model_name == "bprmf":
        return lambda model, loader, device: evaluate_bprmf(model, loader, device)

    raise ValueError(f"No eval fn defined for model: {model_name}")


# ---------------------------------------------------------------------------
# Train loader
# ---------------------------------------------------------------------------


def build_train_loader(
    model_name: str,
    data_dir: Path,
    stats: dict,
    train_kwargs: dict,
):
    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)
    num_neg    = train_kwargs.get("num_neg", 1)

    if model_name in ("sasrec", "gsasrec", "gru4rec", "bert4rec"):
        return get_train_loader(
            model_name,
            data_dir / "train.csv",
            stats,
            batch_size=batch_size,
            max_len=max_len,
            use_confidence=(model_name == "gsasrec"),
            num_neg=num_neg,
        )

    if model_name == "bprmf":
        return get_train_loader(
            "bprmf",
            data_dir / "train.csv",
            stats,
            batch_size=batch_size,
            max_len=max_len,
        )

    return None
