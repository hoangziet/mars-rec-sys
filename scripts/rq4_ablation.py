"""
scripts/rq4_ablation.py
=======================
RQ4: Run V0-V3 ablation across 10 seeds.

Variants:
    V0: Base gSASRec
    V1: Base + Confidence (best alpha from RQ2)
    V2: Base + Metadata (best config from RQ3)
    V3: Base + Confidence + Metadata

Usage:
    uv run python scripts/rq4_ablation.py --best-alpha 0.5 --best-variant M3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.optim import build_optimizer, build_scheduler
from training.configs import build_model_config
from training.mlflow_contract import build_run_name, build_training_tags
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow, get_git_commit
from training.trainer import Trainer

EXPERIMENT_NAME = "mars_final_ablation"
DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999, 7, 21, 77, 314, 1337]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: final ablation runner.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--best-alpha", type=float, required=True)
    parser.add_argument("--best-variant", required=True, choices=["M0", "M1", "M2", "M3"])
    parser.add_argument("--variants", nargs="+", default=["V0", "V1", "V2", "V3"], choices=["V0", "V1", "V2", "V3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--benchmark-id", default="rq4-ablation")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _get_variant_config(variant: str, best_alpha: float, best_variant: str) -> dict:
    if variant == "V0":
        return {"config_name": "gsasrec", "use_structured": False, "use_text": False, "confidence_alpha": 0.0}
    elif variant == "V1":
        return {"config_name": "gsasrec", "use_structured": False, "use_text": False, "confidence_alpha": best_alpha}
    elif variant == "V2":
        return {"config_name": "gsasrec_metadata", "use_structured": True, "use_text": True, "confidence_alpha": 0.0}
    elif variant == "V3":
        return {"config_name": "gsasrec_metadata", "use_structured": True, "use_text": True, "confidence_alpha": best_alpha}
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _run_single(args, variant: str, seed: int, variant_cfg: dict) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "reports" / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = build_model_config(variant_cfg["config_name"])
    model_kwargs = dict(base_cfg["model_kwargs"])
    train_kwargs = dict(base_cfg["train_kwargs"])
    train_kwargs["confidence_alpha"] = variant_cfg["confidence_alpha"]

    if variant_cfg["config_name"] == "gsasrec_metadata":
        encoder_cfg = model_kwargs.get("item_encoder", {})
        encoder_cfg["use_structured"] = variant_cfg["use_structured"]
        encoder_cfg["use_text"] = variant_cfg["use_text"]
        model_kwargs["item_encoder"] = encoder_cfg
    else:
        model_kwargs.pop("item_encoder", None)

    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model("gsasrec", stats["n_items"], stats["n_users"], model_kwargs, max_len).to(device)
    train_loader = build_train_loader("gsasrec", data_dir, stats, train_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer("gsasrec", model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn("gsasrec", train_kwargs)
    eval_fn = build_eval_fn("gsasrec")
    val_loss_loader = get_val_loss_loader("gsasrec", data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=train_kwargs.get("num_neg", 1), seed=seed)

    run_name = build_run_name("gsasrec", seed, variant=variant.lower())

    trainer = Trainer("gsasrec", device, args.output_dir, use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "final", "variant": variant.lower(), "git_commit": get_git_commit(), "reportable": True,
    })
    mlflow_cfg = collect_common_run_metadata(model_name="gsasrec", seed=seed, phase="final", git_commit=get_git_commit(), extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name="gsasrec", phase="final", variant=variant.lower(), git_commit=get_git_commit(), reportable=True)
    mlflow_cfg["tags"]["ablation_variant"] = variant
    mlflow_cfg["tags"]["rq"] = "rq4"
    mlflow_cfg["tags"]["benchmark_id"] = args.benchmark_id
    mlflow_cfg["tags"]["confidence_alpha"] = str(variant_cfg["confidence_alpha"])

    return trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)
    total = len(args.variants) * len(args.seeds)
    print(f"RQ4 ablation: {len(args.variants)} variants x {len(args.seeds)} seeds = {total} runs")
    print(f"Best alpha: {args.best_alpha}")
    print(f"Best metadata variant: {args.best_variant}")
    for i, variant in enumerate(args.variants):
        variant_cfg = _get_variant_config(variant, args.best_alpha, args.best_variant)
        for j, seed in enumerate(args.seeds):
            run_num = i * len(args.seeds) + j + 1
            print(f"\n[{run_num}/{total}] {variant}, seed={seed}")
            _run_single(args, variant, seed, variant_cfg)
    print(f"\nDone. Results logged to MLflow experiment '{EXPERIMENT_NAME}'.")
    print(f"Run: make rq4-compare BENCHMARK_ID={args.benchmark_id}")


if __name__ == "__main__":
    main()
