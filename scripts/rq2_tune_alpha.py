"""
scripts/rq2_tune_alpha.py
==========================
RQ2: Confidence alpha grid search.

Runs gSASRec with different alpha values across seeds.
Logs results to MLflow experiment 'mars_confidence_tuning'.

Usage:
    uv run python scripts/rq2_tune_alpha.py
    uv run python scripts/rq2_tune_alpha.py --alphas 0.0 0.25 0.5 --seeds 42 123
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

EXPERIMENT_NAME = "mars_confidence_tuning"
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
DEFAULT_SEEDS = [42, 123, 2024]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2: confidence alpha grid search.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--benchmark-id", default="rq2-alpha-tune")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _run_single(args, alpha: float, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "reports" / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = build_model_config("gsasrec")
    model_kwargs = dict(base_cfg["model_kwargs"])
    train_kwargs = dict(base_cfg["train_kwargs"])
    train_kwargs["confidence_alpha"] = alpha

    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model("gsasrec", stats["n_items"], stats["n_users"], model_kwargs, max_len).to(device)
    train_loader = build_train_loader("gsasrec", data_dir, stats, train_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer("gsasrec", model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn("gsasrec", train_kwargs)
    eval_fn = build_eval_fn("gsasrec")
    val_loss_loader = get_val_loss_loader("gsasrec", data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=train_kwargs.get("num_neg", 1), seed=seed)

    variant = f"alpha-{alpha}"
    run_name = build_run_name("gsasrec", seed, variant=variant, alpha=alpha)
    run_output_dir = Path(args.output_dir) / "rq2" / args.benchmark_id / variant / f"seed_{seed}"

    trainer = Trainer("gsasrec", device, str(run_output_dir), use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "tuning", "variant": variant, "git_commit": get_git_commit(), "reportable": True,
    })
    mlflow_cfg = collect_common_run_metadata(model_name="gsasrec", seed=seed, phase="tuning", git_commit=get_git_commit(), extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name="gsasrec", phase="tuning", variant=variant, git_commit=get_git_commit(), reportable=True)
    mlflow_cfg["tags"]["confidence_alpha"] = str(alpha)
    mlflow_cfg["tags"]["rq"] = "rq2"
    mlflow_cfg["tags"]["benchmark_id"] = args.benchmark_id

    return trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)
    total = len(args.alphas) * len(args.seeds)
    print(f"RQ2 alpha grid search: {len(args.alphas)} alphas x {len(args.seeds)} seeds = {total} runs")
    print(f"Alphas: {args.alphas}")
    print(f"Seeds:  {args.seeds}")
    for i, alpha in enumerate(args.alphas):
        for j, seed in enumerate(args.seeds):
            run_num = i * len(args.seeds) + j + 1
            print(f"\n[{run_num}/{total}] alpha={alpha}, seed={seed}")
            _run_single(args, alpha, seed)
    print(f"\nDone. Results logged to MLflow experiment '{EXPERIMENT_NAME}'.")
    print(f"Run: make rq2-report BENCHMARK_ID={args.benchmark_id}")


if __name__ == "__main__":
    main()
