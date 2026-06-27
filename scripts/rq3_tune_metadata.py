"""
scripts/rq3_tune_metadata.py
=============================
RQ3: Compare metadata variants M0-M3 (BERT4Rec-only, on top of RQ2 winner).

Variants:
    M0: Base BERT4Rec (no metadata)
    M1: Structured metadata only
    M2: Text embeddings only
    M3: Structured + Text

Seeds: {42, 123, 2024} per variant
Selection: highest mean validation NDCG@10
Tie-break: prefer simpler config (fewer params)
Watch: locked to RQ2 winner configuration

Usage:
    uv run python scripts/rq3_tune_metadata.py --rq2-winner path/to/rq2_best_watch.json
"""

from __future__ import annotations

import argparse
import json
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
from training.mlflow_contract import RQ3_EXPERIMENT_NAME, build_run_name, build_training_tags
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow
from training.trainer import Trainer

EXPERIMENT_NAME = RQ3_EXPERIMENT_NAME
DEFAULT_SEEDS = [42, 123, 2024]
BACKBONE = "bert4rec"

VARIANTS = {
    "M0": {"config_name": "bert4rec", "use_structured": False, "use_text": False, "label": "Base"},
    "M1": {"config_name": "bert4rec_metadata", "use_structured": True, "use_text": False, "label": "Structured"},
    "M2": {"config_name": "bert4rec_metadata", "use_structured": False, "use_text": True, "label": "Text"},
    "M3": {"config_name": "bert4rec_metadata", "use_structured": True, "use_text": True, "label": "Both"},
}


def load_rq2_winner(path: Path) -> dict:
    """Load RQ2 watch winner artifact and validate backbone."""
    data = json.loads(path.read_text())
    if data.get("backbone") != "bert4rec":
        raise RuntimeError(
            f"RQ3 requires bert4rec RQ2 winner, got {data.get('backbone')!r}"
        )
    return data


def apply_watch_winner(model_kwargs: dict, winner: dict) -> dict:
    """Apply RQ2 watch configuration to model kwargs."""
    updated = dict(model_kwargs)
    best_variant = winner["best_variant"]
    updated["watch_num_bins"] = updated.get("watch_num_bins", 5)
    if best_variant == "baseline":
        updated["watch_mode"] = "none"
        updated["watch_alpha"] = 0.0
    elif best_variant == "wl":
        updated["watch_mode"] = "loss"
        updated["watch_alpha"] = winner.get("best_alpha", 1.0)
    elif best_variant == "we":
        updated["watch_mode"] = "embedding"
        updated["watch_alpha"] = 0.0
    elif best_variant == "wlwe":
        updated["watch_mode"] = "both"
        updated["watch_alpha"] = winner.get("best_alpha", 1.0)
    else:
        raise ValueError(f"Unknown RQ2 watch variant: {best_variant!r}")
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ3: BERT4Rec metadata variant comparison on top of RQ2 winner.")
    parser.add_argument("--rq2-winner", required=True, help="Path to rq2_best_watch.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--variants", nargs="+", default=["M0", "M1", "M2", "M3"], choices=["M0", "M1", "M2", "M3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--benchmark-id", default="rq3-metadata-tune")
    parser.add_argument("--preprocessing-version", default="mars-preprocess-v1",
                        help="Tracked in MLflow tags and winner artifact for light provenance")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _run_single(args, variant_name: str, seed: int, rq2_winner: dict) -> dict:
    backbone = BACKBONE
    variant = VARIANTS[variant_name]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "reports" / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = build_model_config(variant["config_name"])
    model_kwargs = dict(base_cfg["model_kwargs"])
    model_kwargs = apply_watch_winner(model_kwargs, rq2_winner)
    train_kwargs = dict(base_cfg["train_kwargs"])
    train_kwargs.pop("confidence_alpha", None)

    if variant_name != "M0":
        encoder_cfg = model_kwargs.get("item_encoder", {})
        if not variant["use_structured"]:
            encoder_cfg["use_structured"] = False
        if not variant["use_text"]:
            encoder_cfg["use_text"] = False
        if encoder_cfg.get("use_structured") or encoder_cfg.get("use_text"):
            # Resolve metadata paths from data_dir
            encoder_cfg["metadata_vocab_path"] = str(data_dir / "item_features" / "metadata_vocab.json")
            encoder_cfg["metadata_csv_path"] = str(data_dir / "item_features" / "item_metadata.csv")
            encoder_cfg["text_emb_path"] = str(data_dir / "item_features" / "text_embeddings.pt")
            model_kwargs["item_encoder"] = encoder_cfg
        else:
            model_kwargs.pop("item_encoder", None)
    else:
        model_kwargs.pop("item_encoder", None)

    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model(backbone, stats["n_items"], stats["n_users"], model_kwargs, max_len, data_dir=data_dir).to(device)
    train_loader = build_train_loader(backbone, data_dir, stats, train_kwargs, model_kwargs=model_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer(backbone, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn(backbone, train_kwargs)
    eval_fn = build_eval_fn(backbone)
    val_loss_loader = get_val_loss_loader(backbone, data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)), seed=seed)

    run_name = build_run_name(backbone, seed, variant=variant_name.lower())
    run_output_dir = Path(args.output_dir) / "rq3" / args.benchmark_id / variant_name / f"seed_{seed}"

    trainer = Trainer(backbone, device, str(run_output_dir), use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "tuning", "variant": variant_name.lower(), "reportable": True,
    })
    mlflow_cfg = collect_common_run_metadata(model_name=backbone, seed=seed, phase="tuning", extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name=backbone, phase="tuning", variant=variant_name.lower(), reportable=True)
    mlflow_cfg["tags"]["metadata_variant"] = variant_name
    mlflow_cfg["tags"]["rq"] = "rq3"
    mlflow_cfg["tags"]["benchmark_id"] = args.benchmark_id
    mlflow_cfg["tags"]["preprocessing_version"] = args.preprocessing_version
    mlflow_cfg["tags"]["data_source"] = str(data_dir.resolve())

    return trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)
    backbone = BACKBONE
    rq2_winner = load_rq2_winner(Path(args.rq2_winner))
    print(f"RQ2 winner: variant={rq2_winner['best_variant']}, alpha={rq2_winner.get('best_alpha')}")

    total = len(args.variants) * len(args.seeds)
    print(f"RQ3 metadata tuning (BERT4Rec-only): {len(args.variants)} variants x {len(args.seeds)} seeds = {total} runs")
    print(f"Variants: {args.variants}")
    print(f"Seeds:    {args.seeds}")
    for i, variant in enumerate(args.variants):
        for j, seed in enumerate(args.seeds):
            run_num = i * len(args.seeds) + j + 1
            print(f"\n[{run_num}/{total}] backbone={backbone}, variant={variant}, seed={seed}")
            _run_single(args, variant, seed, rq2_winner)
    print(f"\nDone. Results logged to MLflow experiment '{EXPERIMENT_NAME}'.")
    print(f"Run: make rq3-report RQ3_BENCHMARK_ID={args.benchmark_id}")


if __name__ == "__main__":
    main()
