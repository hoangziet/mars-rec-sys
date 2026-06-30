"""
scripts/rq2_compare_variants.py
================================
RQ2 Stage B: Compare BERT4Rec watch-integration variants.
Runs baseline, WL, WE, WLWE with fixed best alpha across shared seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.optim import build_optimizer, build_scheduler
from scripts.study_manifest import create_manifest, finalize_manifest, is_completed, mark_completed
from training.configs import build_model_config
from training.mlflow_contract import RQ2_VARIANT_EXPERIMENT_NAME, build_run_name, build_training_tags
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow
from training.repro import seed_everything
from training.trainer import Trainer

EXPERIMENT_NAME = RQ2_VARIANT_EXPERIMENT_NAME
DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999]
BACKBONE = "bert4rec"

VARIANTS = {
    "baseline": {"watch_mode": "none", "watch_alpha": 0.0, "label": "No watch"},
    "wl": {"watch_mode": "loss", "label": "Weighted loss"},
    "we": {"watch_mode": "embedding", "watch_alpha": 0.0, "label": "Watch embedding"},
    "wlwe": {"watch_mode": "both", "label": "Both"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2: BERT4Rec watch variant comparison.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--variants", nargs="+", default=["baseline", "wl", "we", "wlwe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--benchmark-id", default="rq2-watch-variants")
    parser.add_argument("--alpha-artifact", required=True,
        help="Path to rq2_best_alpha.json from Stage A")
    parser.add_argument("--preprocessing-version", default="mars-preprocess-v1")
    return parser


def _run_single(args, variant_name: str, best_alpha: float, seed: int) -> dict:
    backbone = BACKBONE
    variant_cfg = VARIANTS[variant_name]

    seed_everything(seed)

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "reports" / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = build_model_config(backbone)
    model_kwargs = dict(base_cfg["model_kwargs"])
    train_kwargs = dict(base_cfg["train_kwargs"])
    train_kwargs.pop("confidence_alpha", None)

    model_kwargs["watch_mode"] = variant_cfg["watch_mode"]
    model_kwargs["watch_num_bins"] = model_kwargs.get("watch_num_bins", 5)
    if variant_name in ("wl", "wlwe"):
        model_kwargs["watch_alpha"] = best_alpha
    else:
        model_kwargs["watch_alpha"] = variant_cfg.get("watch_alpha", 0.0)

    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model(backbone, stats["n_items"], stats["n_users"], model_kwargs, max_len, data_dir=data_dir).to(device)
    train_loader = build_train_loader(backbone, data_dir, stats, train_kwargs, model_kwargs=model_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer(backbone, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn(backbone, train_kwargs)
    eval_fn = build_eval_fn(backbone)
    val_loss_loader = get_val_loss_loader(backbone, data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)), seed=seed)

    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)

    run_name = build_run_name(backbone, seed, variant=variant_name)
    run_output_dir = Path(args.output_dir) / "rq2" / args.benchmark_id / variant_name / f"seed_{seed}"

    trainer = Trainer(backbone, device, str(run_output_dir), use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "benchmark", "variant": variant_name, "reportable": True,
    })
    mlflow_cfg = collect_common_run_metadata(model_name=backbone, seed=seed, phase="benchmark", extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name=backbone, phase="benchmark", variant=variant_name, reportable=True)
    mlflow_cfg["tags"]["watch_mode"] = variant_cfg["watch_mode"]
    mlflow_cfg["tags"]["watch_alpha"] = str(model_kwargs["watch_alpha"])
    mlflow_cfg["tags"]["rq"] = "rq2"
    mlflow_cfg["tags"]["benchmark_id"] = args.benchmark_id
    mlflow_cfg["tags"]["preprocessing_version"] = args.preprocessing_version
    mlflow_cfg["tags"]["data_source"] = str(data_dir.resolve())

    return trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)


def main() -> None:
    args = build_parser().parse_args()
    configure_mlflow(mlflow_module=mlflow)

    alpha_data = json.loads(Path(args.alpha_artifact).read_text())
    best_alpha = float(alpha_data["best_alpha"])
    print(f"RQ2 variant comparison: BERT4Rec with best alpha={best_alpha}")

    manifest_path = Path(args.output_dir) / "rq2" / args.benchmark_id / "benchmark_manifest.json"
    if not manifest_path.exists():
        create_manifest(
            manifest_path,
            variants=args.variants,
            seeds=args.seeds,
            benchmark_id=args.benchmark_id,
            backbone=BACKBONE,
        )

    total = len(args.variants) * len(args.seeds)
    for i, variant in enumerate(args.variants):
        for j, seed in enumerate(args.seeds):
            run_num = i * len(args.seeds) + j + 1
            if is_completed(manifest_path, variant, seed):
                print(f"\n[{run_num}/{total}] SKIP variant={variant}, seed={seed} (already completed)")
                continue
            print(f"\n[{run_num}/{total}] variant={variant}, seed={seed}")
            _run_single(args, variant, best_alpha, seed)
            mark_completed(manifest_path, variant, seed)

    finalize_manifest(manifest_path)
    print(f"\nDone. Run: make rq2-report RQ2_VARIANT_BENCHMARK_ID={args.benchmark_id}")


if __name__ == "__main__":
    main()
