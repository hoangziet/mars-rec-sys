"""
scripts/train_all.py
====================
Train all models and produce a comparison report.

Usage:
    uv run python scripts/train_all.py
    uv run python scripts/train_all.py sasrec gru4rec bprmf

Supported models:
    sasrec, gsasrec, gru4rec, bert4rec, bprmf, popularity, itemcf
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.configs import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, MODEL_CONFIGS
from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_rq1_train_criterion_fn, build_train_loader
from pipeline.loaders import get_eval_loader, get_rq1_train_loader, get_val_loss_loader, load_stats
from pipeline.metrics import (
    evaluate_itemcf,
    evaluate_popularity,
    print_results,
)
from pipeline.optim import build_optimizer, build_scheduler
from scripts.train import validate_processed_layout
from training.mlflow_contract import build_run_name, build_training_tags, get_experiment_name_for_phase
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow, sanitize_metric_name
from training.trainer import Trainer

DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999]
NEURAL_MODELS = {"sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"}
HEURISTIC_MODELS = {"popularity", "itemcf"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seeds_for_model(model_name: str, seeds: list[int]) -> list[int]:
    return list(seeds) if model_name in NEURAL_MODELS else [seeds[0]]


def build_benchmark_run_dir(output_dir: str | Path, benchmark_id: str, model_name: str, seed: int) -> Path:
    return Path(output_dir) / "benchmark" / benchmark_id / model_name / f"seed_{seed}"


def build_heuristic_save_target(model_name: str, model_output_dir: Path) -> Path:
    if model_name == "popularity":
        return model_output_dir / "popularity_model.json"
    if model_name == "itemcf":
        return model_output_dir
    raise ValueError(f"Unsupported heuristic model: {model_name}")


def build_benchmark_manifest(
    *,
    benchmark_id: str,
    protocol_version: str,
    preprocessing_version: str,
    expected_models: list[str],
    neural_seeds: list[int],
    heuristic_seed: int,
) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "protocol_version": protocol_version,
        "preprocessing_version": preprocessing_version,
        "expected_models": expected_models,
        "neural_seeds": neural_seeds,
        "heuristic_seed": heuristic_seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all models for benchmark orchestration.")
    parser.add_argument("models", nargs="*", default=None, choices=list(MODEL_CONFIGS.keys()),
                        help="Subset of models to run (default: all).")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--protocol-version", default="rq1-v1")
    parser.add_argument("--preprocessing-version", default="mars-preprocess-v1")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Neural model runner
# ---------------------------------------------------------------------------


def run_neural_model(
    model_name: str,
    data_dir: Path,
    stats: dict,
    device: torch.device,
    output_dir: str,
    benchmark_id: str,
    protocol_version: str,
    preprocessing_version: str,
    model_kwargs: dict,
    train_kwargs: dict,
    seed: int,
) -> dict:
    seed_everything(seed)

    phase = "benchmark"
    experiment_name = get_experiment_name_for_phase(phase)
    run_name = f"{benchmark_id}-{build_run_name(model_name, seed, variant='base')}"
    run_dir = build_benchmark_run_dir(output_dir, benchmark_id, model_name, seed)

    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model        = build_model(model_name, stats["n_items"], stats["n_users"], model_kwargs, max_len).to(device)

    if model_name in {"sasrec", "gsasrec", "gru4rec"}:
        train_loader = get_rq1_train_loader(
            model_type=model_name,
            train_csv=str(data_dir / "splits" / "train_sequences.csv"),
            stats=stats,
            batch_size=batch_size,
            max_len=max_len,
            num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)),
            seed=seed,
        )
        train_criterion_fn = build_rq1_train_criterion_fn(model_name, train_kwargs)
    else:
        train_loader = build_train_loader(model_name, data_dir, stats, train_kwargs, model_kwargs=model_kwargs)
        train_criterion_fn = build_criterion_fn(model_name, train_kwargs)

    val_criterion_fn = build_criterion_fn(model_name, train_kwargs)

    val_loader   = get_eval_loader(data_dir / "splits" / "val_sequences.csv",  stats, batch_size=batch_size, max_len=max_len)
    test_loader  = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)

    eval_fn         = build_eval_fn(model_name)
    val_loss_loader = get_val_loss_loader(
        model_name,
        data_dir / "splits" / "val_sequences.csv",
        stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)),
        seed=seed,
    )

    optimizer = build_optimizer(model_name, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))

    trainer   = Trainer(
        model_name, device, output_dir,
        run_dir=run_dir,
        use_mlflow=True,
        mlflow_config={
            "experiment_name": experiment_name,
            "run_name": run_name,
            "log_artifacts": True,
            "phase": phase,
            "variant": "base",
            "reportable": True,
        },
    )

    mlflow_cfg = collect_common_run_metadata(
        model_name=model_name,
        seed=seed,
        phase="benchmark",
        extra_params={**model_kwargs, **train_kwargs},
    )
    mlflow_cfg["tags"] = build_training_tags(
        model_name=model_name,
        phase="benchmark",
        variant="base",
        reportable=True,
    )
    mlflow_cfg["tags"].update(
        {
            "benchmark_id": benchmark_id,
            "protocol_version": protocol_version,
            "preprocessing_version": preprocessing_version,
            "data_source": str(data_dir.resolve()),
        }
    )

    tracker   = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=train_kwargs["epochs"],
        criterion_fn=train_criterion_fn,
        val_criterion_fn=val_criterion_fn,
        eval_fn=eval_fn,
        gradient_clip=train_kwargs.get("gradient_clip", 5.0),
        val_loss_loader=val_loss_loader,
        early_stop_patience=train_kwargs.get("early_stop_patience", 0),
        early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4),
        scheduler=scheduler,
        mlflow_params=mlflow_cfg,
    )
    return tracker.summary()


# ---------------------------------------------------------------------------
# Heuristic model runner
# ---------------------------------------------------------------------------


def run_heuristic_model(
    model_name: str,
    data_dir: Path,
    stats: dict,
    output_dir: str,
    benchmark_id: str,
    protocol_version: str,
    preprocessing_version: str,
    model_kwargs: dict,
    train_kwargs: dict,
    seed: int,
) -> dict:
    seed_everything(seed)

    phase = "benchmark"
    experiment_name = get_experiment_name_for_phase(phase)

    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)
    model_output_dir = build_benchmark_run_dir(output_dir, benchmark_id, model_name, seed)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    val_loader  = get_eval_loader(data_dir / "splits" / "val_sequences.csv",  stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)

    if model_name == "popularity":
        from models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(data_dir / "splits" / "train_sequences.csv")
        val_results  = evaluate_popularity(model.item_counts, val_loader)
        test_results = evaluate_popularity(model.item_counts, test_loader)
        print_results("Popularity", test_results, phase="Test")
        model.save(build_heuristic_save_target(model_name, model_output_dir))
        summary = {
            "model_name": model_name,
            "best_epoch": 0,
            "best_val_ndcg": float(val_results.get("NDCG@10", 0.0)),
            "test_results": test_results,
        }
        save_heuristic_metrics(model_name, model_output_dir, summary)

        import mlflow

        configure_mlflow(mlflow_module=mlflow)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{benchmark_id}-{build_run_name(model_name, seed, variant='base')}"):
            mlflow.log_params(
                collect_common_run_metadata(
                    model_name=model_name,
                    seed=seed,
                    phase="benchmark",
                    extra_params={**model_kwargs, **train_kwargs},
                )
            )
            mlflow.set_tags(
                {
                    **build_training_tags(
                    model_name=model_name,
                    phase="benchmark",
                    variant="base",
                    reportable=True,
                    ),
                    "benchmark_id": benchmark_id,
                    "protocol_version": protocol_version,
                    "preprocessing_version": preprocessing_version,
                    "data_source": str(data_dir.resolve()),
                }
            )
            mlflow.log_metrics(
                {
                    "best_val_ndcg_at_10": float(val_results.get("NDCG@10", 0.0)),
                    "best_epoch": 0.0,
                    **{f"test_{sanitize_metric_name(k)}": v for k, v in test_results.items()},
                }
            )

        return summary

    if model_name == "itemcf":
        from models.itemcf import ItemCFRecommender
        model = ItemCFRecommender(top_k_sim=model_kwargs.get("top_k_sim", 20))
        model.fit(data_dir / "splits" / "train_sequences.csv", stats_path=data_dir / "reports" / "dataset_stats.json")
        val_results  = evaluate_itemcf(model.sim_matrix, model.user_history, val_loader)
        test_results = evaluate_itemcf(model.sim_matrix, model.user_history, test_loader)
        print_results("Item-CF", test_results, phase="Test")
        model.save(build_heuristic_save_target(model_name, model_output_dir))
        summary = {
            "model_name": model_name,
            "best_epoch": 0,
            "best_val_ndcg": float(val_results.get("NDCG@10", 0.0)),
            "test_results": test_results,
        }
        save_heuristic_metrics(model_name, model_output_dir, summary)

        import mlflow

        configure_mlflow(mlflow_module=mlflow)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{benchmark_id}-{build_run_name(model_name, seed, variant='base')}"):
            mlflow.log_params(
                collect_common_run_metadata(
                    model_name=model_name,
                    seed=seed,
                    phase="benchmark",
                    extra_params={**model_kwargs, **train_kwargs},
                )
            )
            mlflow.set_tags(
                {
                    **build_training_tags(
                    model_name=model_name,
                    phase="benchmark",
                    variant="base",
                    reportable=True,
                    ),
                    "benchmark_id": benchmark_id,
                    "protocol_version": protocol_version,
                    "preprocessing_version": preprocessing_version,
                    "data_source": str(data_dir.resolve()),
                }
            )
            mlflow.log_metrics(
                {
                    "best_val_ndcg_at_10": float(val_results.get("NDCG@10", 0.0)),
                    "best_epoch": 0.0,
                    **{f"test_{sanitize_metric_name(k)}": v for k, v in test_results.items()},
                }
            )

        return summary

    raise ValueError(f"Unknown heuristic model: {model_name}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def save_heuristic_metrics(model_name: str, output_dir: Path, summary: dict) -> None:
    metrics_payload = {
        "model_name": summary.get("model_name", model_name),
        "epochs": [],
        "best_epoch": summary.get("best_epoch", 0),
        "best_val_ndcg": summary.get("best_val_ndcg", 0.0),
        "test_results": summary["test_results"],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)


def build_run_record(model_name: str, seed: int, summary: dict) -> dict:
    return {
        "exp_id": f"{model_name}_seed{seed}",
        "model": model_name,
        "model_variant": "default",
        "seed": seed,
        "eval_protocol": "full_sort",
        "metrics": summary["test_results"],
        "train_summary": {
            "best_val_ndcg10": summary.get("best_val_ndcg", 0.0),
            "best_epoch": summary.get("best_epoch", 0),
        },
    }


def aggregate_records(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record["metrics"])
    return {
        model: {
            name: {
                "mean": float(np.mean([m[name] for m in ml])),
                "std": float(np.std([m[name] for m in ml], ddof=1)) if len(ml) > 1 else 0.0,
            }
            for name in ml[0].keys()
        }
        for model, ml in grouped.items()
    }



def plot_comparison(results: dict, output_dir: str) -> None:
    metrics     = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]
    model_names = list(results.keys())
    x, width    = np.arange(len(model_names)), 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(metrics):
        vals = [results[name]["test_results"].get(m, 0) for name in model_names]
        ax.bar(x + i * width, vals, width, label=m)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Results (Full-Sort)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison" / "comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.models:
        args.models = list(MODEL_CONFIGS.keys())

    data_dir   = Path(args.data_dir)
    validate_processed_layout(data_dir)
    stats      = load_stats(data_dir / "reports" / "dataset_stats.json")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comp_dir = Path(args.output_dir) / "benchmark" / args.benchmark_id
    manifest_path = comp_dir / "benchmark_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(
            f"Benchmark manifest already exists for benchmark_id={args.benchmark_id}: {manifest_path}. Use a new benchmark_id."
        )
    comp_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_benchmark_manifest(
        benchmark_id=args.benchmark_id,
        protocol_version=args.protocol_version,
        preprocessing_version=args.preprocessing_version,
        expected_models=list(args.models),
        neural_seeds=list(args.seeds),
        heuristic_seed=args.seeds[0],
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    raw_runs: list[dict] = []
    run_records: list[dict]      = []

    for name in args.models:
        cfg = MODEL_CONFIGS[name]
        for seed in get_seeds_for_model(name, args.seeds):
            print(f"\n{'=' * 60}")
            print(f"  Running: {name.upper()} | seed={seed}")
            print(f"{'=' * 60}")

            if name in NEURAL_MODELS:
                summary = run_neural_model(
                    name, data_dir, stats, device, args.output_dir, args.benchmark_id, args.protocol_version,
                    args.preprocessing_version,
                    cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), seed,
                )
            elif name in HEURISTIC_MODELS:
                summary = run_heuristic_model(
                    name, data_dir, stats, args.output_dir, args.benchmark_id, args.protocol_version,
                    args.preprocessing_version,
                    cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), seed,
                )
            else:
                print(f"[WARNING] Unknown model '{name}', skipping.", file=sys.stderr)
                continue

            raw_runs.append({"model": name, "seed": seed, "summary": summary})
            run_records.append(build_run_record(name, seed, summary))

    with open(comp_dir / "raw_runs.json", "w") as f:
        json.dump(raw_runs, f, indent=2)
    with open(comp_dir / "run_records.json", "w") as f:
        json.dump(run_records, f, indent=2)

    print(f"\nBenchmark run records saved to: {comp_dir}/run_records.json")
    print(f"Use scripts/rq1_report.py --benchmark-id {args.benchmark_id} to aggregate RQ1 results.")


if __name__ == "__main__":
    main()
