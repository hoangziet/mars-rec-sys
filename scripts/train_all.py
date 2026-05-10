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
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, MODEL_CONFIGS
from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.metrics import (
    compare_models,
    evaluate_itemcf,
    evaluate_popularity,
    print_results,
)
from trainer import Trainer


# ---------------------------------------------------------------------------
# Neural model runner
# ---------------------------------------------------------------------------


def run_neural_model(
    model_name: str,
    data_dir: Path,
    stats: dict,
    device: torch.device,
    output_dir: str,
    model_kwargs: dict,
    train_kwargs: dict,
    seed: int,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model        = build_model(model_name, stats["n_items"], stats["n_users"], model_kwargs, max_len).to(device)
    train_loader = build_train_loader(model_name, data_dir, stats, train_kwargs)
    val_loader   = get_eval_loader(data_dir / "val.csv",  stats, batch_size=batch_size, max_len=max_len)
    test_loader  = get_eval_loader(data_dir / "test.csv", stats, batch_size=batch_size, max_len=max_len)

    criterion_fn    = build_criterion_fn(model_name, train_kwargs)
    eval_fn         = build_eval_fn(model_name)
    val_loss_loader = get_val_loss_loader(
        model_name,
        data_dir / "val.csv",
        stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=train_kwargs.get("num_neg", 1),
        seed=seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_kwargs.get("lr", 1e-3),
        betas=(0.9, train_kwargs.get("beta2", 0.999)),
        weight_decay=train_kwargs.get("weight_decay", 0.0),
    )

    warmup_steps = train_kwargs.get("warmup_steps", 0)
    scheduler = None
    if warmup_steps > 0:
        total_steps = train_kwargs["epochs"] * len(train_loader)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - progress)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer   = Trainer(model_name, device, output_dir)
    tracker   = trainer.train(
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
    return tracker.summary()


# ---------------------------------------------------------------------------
# Heuristic model runner
# ---------------------------------------------------------------------------


def run_heuristic_model(
    model_name: str,
    data_dir: Path,
    stats: dict,
    output_dir: str,
    model_kwargs: dict,
    train_kwargs: dict,
    seed: int,
) -> dict:
    np.random.seed(seed)

    max_len    = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    val_loader  = get_eval_loader(data_dir / "val.csv",  stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "test.csv", stats, batch_size=batch_size, max_len=max_len)

    if model_name == "popularity":
        from models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(data_dir / "interactions.csv")
        val_results  = evaluate_popularity(model.item_counts, val_loader)
        test_results = evaluate_popularity(model.item_counts, test_loader)
        print_results("Popularity", test_results, phase="Test")
        model.save(data_dir / "popularity_model.json")
        return {"test_results": test_results, "best_val_ndcg": float(val_results.get("NDCG@10", 0.0))}

    if model_name == "itemcf":
        from models.itemcf import ItemCFRecommender
        model = ItemCFRecommender(top_k_sim=model_kwargs.get("top_k_sim", 20))
        model.fit(data_dir / "interactions.csv", stats_path=data_dir / "dataset_stats.json")
        val_results  = evaluate_itemcf(model.sim_matrix, model.user_history, val_loader)
        test_results = evaluate_itemcf(model.sim_matrix, model.user_history, test_loader)
        print_results("Item-CF", test_results, phase="Test")
        model.save(data_dir)
        return {"test_results": test_results, "best_val_ndcg": float(val_results.get("NDCG@10", 0.0))}

    raise ValueError(f"Unknown heuristic model: {model_name}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def build_run_record(model_name: str, seed: int, summary: dict, commit_sha: str | None = None) -> dict:
    return {
        "exp_id": f"{model_name}_seed{seed}",
        "model": model_name,
        "model_variant": "confidence_weighted_sasrec" if model_name == "gsasrec" else "default",
        "seed": seed,
        "eval_protocol": "full_sort",
        "metrics": summary["test_results"],
        "train_summary": {
            "best_val_ndcg10": summary.get("best_val_ndcg", 0.0),
            "best_epoch": summary.get("best_epoch", 0),
        },
        "git": {"commit": commit_sha or "unknown"},
    }


def aggregate_records(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record["metrics"])
    return {
        model: {
            name: {"mean": float(np.mean([m[name] for m in ml])), "std": float(np.std([m[name] for m in ml]))}
            for name in ml[0].keys()
        }
        for model, ml in grouped.items()
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def plot_comparison(results: dict, output_dir: str) -> None:
    metrics     = ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]
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
    parser = argparse.ArgumentParser(description="Train all models and compare results.")
    parser.add_argument("models", nargs="*", default=None, choices=list(MODEL_CONFIGS.keys()),
                        help="Subset of models to run (default: all).")
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.models:
        args.models = list(MODEL_CONFIGS.keys())

    data_dir   = Path(args.data_dir)
    stats      = load_stats(data_dir / "dataset_stats.json")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit_sha = get_git_commit()

    comp_dir = Path(args.output_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    NEURAL_MODELS    = {"sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"}
    HEURISTIC_MODELS = {"popularity", "itemcf"}

    all_results: dict[str, dict] = {}
    run_records: list[dict]      = []

    for name in args.models:
        cfg = MODEL_CONFIGS[name]
        print(f"\n{'=' * 60}")
        print(f"  Running: {name.upper()}")
        print(f"{'=' * 60}")

        if name in NEURAL_MODELS:
            summary = run_neural_model(
                name, data_dir, stats, device, args.output_dir,
                cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), args.seed,
            )
        elif name in HEURISTIC_MODELS:
            summary = run_heuristic_model(
                name, data_dir, stats, args.output_dir,
                cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), args.seed,
            )
        else:
            print(f"[WARNING] Unknown model '{name}', skipping.", file=sys.stderr)
            continue

        all_results[name] = summary
        run_records.append(build_run_record(name, args.seed, summary, commit_sha))

    print(f"\n{'=' * 60}\n  COMPARISON\n{'=' * 60}")
    compare_models({name: s["test_results"] for name, s in all_results.items()})

    with open(comp_dir / "comparison.json",      "w") as f: json.dump({name: s["test_results"] for name, s in all_results.items()}, f, indent=2)
    with open(comp_dir / "run_records.json",     "w") as f: json.dump(run_records, f, indent=2)
    with open(comp_dir / "aggregate_metrics.json","w") as f: json.dump(aggregate_records(run_records), f, indent=2)

    plot_comparison(all_results, args.output_dir)
    print(f"\nComparison chart saved to: {comp_dir}/comparison.png")


if __name__ == "__main__":
    main()
