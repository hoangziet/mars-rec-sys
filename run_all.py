#!/usr/bin/env python3
"""
run_all.py
==========
Train all models and produce a comparison report.

Usage:
    uv run python run_all.py
    uv run python run_all.py sasrec gru4rec bprmf
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

from configs import MODEL_CONFIGS, DEFAULT_SEED, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from dataloader import get_train_loader, get_eval_loader, load_stats
from evaluate import evaluate_sequential, evaluate_bert4rec, evaluate_bprmf, evaluate_popularity, evaluate_itemcf, compare_models, print_results
from trainer import Trainer


def run_neural_model(model_name, data_dir, stats, device, output_dir,
                     model_kwargs, train_kwargs, seed, neg_mode="random", item_popularity=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_items = stats["n_items"]
    n_users = stats["n_users"]
    max_len = train_kwargs.get("max_len", 50)

    if model_name == "sasrec":
        from models.sasrec import SASRec
        model = SASRec(n_items=n_items, max_len=max_len, **model_kwargs)
    elif model_name == "gsasrec":
        from models.gsasrec import GSASRec
        model = GSASRec(n_items=n_items, max_len=max_len, **model_kwargs)
    elif model_name == "gru4rec":
        from models.gru4rec import GRU4Rec
        model = GRU4Rec(n_items=n_items, **model_kwargs)
    elif model_name == "bert4rec":
        from models.bert4rec import BERT4Rec
        model = BERT4Rec(n_items=n_items, max_len=max_len, **model_kwargs)
    elif model_name == "bprmf":
        from models.bprmf import BPRMF
        model = BPRMF(n_users=n_users, n_items=n_items, **model_kwargs)
    else:
        raise ValueError(f"Unknown neural model: {model_name}")

    model = model.to(device)

    use_conf = (model_name == "gsasrec")
    loader_type = "bprmf" if model_name == "bprmf" else model_name
    train_loader = get_train_loader(
        loader_type, data_dir / "train.csv", stats,
        batch_size=train_kwargs.get("batch_size", 256),
        max_len=max_len, use_confidence=use_conf
    )
    val_loader = get_eval_loader(
        data_dir / "val.csv", stats,
        batch_size=train_kwargs.get("batch_size", 256),
        max_len=max_len,
        neg_mode=neg_mode,
        item_popularity=item_popularity,
    )
    test_loader = get_eval_loader(
        data_dir / "test.csv", stats,
        batch_size=train_kwargs.get("batch_size", 256),
        max_len=max_len,
        neg_mode=neg_mode,
        item_popularity=item_popularity,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs.get("lr", 1e-3))

    if model_name in ("sasrec", "gru4rec"):
        import torch.nn as nn
        ce = nn.CrossEntropyLoss()
        def criterion_fn(m, batch, dev):
            inp = batch["input_seq"].to(dev)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(dev)
            logits = m(inp, mask=mask)
            return ce(logits, batch["target"].to(dev))
        eval_fn = lambda m, l, d: evaluate_sequential(m, l, d)

    elif model_name == "gsasrec":
        from models.gsasrec import weighted_ce_loss
        def criterion_fn(m, batch, dev):
            logits = m(batch["input_seq"].to(dev), mask=batch["mask"].to(dev))
            return weighted_ce_loss(logits, batch["target"].to(dev), batch["confidence"].to(dev))
        eval_fn = lambda m, l, d: evaluate_sequential(m, l, d)

    elif model_name == "bert4rec":
        import torch.nn.functional as F
        def criterion_fn(m, batch, dev):
            logits = m(batch["input_seq"].to(dev))
            labels = batch["labels"].to(dev)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
        eval_fn = lambda m, l, d: evaluate_bert4rec(m, l, d)

    elif model_name == "bprmf":
        from models.bprmf import bpr_loss
        reg = train_kwargs.get("reg_lambda", 1e-4)
        def criterion_fn(m, batch, dev):
            pos, neg = m(
                batch["user"].to(dev),
                batch["pos_item"].to(dev),
                batch["neg_item"].to(dev)
            )
            return bpr_loss(pos, neg, reg_lambda=reg, model=m)
        eval_fn = lambda m, l, d: evaluate_bprmf(m, l, d)

    trainer = Trainer(model_name, device, output_dir)
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

    return tracker.summary()


def run_heuristic_model(model_name, data_dir, stats, device, output_dir,
                        model_kwargs, train_kwargs, seed, neg_mode="random", item_popularity=None):
    np.random.seed(seed)

    n_items = stats["n_items"]
    n_users = stats["n_users"]
    num_neg = train_kwargs.get("num_neg", 99)
    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    val_loader = get_eval_loader(
        data_dir / "val.csv", stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=num_neg,
        neg_mode=neg_mode,
        item_popularity=item_popularity,
    )
    test_loader = get_eval_loader(
        data_dir / "test.csv", stats,
        batch_size=batch_size,
        max_len=max_len,
        num_neg=num_neg,
        neg_mode=neg_mode,
        item_popularity=item_popularity,
    )

    if model_name == "popularity":
        from models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(data_dir / "interactions.csv")

        val_results = evaluate_popularity(model.item_counts, val_loader, k_list=(10, 20))
        test_results = evaluate_popularity(model.item_counts, test_loader, k_list=(10, 20))
        print_results("Popularity", test_results, phase="Test")

        model.save(data_dir / "popularity_model.json")
        return {"test_results": test_results, "best_val_ndcg": float(val_results.get("NDCG@10", 0.0))}

    if model_name == "itemcf":
        from models.itemcf import ItemCFRecommender
        top_k = model_kwargs.get("top_k_sim", 20)
        model = ItemCFRecommender(top_k_sim=top_k)
        model.fit(data_dir / "interactions.csv", stats_path=data_dir / "dataset_stats.json")

        val_results = evaluate_itemcf(model.sim_matrix, model.user_history, val_loader, k_list=(10, 20))
        test_results = evaluate_itemcf(model.sim_matrix, model.user_history, test_loader, k_list=(10, 20))
        print_results("Item-CF", test_results, phase="Test")

        model.save(data_dir)
        return {"test_results": test_results, "best_val_ndcg": float(val_results.get("NDCG@10", 0.0))}

    raise ValueError(f"Unknown heuristic model: {model_name}")


def build_run_record(model_name, seed, neg_mode, num_neg, summary, commit_sha=None):
    model_variant = "confidence_weighted_sasrec" if model_name == "gsasrec" else "default"
    return {
        "exp_id": f"{model_name}_{neg_mode}_seed{seed}",
        "model": model_name,
        "model_variant": model_variant,
        "seed": seed,
        "eval_protocol": {
            "neg_mode": neg_mode,
            "num_neg": num_neg,
            "target_position": 0,
        },
        "metrics": summary["test_results"],
        "train_summary": {
            "best_val_ndcg10": summary.get("best_val_ndcg", 0.0),
            "best_epoch": summary.get("best_epoch", 0),
        },
        "git": {
            "commit": commit_sha or "unknown",
        },
    }


def aggregate_records(records):
    grouped = {}
    for record in records:
        key = f"{record['model']}::{record['eval_protocol']['neg_mode']}"
        grouped.setdefault(key, []).append(record["metrics"])

    aggregated = {}
    for key, metrics_list in grouped.items():
        metric_names = metrics_list[0].keys()
        aggregated[key] = {
            metric_name: {
                "mean": float(np.mean([metrics[metric_name] for metrics in metrics_list])),
                "std": float(np.std([metrics[metric_name] for metrics in metrics_list])),
            }
            for metric_name in metric_names
        }
    return aggregated


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_item_popularity(data_dir, n_items):
    interactions_path = data_dir / "interactions.csv"
    if not interactions_path.exists():
        return None
    counts = np.zeros(n_items + 1, dtype=float)
    with open(interactions_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                item_idx = int(float(parts[1]))
            except ValueError:
                continue
            if 1 <= item_idx <= n_items:
                counts[item_idx] += 1.0
    return counts


def plot_comparison(results, output_dir):
    metrics = ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(metrics):
        vals = [results[name]["test_results"].get(m, 0) for name in model_names]
        ax.bar(x + i * width, vals, width, label=m)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Results")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison" / "comparison.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--neg_mode", choices=["random", "popularity", "mixed"], default="random")
    args = parser.parse_args()

    if not args.models:
        args.models = list(MODEL_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    stats = load_stats(data_dir / "dataset_stats.json")
    item_popularity = build_item_popularity(data_dir, stats["n_items"]) if args.neg_mode in {"popularity", "mixed"} else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comp_dir = Path(args.output_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    run_records = []
    commit_sha = get_git_commit()

    for name in args.models:
        cfg = MODEL_CONFIGS[name]
        is_neural = name in ("sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf")

        if is_neural:
            summary = run_neural_model(
                name, data_dir, stats, device, args.output_dir,
                cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), args.seed,
                neg_mode=args.neg_mode,
                item_popularity=item_popularity,
            )
        else:
            summary = run_heuristic_model(
                name, data_dir, stats, device, args.output_dir,
                cfg["model_kwargs"].copy(), cfg["train_kwargs"].copy(), args.seed,
                neg_mode=args.neg_mode,
                item_popularity=item_popularity,
            )

        all_results[name] = summary
        run_records.append(build_run_record(
            name,
            args.seed,
            args.neg_mode,
            cfg["train_kwargs"].get("num_neg", 99),
            summary,
            commit_sha,
        ))

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    compare_models(
        {name: s["test_results"] for name, s in all_results.items()},
        k_list=(10, 20)
    )

    comp_data = {}
    for name, s in all_results.items():
        comp_data[name] = s["test_results"]
    with open(comp_dir / "comparison.json", "w") as f:
        json.dump(comp_data, f, indent=2)
    with open(comp_dir / "run_records.json", "w") as f:
        json.dump(run_records, f, indent=2)
    aggregated = aggregate_records(run_records)
    with open(comp_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    plot_comparison(all_results, args.output_dir)
    print(f"\nComparison chart saved to: {comp_dir}/comparison.png")


if __name__ == "__main__":
    main()
