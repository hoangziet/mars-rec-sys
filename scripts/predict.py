"""
scripts/predict.py
==================
Load a trained model checkpoint and generate top-K recommendations for a user.

Usage:
    uv run python scripts/predict.py sasrec --user_id 42 --top_k 10
    uv run python scripts/predict.py bprmf  --user_id 42 --top_k 20
    uv run python scripts/predict.py sasrec --user_id 42 --show_titles

The user's training history is read from
    <data_dir>/splits/train_sequences.csv
Item titles (if available) are read from
    <data_dir>/item_features/item_metadata.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.configs import DEFAULT_DATA_DIR, build_model_config
from pipeline.builder import build_model
from pipeline.loaders import load_stats, parse_seq, pad_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_checkpoint(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_user_history(train_csv: Path, user_id: int) -> list[int]:
    import pandas as pd
    df = pd.read_csv(train_csv)
    row = df[df["user_idx"] == user_id]
    if row.empty:
        return []
    return parse_seq(row.iloc[0]["item_sequence"])


def load_item_titles(meta_csv: Path) -> dict[int, str]:
    import pandas as pd
    if not meta_csv.exists():
        return {}
    df = pd.read_csv(meta_csv)
    id_col    = "item_idx" if "item_idx" in df.columns else df.columns[0]
    name_col  = next((c for c in df.columns if "name" in c.lower() or "title" in c.lower()), None)
    if name_col is None:
        return {}
    return dict(zip(df[id_col].tolist(), df[name_col].tolist()))


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_sequential(model, history: list[int], n_items: int, max_len: int,
                        top_k: int, device: torch.device) -> list[int]:
    seq    = torch.tensor([pad_sequence(history, max_len)], dtype=torch.long, device=device)
    logits = model(seq)[0]               # (n_items+1,)
    logits[0] = float("-inf")            # mask padding
    for item in history:
        logits[item] = float("-inf")     # mask seen items
    _, indices = logits.topk(top_k)
    return indices.cpu().tolist()


@torch.no_grad()
def predict_bert4rec(model, history: list[int], n_items: int, max_len: int,
                     top_k: int, device: torch.device) -> list[int]:
    seq     = pad_sequence(history + [model.mask_token], max_len)
    seq_t   = torch.tensor([seq], dtype=torch.long, device=device)
    logits  = model(seq_t)[0, -1, :]    # (vocab,)
    logits[0] = float("-inf")
    for item in history:
        logits[item] = float("-inf")
    _, indices = logits.topk(top_k)
    return indices.cpu().tolist()


@torch.no_grad()
def predict_bprmf(model, user_id: int, history: list[int], n_items: int,
                  top_k: int, device: torch.device) -> list[int]:
    all_item_ids = torch.arange(n_items + 1, device=device)
    all_item_emb = model.item_embedding(all_item_ids)                  # (n+1, D)
    user_emb     = model.user_embedding(torch.tensor([user_id], device=device))  # (1, D)
    scores       = (user_emb @ all_item_emb.T)[0]                      # (n+1,)
    scores[0]    = float("-inf")
    for item in history:
        scores[item] = float("-inf")
    _, indices = scores.topk(top_k)
    return indices.cpu().tolist()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate top-K recommendations for a user.")
    parser.add_argument("model",       choices=["sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"])
    parser.add_argument("--user_id",   type=int, required=True, help="Remapped user index (1-based)")
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--data_dir",  default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt_dir",  default="experiments")
    parser.add_argument("--show_titles", action="store_true", help="Show item titles if available")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stats    = load_stats(data_dir / "reports" / "dataset_stats.json")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg          = build_model_config(args.model)
    model_kwargs = cfg["model_kwargs"].copy()
    train_kwargs = cfg["train_kwargs"].copy()
    max_len      = train_kwargs.get("max_len", 50)

    model = build_model(
        args.model, stats["n_items"], stats["n_users"], model_kwargs, max_len
    ).to(device)

    ckpt_path = Path(args.ckpt_dir) / args.model / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model = load_checkpoint(model, ckpt_path, device)

    history = load_user_history(data_dir / "splits" / "train_sequences.csv", args.user_id)
    if not history:
        print(f"User {args.user_id} not found in training data.", file=sys.stderr)
        sys.exit(1)

    print(f"\nUser {args.user_id} | history length: {len(history)}")
    print(f"Last 5 items in history: {history[-5:]}")
    print(f"\nTop-{args.top_k} recommendations ({args.model}):")

    if args.model in ("sasrec", "gsasrec", "gru4rec"):
        recs = predict_sequential(model, history, stats["n_items"], max_len, args.top_k, device)
    elif args.model == "bert4rec":
        recs = predict_bert4rec(model, history, stats["n_items"], max_len, args.top_k, device)
    elif args.model == "bprmf":
        recs = predict_bprmf(model, args.user_id, history, stats["n_items"], args.top_k, device)
    else:
        print(f"Model {args.model} not supported for inference.", file=sys.stderr)
        sys.exit(1)

    titles = load_item_titles(data_dir / "item_features" / "item_metadata.csv") if args.show_titles else {}

    for rank, item_id in enumerate(recs, 1):
        title = f"  [{titles[item_id]}]" if item_id in titles else ""
        print(f"  {rank:2d}. item_idx={item_id}{title}")


if __name__ == "__main__":
    main()
