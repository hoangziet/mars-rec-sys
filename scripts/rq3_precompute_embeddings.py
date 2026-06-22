"""
scripts/rq3_precompute_embeddings.py
=====================================
Precompute text embeddings for all items using a sentence transformer.

Input:  data/processed/item_features/item_metadata.csv (title, description)
Model:  dangvantuan/sentence-camembert-base (frozen)
Output: data/processed/item_features/text_embeddings.pt
        Shape: (n_items+1, 768) — index 0 = zero vector (padding)

Usage:
    uv run python scripts/rq3_precompute_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = "dangvantuan/sentence-camembert-base"
DEFAULT_INPUT = "data/processed/item_features/item_metadata.csv"
DEFAULT_OUTPUT = "data/processed/item_features/text_embeddings.pt"
MISSING_TEXT = "contenu non disponible"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute text embeddings for items.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_name, device="cpu")

    df = pd.read_csv(args.input)
    df = df.sort_values("item_idx").reset_index(drop=True)

    if df["item_idx"].duplicated().any():
        dupes = df[df["item_idx"].duplicated()]["item_idx"].tolist()
        raise ValueError(f"Duplicate item_idx: {dupes[:10]}")

    n_items = len(df)
    expected_idx = set(range(1, n_items + 1))
    actual_idx = set(df["item_idx"].astype(int))
    if actual_idx != expected_idx:
        missing = sorted(expected_idx - actual_idx)
        raise ValueError(f"item_idx gap. Missing={missing[:10]}")

    texts = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("title")):
            parts.append(str(row["title"]))
        if pd.notna(row.get("description")):
            parts.append(str(row["description"]))
        text = " ".join(parts) if parts else MISSING_TEXT
        texts.append(text)

    emb_dim = model.get_sentence_embedding_dimension()

    print(f"Encoding {n_items} items (dim={emb_dim})...")
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=False)

    full_embeddings = torch.zeros(n_items + 1, emb_dim)
    item_indices = df["item_idx"].astype(int).tolist()
    for i, idx in enumerate(item_indices):
        full_embeddings[idx] = torch.tensor(embeddings[i], dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_embeddings, output_path)

    import hashlib
    map_path = output_path.parent.parent / "mappings" / "item_id_map.csv"
    meta_csv_path = output_path.parent / "item_metadata.csv"
    item_id_map_sha256 = hashlib.sha256(map_path.read_bytes()).hexdigest() if map_path.exists() else None
    metadata_sha256 = hashlib.sha256(meta_csv_path.read_bytes()).hexdigest() if meta_csv_path.exists() else None

    manifest_path = output_path.with_name("text_embeddings_manifest.json")
    manifest_path.write_text(json.dumps({
        "n_items": n_items,
        "embedding_dim": emb_dim,
        "padding_row": 0,
        "item_id_map_sha256": item_id_map_sha256,
        "metadata_sha256": metadata_sha256,
        "encoder": args.model_name,
        "encoder_revision": "main",
        "text_template_version": "v1",
        "shape": list(full_embeddings.shape),
    }, indent=2))

    print(f"Saved: {output_path}  shape={full_embeddings.shape}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
