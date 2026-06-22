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

import hashlib
import json
from pathlib import Path

import torch

from pipeline.loaders import get_train_loader
from pipeline.metrics import (
    evaluate_bert4rec,
    evaluate_bprmf,
    evaluate_sequential,
)


# ---------------------------------------------------------------------------
# Text embedding manifest validation
# ---------------------------------------------------------------------------


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_text_embedding_manifest(
    embeddings_path,
    manifest: dict,
    item_id_map_path,
    metadata_csv_path,
    n_items: int,
) -> None:
    """Fail-fast validation of precomputed text embeddings against manifest.

    Checks:
        - Embeddings file exists and is loadable
        - Shape matches n_items+1 x embedding_dim
        - Row 0 (padding) is exactly zero
        - No NaN or Inf values
        - item_id_map_sha256 matches current item_id_map.csv
        - metadata_sha256 matches current item_metadata.csv
    """
    embeddings_path = Path(embeddings_path)
    item_id_map_path = Path(item_id_map_path)
    metadata_csv_path = Path(metadata_csv_path)

    if not embeddings_path.exists():
        raise RuntimeError(f"Text embeddings file not found: {embeddings_path}")
    if not item_id_map_path.exists():
        raise RuntimeError(f"item_id_map.csv not found: {item_id_map_path}")
    if not metadata_csv_path.exists():
        raise RuntimeError(f"item_metadata.csv not found: {metadata_csv_path}")

    emb = torch.load(embeddings_path, weights_only=True)
    expected_shape = (manifest["n_items"] + 1, manifest["embedding_dim"])
    if tuple(emb.shape) != expected_shape:
        raise RuntimeError(
            f"Embeddings shape mismatch: expected {expected_shape}, got {tuple(emb.shape)}"
        )
    if emb.shape[0] - 1 != n_items:
        raise RuntimeError(
            f"Embeddings n_items mismatch: manifest says {manifest['n_items']}, "
            f"caller says {n_items}"
        )

    if not torch.all(emb[0] == 0).item():
        raise RuntimeError(
            f"Embeddings padding row 0 is not zero. This will leak signal into "
            f"padding positions. Recompute with rq3_precompute_embeddings.py."
        )

    if torch.isnan(emb).any().item() or torch.isinf(emb).any().item():
        raise RuntimeError("Embeddings contain NaN or Inf values. Recompute.")

    actual_map_sha = _sha256_file(item_id_map_path)
    expected_map_sha = manifest.get("item_id_map_sha256")
    if expected_map_sha and actual_map_sha != expected_map_sha:
        raise RuntimeError(
            f"item_id_map_sha256 mismatch: manifest has {expected_map_sha[:12]}..., "
            f"current file has {actual_map_sha[:12]}... "
            f"This usually means item indices have changed since embeddings were computed. "
            f"Recompute with rq3_precompute_embeddings.py."
        )

    actual_meta_sha = _sha256_file(metadata_csv_path)
    expected_meta_sha = manifest.get("metadata_sha256")
    if expected_meta_sha and actual_meta_sha != expected_meta_sha:
        raise RuntimeError(
            f"metadata_sha256 mismatch: manifest has {expected_meta_sha[:12]}..., "
            f"current file has {actual_meta_sha[:12]}... "
            f"Recompute with rq3_precompute_embeddings.py."
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

        item_encoder = None
        encoder_cfg = model_kwargs.pop("item_encoder", None)
        if encoder_cfg is not None:
            from pipeline.item_encoder import ItemEncoder
            from pipeline.metadata_utils import MetadataVocab, build_metadata_tensors, load_item_metadata

            vocab = MetadataVocab.load(encoder_cfg["metadata_vocab_path"])
            meta_df = load_item_metadata(
                encoder_cfg.get("metadata_csv_path", "data/processed/item_features/item_metadata.csv"),
                n_items,
            )
            meta_tensors = build_metadata_tensors(vocab, meta_df, n_items)

            text_emb = None
            if encoder_cfg.get("use_text", False):
                text_emb_path = Path(encoder_cfg["text_emb_path"])
                manifest_path = text_emb_path.with_name("text_embeddings_manifest.json")
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    _validate_text_embedding_manifest(
                        embeddings_path=text_emb_path,
                        manifest=manifest,
                        item_id_map_path=Path("data/processed/mappings/item_id_map.csv"),
                        metadata_csv_path=Path(encoder_cfg.get("metadata_csv_path", "data/processed/item_features/item_metadata.csv")),
                        n_items=n_items,
                    )
                else:
                    raise RuntimeError(
                        f"Text embeddings manifest not found: {manifest_path}. "
                        f"Re-run `make rq3-precompute` to generate it."
                    )
                text_emb = torch.load(text_emb_path, weights_only=True)

            item_encoder = ItemEncoder(
                n_items=n_items,
                hidden_dim=model_kwargs.get("hidden_dim", 64),
                metadata_tensors=meta_tensors,
                text_embeddings=text_emb,
                use_structured=encoder_cfg.get("use_structured", True),
                use_text=encoder_cfg.get("use_text", True),
            )

        return GSASRec(n_items=n_items, max_len=max_len, item_encoder=item_encoder, **model_kwargs)

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
        alpha = train_kwargs.get("confidence_alpha", 0.0)

        def fn(model, batch, device):
            return model.loss(
                batch["input_seq"].to(device),
                batch["pos_items"].to(device),
                batch["neg_items"].to(device),
                reduction="none" if alpha > 0 else "mean",
            )

        if alpha > 0:
            from pipeline.confidence import WeightedCriterionFn
            return WeightedCriterionFn(fn, alpha=alpha)

        return fn

    if model_name == "gru4rec":
        loss_type = train_kwargs.get("loss_type", "ce")
        def fn(model, batch, device):
            if loss_type == "ce":
                return model.loss(
                    batch["input_seq"].to(device),
                    batch["pos_items"].to(device),
                )
            else:
                if not hasattr(model, "_loss_fn"):
                    if loss_type == "top1":
                        model._loss_fn = model.top1_loss
                    elif loss_type == "bpr_max":
                        model._loss_fn = model.bpr_max_loss
                    else:
                        raise ValueError(f"Unknown GRU4Rec loss_type: {loss_type}")
                return model.loss(
                    batch["input_seq"].to(device),
                    batch["pos_items"].to(device),
                    batch["neg_items"].to(device),
                )
        return fn

    if model_name == "bert4rec":
        import torch.nn.functional as F

        def fn(model, batch, device):
            input_seq = batch["input_seq"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_seq)
            mask = (labels != 0)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            logits_masked = logits[mask]
            labels_masked = labels[mask]
            return F.cross_entropy(logits_masked, labels_masked)
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
        extra = {}
        if model_name == "bert4rec":
            extra["mask_prob"] = train_kwargs.get("mask_ratio", 0.15)
            extra["dupe_factor"] = train_kwargs.get("dupe_factor", 1)
            extra["prop_sliding_window"] = train_kwargs.get("prop_sliding_window", -1.0)
            extra["force_last_item_mask"] = train_kwargs.get("force_last_item_mask", False)
        return get_train_loader(
            model_name,
            data_dir / "splits" / "train_sequences.csv",
            stats,
            batch_size=batch_size,
            max_len=max_len,
            num_neg=num_neg,
            **extra,
        )

    if model_name == "bprmf":
        return get_train_loader(
            "bprmf",
            data_dir / "splits" / "train_sequences.csv",
            stats,
            batch_size=batch_size,
            max_len=max_len,
        )

    return None
