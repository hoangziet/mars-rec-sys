"""
scripts/rq4_ablation.py
=======================
RQ4: Run V0-V3 ablation across 10 seeds.

Reads all config from a frozen protocol manifest.
Training parameters cannot be overridden at CLI.

Usage:
    uv run python scripts/rq4_ablation.py --protocol experiments/rq4/rq4-ablation/rq4_protocol_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.builder import build_criterion_fn, build_eval_fn, build_model, build_train_loader
from pipeline.loaders import get_eval_loader, get_val_loss_loader, load_stats
from pipeline.optim import build_optimizer, build_scheduler
from pipeline.training_grid import enforce_final_grid
from training.configs import build_model_config
from training.mlflow_contract import build_run_name, build_training_tags
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow, get_git_commit
from training.trainer import NoValidCheckpointError, Trainer
from scripts.rq4_init_protocol import verify_protocol_hashes


def _enforce_git_commit_match(protocol: dict) -> None:
    """Fail unless HEAD matches the protocol's git_commit and the tree is clean.

    Content hashes (data, config, text) protect against drifted files, but
    they don't cover model code, training logic, or evaluation code. Git is
    the only source of truth for that. Final experiments require an exact
    commit match and a clean working tree.
    """
    expected_commit = protocol.get("git_commit")
    if not expected_commit or expected_commit == "unknown":
        raise RuntimeError(
            "Protocol manifest has no git_commit — cannot verify the code "
            "used to train. Re-init the protocol with a real commit."
        )

    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()

    mismatches = []
    if actual_commit != expected_commit:
        mismatches.append(
            f"git_commit mismatch: protocol has {expected_commit[:12]}, "
            f"HEAD is {actual_commit[:12]}"
        )
    if dirty:
        mismatches.append(f"working tree is dirty ({len(dirty.splitlines())} files)")

    if mismatches:
        raise RuntimeError(
            "Git provenance check failed — the working directory differs from "
            "the frozen protocol. Commit, stash, or checkout the protocol's "
            "commit before running final experiments.\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )

EXPERIMENT_NAME = "mars_final_ablation"

METADATA_FLAGS = {
    "M0": (False, False),
    "M1": (True, False),
    "M2": (False, True),
    "M3": (True, True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: final ablation runner.")
    parser.add_argument("--protocol", required=True, help="Path to rq4_protocol_manifest.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _get_variant_config(variant: str, best_alpha: float, best_variant: str) -> dict:
    use_structured, use_text = METADATA_FLAGS[best_variant]
    if variant == "V0":
        return {"config_name": "gsasrec", "use_structured": False, "use_text": False, "confidence_alpha": 0.0}
    elif variant == "V1":
        return {"config_name": "gsasrec", "use_structured": False, "use_text": False, "confidence_alpha": best_alpha}
    elif variant == "V2":
        return {"config_name": "gsasrec_metadata", "use_structured": use_structured, "use_text": use_text, "confidence_alpha": 0.0}
    elif variant == "V3":
        return {"config_name": "gsasrec_metadata", "use_structured": use_structured, "use_text": use_text, "confidence_alpha": best_alpha}
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _run_single(args, variant: str, seed: int, variant_cfg: dict, benchmark_id: str, protocol: dict) -> dict:
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
    train_kwargs = enforce_final_grid(base_cfg["train_kwargs"])
    train_kwargs["confidence_alpha"] = variant_cfg["confidence_alpha"]

    if variant_cfg["config_name"] == "gsasrec_metadata":
        encoder_cfg = model_kwargs.get("item_encoder", {})
        encoder_cfg["use_structured"] = variant_cfg["use_structured"]
        encoder_cfg["use_text"] = variant_cfg["use_text"]
        encoder_cfg["metadata_vocab_path"] = str(data_dir / "item_features" / "metadata_vocab.json")
        encoder_cfg["metadata_csv_path"] = str(data_dir / "item_features" / "item_metadata.csv")
        encoder_cfg["text_emb_path"] = str(data_dir / "item_features" / "text_embeddings.pt")
        model_kwargs["item_encoder"] = encoder_cfg
    else:
        model_kwargs.pop("item_encoder", None)

    max_len = train_kwargs.get("max_len", 50)
    batch_size = train_kwargs.get("batch_size", 256)

    model = build_model("gsasrec", stats["n_items"], stats["n_users"], model_kwargs, max_len, data_dir=data_dir).to(device)
    train_loader = build_train_loader("gsasrec", data_dir, stats, train_kwargs, model_kwargs=model_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer("gsasrec", model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn("gsasrec", train_kwargs)
    eval_fn = build_eval_fn("gsasrec")
    val_loss_loader = get_val_loss_loader("gsasrec", data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)), seed=seed)

    run_name = build_run_name("gsasrec", seed, variant=variant.lower())
    run_output_dir = Path(args.output_dir) / "rq4" / benchmark_id / variant / f"seed_{seed}"

    trainer = Trainer("gsasrec", device, str(run_output_dir), use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "final", "variant": variant.lower(), "git_commit": get_git_commit(), "reportable": False,
    })
    mlflow_cfg = collect_common_run_metadata(model_name="gsasrec", seed=seed, phase="final", git_commit=get_git_commit(), extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name="gsasrec", phase="final", variant=variant.lower(), git_commit=get_git_commit(), reportable=False)
    mlflow_cfg["tags"]["ablation_variant"] = variant
    mlflow_cfg["tags"]["rq"] = "rq4"
    mlflow_cfg["tags"]["benchmark_id"] = benchmark_id
    mlflow_cfg["tags"]["confidence_alpha"] = str(variant_cfg["confidence_alpha"])
    mlflow_cfg["tags"]["use_structured"] = str(variant_cfg["use_structured"]).lower()
    mlflow_cfg["tags"]["use_text"] = str(variant_cfg["use_text"]).lower()
    mlflow_cfg["tags"]["protocol_version"] = protocol.get("benchmark_id", "unknown")
    mlflow_cfg["tags"]["protocol_sha256"] = protocol.get("protocol_sha256", "unknown")
    mlflow_cfg["tags"]["dataset_manifest_sha256"] = protocol.get("dataset_manifest_sha256", "unknown")
    mlflow_cfg["tags"]["config_sha256"] = protocol.get("config_sha256", "unknown")
    if protocol.get("text_artifact_sha256"):
        mlflow_cfg["tags"]["text_artifact_sha256"] = protocol["text_artifact_sha256"]
    # per_user_complete starts false — only set to true after export succeeds
    mlflow_cfg["tags"]["per_user_complete"] = "false"

    try:
        result = trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)
    except NoValidCheckpointError:
        print(f"  {variant} seed={seed}: skipping per-user export (no valid checkpoint)")
        return None
    except Exception:
        if trainer._use_mlflow and trainer._mlflow_run:
            trainer._mlflow.end_run(status="FAILED")
        raise

    # Training succeeded — now export per-user results.  Only after
    # per-user export succeeds do we mark the run as reportable.
    run_id = trainer._mlflow_run.info.run_id if trainer._mlflow_run else None
    try:
        _export_per_user(model, test_loader, device, variant, seed, args.output_dir, benchmark_id, run_id)
    except Exception:
        if trainer._use_mlflow and run_id:
            mlflow.tracking.MlflowClient().set_terminated(run_id, status="FAILED")
        raise

    # Per-user export succeeded — mark the run as complete and reportable.
    if run_id:
        client = mlflow.tracking.MlflowClient()
        client.set_tag(run_id, "per_user_complete", "true")
        client.set_tag(run_id, "reportable", "true")

    return result


def _export_per_user(model, test_loader, device, variant, seed, output_dir, benchmark_id, run_id=None):
    from pipeline.metrics import evaluate_sequential_detailed
    import csv as csv_mod
    import mlflow as mlflow_mod

    _, per_user = evaluate_sequential_detailed(model, test_loader, device)
    for row in per_user:
        row["variant"] = variant
        row["seed"] = seed

    user_dir = Path(output_dir) / "rq4" / benchmark_id / "per_user"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{variant}_s{seed}.csv"
    fields = ["variant", "seed", "user_idx", "target_item", "rank", "hit_at_10", "ndcg_at_10", "hit_at_20", "ndcg_at_20"]
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_user)

    if run_id:
        client = mlflow_mod.tracking.MlflowClient()
        client.log_artifact(run_id, str(path), artifact_path="per_user")

    return path


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    protocol = json.loads(Path(args.protocol).read_text())
    best_alpha = float(protocol["best_alpha"])
    best_variant = protocol["best_metadata_variant"]
    seeds = [int(s) for s in protocol["neural_seeds"]]
    variants_list = protocol["variants"]
    benchmark_id = protocol["benchmark_id"]

    # Provenance check: recompute SHA256 of the preprocessing report, config
    # bundle, and text embedding artifact. Refuse to train if any of them
    # drifted since the protocol was frozen. The git_commit field is kept
    # for traceability but is no longer enforced here (it is replaced by
    # the more meaningful content hashes below).
    verify_protocol_hashes(
        manifest=protocol,
        data_dir=Path(args.data_dir),
        configs_glob="configs/model/*.yaml",
    )

    _enforce_git_commit_match(protocol)

    total = len(variants_list) * len(seeds)
    print(f"RQ4 ablation: {len(variants_list)} variants x {len(seeds)} seeds = {total} runs")
    print(f"Protocol: {args.protocol}")
    print(f"Best alpha: {best_alpha}")
    print(f"Best metadata variant: {best_variant}")
    print(f"Benchmark ID: {benchmark_id}")

    failed_runs: list[tuple[str, int]] = []
    for i, variant in enumerate(variants_list):
        variant_cfg = _get_variant_config(variant, best_alpha, best_variant)
        for j, seed in enumerate(seeds):
            run_num = i * len(seeds) + j + 1
            print(f"\n[{run_num}/{total}] {variant}, seed={seed}")
            result = _run_single(args, variant, seed, variant_cfg, benchmark_id, protocol)
            if result is None:
                failed_runs.append((variant, seed))

    if failed_runs:
        n_failed = len(failed_runs)
        print(f"\nINCOMPLETE: {n_failed}/{total} runs failed to produce valid checkpoints.")
        print(f"  Failed: {failed_runs}")
        raise RuntimeError(
            f"RQ4 campaign incomplete: {n_failed} runs failed. "
            f"Run `rq4-collect` to verify which seeds are missing."
        )

    print(f"\nDone. {total} runs completed. Results logged to MLflow experiment '{EXPERIMENT_NAME}'.")
    print(f"Run: make rq4-collect")


if __name__ == "__main__":
    main()
