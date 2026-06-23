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
from scripts.rq4_per_user import (
    PerUserExportError,
    fail_run_atomically,
    promote_run_complete,
    validate_per_user_file,
    write_per_user_atomic,
)
from training.configs import build_model_config
from training.mlflow_contract import build_run_name, build_training_tags
from training.mlflow_utils import collect_common_run_metadata, configure_mlflow, get_git_commit
from training.trainer import NoValidCheckpointError, Trainer


def _validate_protocol_backbone(protocol: dict) -> str:
    """RQ4 is gSASRec-only — refuse any protocol whose backbone is not gsasrec.

    The protocol is frozen by ``rq4-init`` from the RQ2/RQ3 winners, and the
    init step already enforces backbone="gsasrec". This guard is the last
    line of defense: if someone hand-edits the protocol manifest or runs
    rq4-ablation against a stale file, we fail loud here.
    """
    backbone = protocol.get("backbone")
    if backbone != "gsasrec":
        raise RuntimeError(
            f"RQ4 is gSASRec-only, but protocol backbone is {backbone!r}. "
            f"Re-run rq4-init from gSASRec RQ2/RQ3 winners."
        )
    return backbone


def _enforce_git_commit_match(protocol: dict) -> None:
    """Deprecated no-op stub.

    Git-commit runtime gating was removed from the RQ4 research contract.
    Provenance is now lightweight: ``preprocessing_version`` and
    ``data_source`` only. RQ4 no longer blocks runs on HEAD mismatch or
    working-tree dirtiness. This symbol is kept only to avoid breaking
    external imports; new code should not call it.
    """
    return None


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


def _run_single(args, backbone: str, variant: str, seed: int, variant_cfg: dict, benchmark_id: str, protocol: dict) -> dict:
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

    model = build_model(backbone, stats["n_items"], stats["n_users"], model_kwargs, max_len, data_dir=data_dir).to(device)
    train_loader = build_train_loader(backbone, data_dir, stats, train_kwargs, model_kwargs=model_kwargs)
    val_loader = get_eval_loader(data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    test_loader = get_eval_loader(data_dir / "splits" / "test_sequences.csv", stats, batch_size=batch_size, max_len=max_len)
    optimizer = build_optimizer(backbone, model, train_kwargs)
    scheduler = build_scheduler(optimizer, train_kwargs, len(train_loader))
    criterion_fn = build_criterion_fn(backbone, train_kwargs)
    eval_fn = build_eval_fn(backbone)
    val_loss_loader = get_val_loss_loader(backbone, data_dir / "splits" / "val_sequences.csv", stats, batch_size=batch_size, max_len=max_len, num_neg=model_kwargs.get("num_neg", train_kwargs.get("num_neg", 1)), seed=seed)

    run_name = build_run_name(backbone, seed, variant=variant.lower())
    run_output_dir = Path(args.output_dir) / "rq4" / benchmark_id / variant / f"seed_{seed}"

    trainer = Trainer(backbone, device, str(run_output_dir), use_mlflow=True, mlflow_config={
        "experiment_name": EXPERIMENT_NAME, "run_name": run_name, "log_artifacts": True,
        "phase": "final", "variant": variant.lower(), "git_commit": get_git_commit(), "reportable": False,
    })
    mlflow_cfg = collect_common_run_metadata(model_name=backbone, seed=seed, phase="final", git_commit=get_git_commit(), extra_params={**model_kwargs, **train_kwargs})
    mlflow_cfg["tags"] = build_training_tags(model_name=backbone, phase="final", variant=variant.lower(), git_commit=get_git_commit(), reportable=False)
    mlflow_cfg["tags"]["ablation_variant"] = variant
    mlflow_cfg["tags"]["rq"] = "rq4"
    mlflow_cfg["tags"]["benchmark_id"] = benchmark_id
    mlflow_cfg["tags"]["backbone"] = backbone
    mlflow_cfg["tags"]["confidence_alpha"] = str(variant_cfg["confidence_alpha"])
    mlflow_cfg["tags"]["use_structured"] = str(variant_cfg["use_structured"]).lower()
    mlflow_cfg["tags"]["use_text"] = str(variant_cfg["use_text"]).lower()
    mlflow_cfg["tags"]["protocol_version"] = protocol.get("benchmark_id", "unknown")
    # Lightweight provenance (no SHA256 hashing in the RQ4 contract).
    mlflow_cfg["tags"]["preprocessing_version"] = protocol.get("preprocessing_version", "unknown")
    mlflow_cfg["tags"]["data_source"] = protocol.get("data_source", "unknown")
    # per_user_complete starts false — only set to true after export succeeds
    mlflow_cfg["tags"]["per_user_complete"] = "false"

    try:
        result = trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, epochs=train_kwargs["epochs"], criterion_fn=criterion_fn, eval_fn=eval_fn, gradient_clip=train_kwargs.get("gradient_clip", 5.0), val_loss_loader=val_loss_loader, early_stop_patience=train_kwargs.get("early_stop_patience", 10), early_stop_min_delta=train_kwargs.get("early_stop_min_delta", 1e-4), scheduler=scheduler, mlflow_params=mlflow_cfg)
    except NoValidCheckpointError:
        print(f"  {variant} seed={seed}: skipping per-user export (no valid checkpoint)")
        return None

    # Training succeeded — now export per-user results through the atomic helper.
    # Only after the helper fully succeeds do we mark the run as reportable.
    # Use Trainer.last_run_id (not _mlflow_run) — the trainer closes its
    # active run in finally, so _mlflow_run is None here even on success.
    run_id = trainer.last_run_id
    if not run_id:
        raise RuntimeError(
            "RQ4 training completed but Trainer.last_run_id is missing; "
            "cannot export/promote per-user artifacts."
        )
    try:
        _export_per_user(model, test_loader, device, variant, seed, args.output_dir, benchmark_id, run_id)
    except Exception as exc:
        if run_id:
            fail_run_atomically(run_id, exc)
        # Re-raise the original exception so the caller knows the run failed.
        raise

    # Per-user export succeeded — promote tags in one shot. If anything
    # goes wrong here we fail the run so the collector won't pick it up
    # as a stale partial result.
    if run_id:
        try:
            promote_run_complete(run_id)
        except Exception as exc:
            fail_run_atomically(run_id, exc)
            raise

    return result


def _export_per_user(model, test_loader, device, variant, seed, output_dir, benchmark_id, run_id=None):
    """Atomic per-user export:

        1. Evaluate on test loader.
        2. Write CSV atomically (write tmp, validate, rename).
        3. Validate the canonical file.
        4. Upload to MLflow if ``run_id`` is given.

    Any failure raises and leaves the file system in a state with no
    partial canonical artifact at the target path. Caller is responsible
    for marking the run FAILED on failure.
    """
    from pipeline.metrics import evaluate_sequential_detailed

    _, per_user = evaluate_sequential_detailed(model, test_loader, device)
    for row in per_user:
        row["variant"] = variant
        row["seed"] = seed

    user_dir = Path(output_dir) / "rq4" / benchmark_id / "per_user"
    canonical_path = user_dir / f"{variant}_s{seed}.csv"

    write_per_user_atomic(per_user, canonical_path)
    # Sanity re-read after rename.
    validate_per_user_file(canonical_path, expected_min_rows=1)

    if run_id:
        import mlflow as mlflow_mod
        try:
            mlflow_mod.tracking.MlflowClient().log_artifact(
                run_id, str(canonical_path), artifact_path="per_user"
            )
        except Exception as exc:
            # Roll back the on-disk artifact if upload fails so the
            # collector cannot read a stale file the run never reported.
            try:
                if canonical_path.exists():
                    canonical_path.unlink()
            except OSError:
                pass
            raise PerUserExportError(
                f"Failed to upload per-user artifact to MLflow run {run_id}: {exc}"
            ) from exc

    return canonical_path


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    protocol = json.loads(Path(args.protocol).read_text())
    best_alpha = float(protocol["best_alpha"])
    best_variant = protocol["best_metadata_variant"]
    seeds = [int(s) for s in protocol["neural_seeds"]]
    variants_list = protocol["variants"]
    benchmark_id = protocol["benchmark_id"]

    # The backbone MUST come from the protocol manifest (frozen by
    # rq4-init from the gSASRec RQ2/RQ3 winners). We enforce gsasrec-only
    # here so a stale or hand-edited protocol cannot drive a non-gsasrec
    # run. The RQ4 contract uses light provenance (preprocessing_version
    # + data_source); git_commit is recorded as a tag for traceability
    # only.
    backbone = _validate_protocol_backbone(protocol)

    total = len(variants_list) * len(seeds)
    print(f"RQ4 ablation: backbone={backbone}, {len(variants_list)} variants x {len(seeds)} seeds = {total} runs")
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
            result = _run_single(args, backbone, variant, seed, variant_cfg, benchmark_id, protocol)
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
