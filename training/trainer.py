"""
training/trainer.py
===================
Unified training framework for all recommendation models.

Classes:
    ExperimentTracker — per-epoch metrics, JSON export, plot generation
    Trainer           — unified training loop, checkpointing, evaluation
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from training.mlflow_contract import ARTIFACT_PATHS
from training.mlflow_utils import configure_mlflow, sanitize_metric_name


def _should_use_tqdm() -> bool:
    """Enable live progress bars only when stderr looks like an interactive TTY."""
    if os.environ.get("DISABLE_TQDM") == "1":
        return False
    return sys.stderr.isatty()


def _progress(iterable, desc: str):
    """Create a tqdm iterator with a stable, compact console format."""
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        disable=not _should_use_tqdm(),
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{desc:<18} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


# Experiment Tracker 
class ExperimentTracker:
    """Track per-epoch metrics, save JSON, generate plots."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.epochs: list[dict] = []
        self.best_epoch: int = -1
        self.best_val_ndcg: float = -1.0
        self.test_results: dict = {}

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float | None,
                  val_metrics: dict):
        """Record one epoch's results."""
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": None if val_loss is None else round(val_loss, 6),
        }
        entry.update({k: round(v, 6) for k, v in val_metrics.items()})
        self.epochs.append(entry)

        ndcg = val_metrics.get("NDCG@10", 0.0)
        if ndcg > self.best_val_ndcg:
            self.best_val_ndcg = ndcg
            self.best_epoch = epoch

    def finalize(self, test_results: dict):
        self.test_results = {k: round(v, 6) for k, v in test_results.items()}

    def save_metrics(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_ndcg": round(self.best_val_ndcg, 6),
            "test_results": self.test_results,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def plot_losses(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        epochs = [e["epoch"] for e in self.epochs]
        train = [e["train_loss"] for e in self.epochs]
        val = [e["val_loss"] for e in self.epochs]
        has_val = any(v is not None for v in val)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train, "b-o", label="Train Loss", markersize=4)
        if has_val:
            plt.plot(epochs, val, "r-o", label="Val Loss", markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.model_name} — Loss Curve")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_metrics(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        epochs = [e["epoch"] for e in self.epochs]
        metrics = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

        plt.figure(figsize=(8, 5))
        for m, c in zip(metrics, colors):
            vals = [e.get(m, 0) for e in self.epochs]
            plt.plot(epochs, vals, "-o", color=c, label=m, markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{self.model_name} — Evaluation Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_ndcg": round(self.best_val_ndcg, 6),
            "test_results": self.test_results,
        }


# Trainer
class Trainer:
    """Unified training loop for all model types."""

    def __init__(self, model_name: str, device: str | torch.device,
                 output_dir: str | Path = "experiments",
                 run_dir: str | Path | None = None,
                 use_mlflow: bool = False,
                 mlflow_config: dict | None = None):
        self.model_name = model_name
        self.device = torch.device(device)
        self.output_dir = Path(run_dir) if run_dir is not None else Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(model_name)

        self._use_mlflow = use_mlflow
        self._mlflow_run = None
        self._mlflow = None
        self._mlflow_log_artifacts = False
        if use_mlflow:
            import mlflow

            self._mlflow = mlflow
            mlflow_config = mlflow_config or {}
            experiment_name = mlflow_config.get("experiment_name", "mars_rec_sys")
            run_name = mlflow_config.get("run_name", model_name)
            self._mlflow_log_artifacts = mlflow_config.get("log_artifacts", True)
            self._mlflow_phase = mlflow_config.get("phase")
            self._mlflow_variant = mlflow_config.get("variant")
            self._mlflow_git_commit = mlflow_config.get("git_commit")
            self._mlflow_reportable = mlflow_config.get("reportable", True)
            configure_mlflow(mlflow_module=self._mlflow)
            self._mlflow.set_experiment(experiment_name)
            self._mlflow_run = self._mlflow.start_run(run_name=run_name)

    # Public API
    def train(self, model, train_loader, val_loader,
              optimizer, epochs: int,
              criterion_fn, eval_fn,
              gradient_clip: float = 5.0,
              val_loss_loader=None,
              test_loader=None,
              early_stop_patience: int = 0,
              early_stop_min_delta: float = 1e-4,
              scheduler=None,
              mlflow_params: dict | None = None):
        """
        Full training pipeline.

        Args:
            model          : nn.Module
            train_loader   : DataLoader for training
            val_loader     : DataLoader for validation
            test_loader    : DataLoader for testing
            optimizer      : torch.optim.Optimizer
            epochs         : number of epochs
            criterion_fn   : callable(model, batch, device) -> loss scalar
            eval_fn        : callable(model, eval_loader, device) -> metrics dict
            gradient_clip  : max norm for gradient clipping (0 = disabled)
        """
        print(f"\n{'='*50}")
        print(f"  {self.model_name}")
        print(f"{'='*50}")
        print(f"  Device    : {self.device}")
        print(f"  Epochs    : {epochs}")
        print(f"  Train sz  : {len(train_loader.dataset):,}")
        print(f"  Val sz    : {len(val_loader.dataset):,}")
        if test_loader is not None:
            print(f"  Test sz   : {len(test_loader.dataset):,}")
        print(f"{'='*50}\n")

        if self._use_mlflow and mlflow_params:
            self._mlflow.log_params(_flatten_dict(mlflow_params))

        tags = mlflow_params.get("tags", {}) if mlflow_params else {}
        if tags and self._use_mlflow:
            self._mlflow.set_tags(tags)

        best_val_ndcg = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, criterion_fn, gradient_clip,
                scheduler,
            )

            # Stop immediately if training has collapsed to NaN — parameters are
            # now corrupt, so there is no point continuing or evaluating.
            if math.isnan(train_loss):
                print(
                    f"  Epoch {epoch:02d}/{epochs:02d} | "
                    f"train_loss=nan — NaN detected, stopping early."
                )
                break

            # Validate
            val_metrics = eval_fn(model, val_loader, self.device)
            val_loss = (
                self._compute_val_loss(model, val_loss_loader, criterion_fn)
                if val_loss_loader is not None
                else val_metrics.get("val_loss")
            )

            # Log
            self.tracker.log_epoch(epoch, train_loss, val_loss, val_metrics)

            if self._use_mlflow:
                ml_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else 0.0,
                }
                ml_metrics.update(val_metrics)
                ml_metrics = {sanitize_metric_name(k): v for k, v in ml_metrics.items()}
                self._mlflow.log_metrics(ml_metrics, step=epoch)

            # Print
            hr10 = val_metrics.get("Recall@10", 0)
            ndcg10 = val_metrics.get("NDCG@10", 0)
            val_loss_str = "NA" if val_loss is None else f"{val_loss:.4f}"
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss_str} | "
                f"Recall@10={hr10:.4f} | "
                f"NDCG@10={ndcg10:.4f}"
            )

            # Capture best before tracking — needed for correct early-stopping delta
            prev_best = best_val_ndcg

            # Best model tracking — guard against NaN/Inf metrics (e.g. from NaN
            # embeddings) which would otherwise overwrite a valid checkpoint.
            if math.isfinite(ndcg10) and ndcg10 > best_val_ndcg:
                best_val_ndcg = ndcg10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Early stopping — coexists with NaN early-stop above
            if early_stop_patience > 0:
                if math.isfinite(ndcg10) and ndcg10 > prev_best + early_stop_min_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(
                            f"  Early stopping triggered at epoch {epoch} "
                            f"(no NDCG@10 improvement > {early_stop_min_delta} "
                            f"for {early_stop_patience} epochs)"
                        )
                        break

            # Step ReduceLROnPlateau scheduler after validation (needs metric).
            # All other schedulers (e.g. LambdaLR for warmup) are stepped
            # per-batch inside _train_one_epoch.
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("NDCG@10", 0))

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = {}
        if best_state is not None:
            # Only run test if we have a valid checkpoint
            if test_loader is not None:
                print(f"\n  Evaluating on Test set...")
                test_metrics = eval_fn(model, test_loader, self.device)
            self.tracker.finalize(test_metrics)
            self._save_checkpoint(model, best_val_ndcg)
        else:
            # No valid checkpoint — mark run FAILED and raise so the caller
            # never exports per-user results or treats this as a successful run.
            print(f"\n  Run FAILED: no valid checkpoint produced (all training was non-finite).")
            print(f"  Skipping test evaluation. Skipping checkpoint save.")
            self.tracker.finalize({})

            if self._use_mlflow and self._mlflow_run:
                self._mlflow.tracking.MlflowClient().set_terminated(
                    self._mlflow_run.info.run_id, status="FAILED"
                )
                self._mlflow.end_run()

            self.tracker.save_metrics(self.output_dir / "metrics.json")
            self.tracker.plot_losses(self.output_dir / "loss_plot.png")
            self.tracker.plot_metrics(self.output_dir / "metrics_plot.png")

            raise RuntimeError(
                "No valid checkpoint produced — all training was non-finite. "
                "MLflow run has been marked FAILED. Check learning rate, "
                "gradient clip, or data validity."
            )

        self.tracker.save_metrics(self.output_dir / "metrics.json")
        self.tracker.plot_losses(self.output_dir / "loss_plot.png")
        self.tracker.plot_metrics(self.output_dir / "metrics_plot.png")

        # Print summary
        self._print_summary()

        if self._use_mlflow:
            test_ml_metrics = {
                "best_val_ndcg_at_10": self.tracker.best_val_ndcg,
                "best_epoch": float(self.tracker.best_epoch),
                **{f"test_{sanitize_metric_name(k)}": v for k, v in test_metrics.items()},
            }
            self._mlflow.log_metrics(test_ml_metrics)
            if self._mlflow_log_artifacts:
                if (self.output_dir / "metrics.json").exists():
                    self._mlflow.log_artifact(str(self.output_dir / "metrics.json"), artifact_path="metrics")
                if (self.output_dir / "loss_plot.png").exists():
                    self._mlflow.log_artifact(str(self.output_dir / "loss_plot.png"), artifact_path="plots")
                if (self.output_dir / "metrics_plot.png").exists():
                    self._mlflow.log_artifact(str(self.output_dir / "metrics_plot.png"), artifact_path="plots")
                if (self.output_dir / "best_model.pt").exists():
                    self._mlflow.log_artifact(str(self.output_dir / "best_model.pt"), artifact_path="checkpoints")
            self._mlflow.end_run()

        return self.tracker

    def _compute_val_loss(self, model, val_loss_loader, criterion_fn) -> float:
        """Compute mean loss on val_loss_loader in eval mode."""
        model.eval()
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loss_loader:
                loss = criterion_fn(model, batch, self.device)
                bs = batch[list(batch.keys())[0]].size(0)
                total_loss += loss.item() * bs
                n_samples += bs
        model.train()
        return total_loss / max(n_samples, 1)

    # Internal
    def _train_one_epoch(self, model, loader, optimizer, criterion_fn,
                         gradient_clip: float, scheduler=None):
        model.train()
        total_loss = 0.0
        n_samples = 0

        nan_batches = 0
        total_batches = 0
        for batch in _progress(loader, desc=f"{self.model_name} train"):
            total_batches += 1
            optimizer.zero_grad()
            loss = criterion_fn(model, batch, self.device)

            # Loss is already NaN/Inf — model weights are corrupt.  Report NaN
            # to the outer loop so training stops before we do more damage.
            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            loss.backward()

            # clip_grad_norm_ returns the total gradient norm before clipping.
            # If it is NaN/Inf (overflow in backward pass), discard the update
            # to protect model parameters.  This is the first line of defence
            # against gradient explosion — the batch is skipped cleanly.
            if gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            else:
                grad_norm = torch.stack(
                    [p.grad.norm() for p in model.parameters() if p.grad is not None]
                ).norm()

            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                nan_batches += 1
                continue

            optimizer.step()

            # Step per-batch scheduler (warmup, decay, etc.).
            # ReduceLROnPlateau is excluded — it is stepped per-epoch.
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            bs = batch[list(batch.keys())[0]].size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        # If most batches had non-finite gradients the model has diverged.
        # Return NaN so the outer loop triggers early stopping.
        if nan_batches > 0 and nan_batches * 2 > total_batches:
            return float("nan")

        return total_loss / max(n_samples, 1)

    def _save_checkpoint(self, model, best_val_ndcg: float):
        ckpt = {
            "state_dict": model.state_dict(),
            "model_name": self.model_name,
            "best_val_ndcg": round(best_val_ndcg, 6),
        }
        torch.save(ckpt, self.output_dir / "best_model.pt")

    def _print_summary(self):
        s = self.tracker.summary()
        print(f"\n{'─'*50}")
        print(f"  {s['model_name']} — Summary")
        print(f"{'─'*50}")
        print(f"  Best epoch   : {s['best_epoch']}")
        print(f"  Best val NDCG: {s['best_val_ndcg']:.4f}")
        for k, v in s['test_results'].items():
            print(f"  {k:<12} : {v:.4f}")
        print(f"{'─'*50}")
        print(f"  Artifacts saved to: {self.output_dir}/")
        print("    metrics.json, loss_plot.png, metrics_plot.png, best_model.pt")
