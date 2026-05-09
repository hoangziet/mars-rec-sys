"""
trainer.py
==========
Unified training framework for all recommendation models.

Classes:
    ExperimentTracker — per-epoch metrics, JSON export, plot generation
    Trainer           — unified training loop, checkpointing, evaluation
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


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
        metrics = ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]
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
                 output_dir: str | Path = "experiments"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(model_name)

    # Public API
    def train(self, model, train_loader, val_loader, test_loader,
              optimizer, epochs: int,
              criterion_fn, eval_fn,
              gradient_clip: float = 5.0):
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
        print(f"  Test sz   : {len(test_loader.dataset):,}")
        print(f"{'='*50}\n")

        best_val_ndcg = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, criterion_fn, gradient_clip
            )

            # Validate
            val_metrics = eval_fn(model, val_loader, self.device)
            val_loss = val_metrics.get("val_loss")

            # Log
            self.tracker.log_epoch(epoch, train_loss, val_loss, val_metrics)

            # Print
            hr10 = val_metrics.get("HR@10", 0)
            ndcg10 = val_metrics.get("NDCG@10", 0)
            val_loss_str = "NA" if val_loss is None else f"{val_loss:.4f}"
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss_str} | "
                f"HR@10={hr10:.4f} | "
                f"NDCG@10={ndcg10:.4f}"
            )

            # Best model tracking
            if ndcg10 > best_val_ndcg:
                best_val_ndcg = ndcg10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Test
        print(f"\n  Evaluating on Test set...")
        test_metrics = eval_fn(model, test_loader, self.device)
        self.tracker.finalize(test_metrics)

        # Save everything
        self._save_checkpoint(model, best_val_ndcg)
        self.tracker.save_metrics(self.output_dir / "metrics.json")
        self.tracker.plot_losses(self.output_dir / "loss_plot.png")
        self.tracker.plot_metrics(self.output_dir / "metrics_plot.png")

        # Print summary
        self._print_summary()

        return self.tracker

    # Internal
    def _train_one_epoch(self, model, loader, optimizer, criterion_fn,
                         gradient_clip: float):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(loader, desc=f"  [{self.model_name}] Train", leave=False):
            optimizer.zero_grad()
            loss = criterion_fn(model, batch, self.device)
            loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            bs = batch[list(batch.keys())[0]].size(0)
            total_loss += loss.item() * bs
            n_samples += bs

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
