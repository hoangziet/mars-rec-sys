"""
pipeline/optim.py
=================
Optimizer and scheduler helpers shared by training entry points.
"""

import torch


class LinearWarmupDecayScheduler:
    """Linear warmup + linear decay scheduler matching the BERT4Rec reference."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        num_train_steps: int,
        num_warmup_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.num_train_steps = max(int(num_train_steps), 1)
        self.num_warmup_steps = max(int(num_warmup_steps), 0)
        self.global_step = 0
        self._set_lr(self._lr_for_step(self.global_step))

    def _lr_for_step(self, step: int) -> float:
        step = min(max(int(step), 0), self.num_train_steps)
        decay_lr = self.init_lr * max(0.0, 1.0 - (step / self.num_train_steps))
        if self.num_warmup_steps > 0 and step < self.num_warmup_steps:
            return self.init_lr * (step / self.num_warmup_steps)
        return decay_lr

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.global_step += 1
        self._set_lr(self._lr_for_step(self.global_step))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def build_optimizer(
    model_name: str,
    model: torch.nn.Module,
    train_kwargs: dict,
) -> torch.optim.Optimizer:
    lr = train_kwargs.get("lr", 1e-3)
    beta2 = train_kwargs.get("beta2", 0.999)
    weight_decay = train_kwargs.get("weight_decay", 0.0)

    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, beta2),
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_kwargs: dict,
    num_train_batches: int,
):
    warmup_steps = train_kwargs.get("warmup_steps", 0)
    if warmup_steps <= 0:
        return None

    total_steps = train_kwargs["epochs"] * num_train_batches
    return LinearWarmupDecayScheduler(
        optimizer=optimizer,
        init_lr=train_kwargs.get("lr", 1e-3),
        num_train_steps=total_steps,
        num_warmup_steps=warmup_steps,
    )
