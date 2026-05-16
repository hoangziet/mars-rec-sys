"""
pipeline/optim.py
=================
Optimizer and scheduler helpers shared by training entry points.
"""

import torch
import torch.nn as nn


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


def _layer_norm_param_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                names.add(full_name)
    return names


def build_optimizer(
    model_name: str,
    model: torch.nn.Module,
    train_kwargs: dict,
) -> torch.optim.Optimizer:
    lr = train_kwargs.get("lr", 1e-3)
    beta2 = train_kwargs.get("beta2", 0.999)
    weight_decay = train_kwargs.get("weight_decay", 0.0)

    if model_name == "bert4rec":
        layer_norm_names = _layer_norm_param_names(model)
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or name in layer_norm_names:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, beta2),
            eps=1e-6,
        )

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
