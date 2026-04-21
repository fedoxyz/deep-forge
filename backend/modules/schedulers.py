"""
Learning rate scheduler module.
"""

from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    ConstantLR,
    SequentialLR,
    OneCycleLR,
)

from backend.core.base import BaseScheduler


class CosineScheduler(BaseScheduler):
    def __init__(self, T_max: int = 1000, eta_min: float = 1e-7):
        self.T_max = T_max
        self.eta_min = eta_min

    def create(self, optimizer, **kwargs):
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', self.T_max),
            eta_min=self.eta_min,
        )


class CosineWarmupScheduler(BaseScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        eta_min: float = 1e-7,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

    def create(self, optimizer, **kwargs):
        warmup_steps = kwargs.get('warmup_steps', self.warmup_steps)
        total_steps = kwargs.get('total_steps', self.total_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.eta_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )


class ConstantWarmupScheduler(BaseScheduler):
    """Constant LR with linear warmup."""

    def __init__(self, warmup_steps: int = 100):
        self.warmup_steps = warmup_steps

    def create(self, optimizer, **kwargs):
        warmup_steps = kwargs.get('warmup_steps', self.warmup_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        constant = ConstantLR(optimizer, factor=1.0, total_iters=999999)
        return SequentialLR(
            optimizer,
            schedulers=[warmup, constant],
            milestones=[warmup_steps],
        )


class OneCycleSchedulerWrapper(BaseScheduler):
    def __init__(self, max_lr: float = 1e-4, total_steps: int = 1000):
        self.max_lr = max_lr
        self.total_steps = total_steps

    def create(self, optimizer, **kwargs):
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', self.max_lr),
            total_steps=kwargs.get('total_steps', self.total_steps),
        )


SCHEDULER_REGISTRY = {
    'cosine': CosineScheduler,
    'cosine_warmup': CosineWarmupScheduler,
    'constant_warmup': ConstantWarmupScheduler,
    'one_cycle': OneCycleSchedulerWrapper,
}


def create_scheduler(name: str, optimizer, **kwargs):
    """Create a scheduler by name."""
    if name.lower() not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    wrapper = SCHEDULER_REGISTRY[name.lower()](**kwargs)
    return wrapper.create(optimizer, **kwargs)
