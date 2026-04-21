"""
Loss function module. Composable and extensible.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.base import BaseLoss


class MSELoss(BaseLoss):
    """Standard MSE loss (default for diffusion model noise prediction)."""

    def __init__(self, reduction: str = 'mean'):
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def compute(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions, targets)


class HuberLoss(BaseLoss):
    """Huber loss (smooth L1) - more robust to outliers."""

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        self.loss_fn = nn.HuberLoss(delta=delta, reduction=reduction)

    def compute(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions, targets)


class SNRWeightedLoss(BaseLoss):
    """
    Min-SNR weighted loss for diffusion training.
    Weights the loss based on signal-to-noise ratio at each timestep.
    From: https://arxiv.org/abs/2303.09556
    """

    def __init__(self, gamma: float = 5.0, base_loss: str = 'mse'):
        self.gamma = gamma
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none')
        else:
            self.base_loss = nn.MSELoss(reduction='none')

    def compute(self, predictions, targets, **kwargs):
        snr = kwargs.get('snr', None)
        loss = self.base_loss(predictions, targets)

        if snr is not None:
            # min(snr, gamma) / snr
            weight = torch.clamp(snr, max=self.gamma) / snr
            # Reshape weight for broadcasting
            while weight.dim() < loss.dim():
                weight = weight.unsqueeze(-1)
            loss = loss * weight

        return loss.mean()


class VPredLoss(BaseLoss):
    """V-prediction loss for v-parameterized diffusion models."""

    def __init__(self, reduction: str = 'mean'):
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def compute(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions, targets)


class CompositeLoss(BaseLoss):
    """
    Combine multiple loss functions with weights.
    Allows experimenting with loss combinations from papers.

    Usage:
        loss = CompositeLoss([
            (MSELoss(), 1.0),
            (PerceptualLoss(), 0.1),
        ])
    """

    def __init__(self, losses: List[tuple]):
        """
        Args:
            losses: List of (loss_fn, weight) tuples
        """
        self.losses = losses

    def compute(self, predictions, targets, **kwargs):
        total = 0.0
        for loss_fn, weight in self.losses:
            total = total + weight * loss_fn.compute(predictions, targets, **kwargs)
        return total


LOSS_REGISTRY = {
    'mse': MSELoss,
    'huber': HuberLoss,
    'snr_weighted': SNRWeightedLoss,
    'v_pred': VPredLoss,
}


def create_loss(name: str, **kwargs) -> BaseLoss:
    """Create a loss function by name."""
    if name.lower() not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name.lower()](**kwargs)
