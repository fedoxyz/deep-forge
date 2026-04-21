"""
Optimizer module. Pluggable optimizers with sensible defaults.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from backend.core.base import BaseOptimizer


class AdamWOptimizer(BaseOptimizer):
    def __init__(
        self,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def create(self, parameters, **kwargs) -> torch.optim.Optimizer:
        lr = kwargs.get('lr', self.lr)
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    def create_from_groups(self, param_groups: list, **kwargs) -> torch.optim.Optimizer:
        """Create optimizer with per-component param groups.
        
        Each group can override: lr, weight_decay.
        Values not specified in a group fall back to the global defaults.
        """
        # Clean out non-optimizer keys from groups (metadata we added)
        cleaned = []
        for group in param_groups:
            g = {k: v for k, v in group.items() if not k.startswith('_')}
            cleaned.append(g)
        
        # Global defaults passed as constructor args — groups override these
        return optim.AdamW(
            cleaned,
            lr=kwargs.get('lr', self.lr),
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


class Adam8bitOptimizer(BaseOptimizer):
    """8-bit Adam via bitsandbytes (memory efficient)."""

    def __init__(self, lr: float = 1e-4, weight_decay: float = 0.01):
        self.lr = lr
        self.weight_decay = weight_decay

    def create(self, parameters, **kwargs) -> torch.optim.Optimizer:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                parameters,
                lr=kwargs.get('lr', self.lr),
                weight_decay=self.weight_decay,
            )
        except ImportError:
            print("[Optimizer] bitsandbytes not available, falling back to AdamW")
            return optim.AdamW(
                parameters,
                lr=kwargs.get('lr', self.lr),
                weight_decay=self.weight_decay,
            )

    def create_from_groups(self, param_groups: list, **kwargs) -> torch.optim.Optimizer:
        cleaned = [{k: v for k, v in g.items() if not k.startswith('_')} for g in param_groups]
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                cleaned,
                lr=kwargs.get('lr', self.lr),
                weight_decay=self.weight_decay,
            )
        except ImportError:
            print("[Optimizer] bitsandbytes not available, falling back to AdamW")
            return optim.AdamW(cleaned, lr=kwargs.get('lr', self.lr), weight_decay=self.weight_decay)


class SGDOptimizer(BaseOptimizer):
    """SGD with momentum."""

    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def create(self, parameters, **kwargs) -> torch.optim.Optimizer:
        return optim.SGD(
            parameters,
            lr=kwargs.get('lr', self.lr),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def create_from_groups(self, param_groups: list, **kwargs) -> torch.optim.Optimizer:
        cleaned = [{k: v for k, v in g.items() if not k.startswith('_')} for g in param_groups]
        return optim.SGD(
            cleaned,
            lr=kwargs.get('lr', self.lr),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )


class ProdigyOptimizer(BaseOptimizer):
    """Prodigy optimizer (adaptive LR)."""

    def __init__(self, lr: float = 1.0, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def create(self, parameters, **kwargs) -> torch.optim.Optimizer:
        try:
            from prodigyopt import Prodigy
            return Prodigy(
                parameters,
                lr=kwargs.get('lr', self.lr),
                weight_decay=self.weight_decay,
            )
        except ImportError:
            print("[Optimizer] prodigyopt not available, falling back to AdamW")
            return optim.AdamW(parameters, lr=1e-4, weight_decay=self.weight_decay)

    def create_from_groups(self, param_groups: list, **kwargs) -> torch.optim.Optimizer:
        cleaned = [{k: v for k, v in g.items() if not k.startswith('_')} for g in param_groups]
        try:
            from prodigyopt import Prodigy
            return Prodigy(cleaned, lr=kwargs.get('lr', self.lr), weight_decay=self.weight_decay)
        except ImportError:
            print("[Optimizer] prodigyopt not available, falling back to AdamW")
            return optim.AdamW(cleaned, lr=1e-4, weight_decay=self.weight_decay)


# Registry for easy lookup
OPTIMIZER_REGISTRY = {
    'adamw': AdamWOptimizer,
    'adam8bit': Adam8bitOptimizer,
    'sgd': SGDOptimizer,
    'prodigy': ProdigyOptimizer,
}

def create_optimizer(
    name: str, parameters, **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer by name.
    
    `parameters` can be:
      - list of torch.nn.Parameter (flat)
      - list of param group dicts: [{"params": [...], "lr": 1e-5}, ...]
    
    kwargs become defaults for groups that don't specify their own values.
    """
    if name.lower() not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    
    # Detect param groups vs flat param list
    is_param_groups = (
        isinstance(parameters, list) 
        and len(parameters) > 0 
        and isinstance(parameters[0], dict)
    )
    
    wrapper = OPTIMIZER_REGISTRY[name.lower()](**kwargs)
    
    if is_param_groups:
        return wrapper.create_from_groups(parameters, **kwargs)
    return wrapper.create(parameters, **kwargs)
