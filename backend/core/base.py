"""
Core base classes for the modular training framework.
All modules inherit from these to ensure pluggability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseModel(ABC):
    """Base class for all model wrappers."""

    @abstractmethod
    def load(self, path: str, **kwargs) -> nn.Module:
        """Load model from path (supports .safetensors, .ckpt, etc.)"""
        pass

    @abstractmethod
    def get_targetable_layers(self) -> Dict[str, List[str]]:
        """Return dict of layer groups that can be targeted by LoRA.
        e.g. {'unet_attention': [...], 'text_encoder_1': [...], 'text_encoder_2': [...]}
        """
        pass

    @abstractmethod
    def forward_pass(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Run forward pass, return dict with at least 'loss' key."""
        pass

    @abstractmethod
    def prepare_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space if applicable."""
        pass

    @abstractmethod
    def encode_text(self, text: List[str]) -> Any:
        """Encode text prompts."""
        pass


class BaseLoss(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor,
                **kwargs) -> torch.Tensor:
        pass


class BaseOptimizer(ABC):
    """Base class for optimizer wrappers."""

    @abstractmethod
    def create(self, parameters, **kwargs) -> torch.optim.Optimizer:
        pass


class BaseScheduler(ABC):
    """Base class for LR scheduler wrappers."""

    @abstractmethod
    def create(self, optimizer: torch.optim.Optimizer, **kwargs) -> Any:
        pass


class BaseDataset(ABC):
    """Base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass


class BaseCallback(ABC):
    """Base class for training callbacks (logging, checkpointing, etc.)."""

    def on_train_start(self, state: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, state: Dict[str, Any]) -> None:
        pass

    def on_epoch_start(self, epoch: int, state: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        pass

    def on_step_start(self, step: int, state: Dict[str, Any]) -> None:
        pass

    def on_step_end(self, step: int, loss: float, state: Dict[str, Any]) -> None:
        pass


class BaseTrainer(ABC):
    """Base class for trainers."""

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        pass
