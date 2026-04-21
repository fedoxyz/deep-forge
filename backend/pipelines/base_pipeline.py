"""
Abstract base class for diffusion training pipelines.

A pipeline knows HOW to:
  1. Encode text prompts → conditioning tensors
  2. Encode images → latents (VAE encode)
  3. Add noise at timestep t (forward diffusion / flow matching)
  4. Run the denoiser (DiT/UNet) to predict noise/velocity
  5. Compute the training loss
  6. Decode latents → images (VAE decode, for sampling)
  7. Run full inference (for generating samples during training)

The pipeline does NOT own the training loop — UnifiedTrainer does.
The pipeline is a stateless helper that the trainer calls into.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PipelineComponents:
    """Holds references to the actual nn.Modules a pipeline uses."""
    denoiser: Optional[nn.Module] = None
    text_encoder: Optional[nn.Module] = None
    vae: Optional[nn.Module] = None
    # Extra components (e.g. SigLIP for Z-Image-Edit, second CLIP for Flux)
    extras: Dict[str, nn.Module] = field(default_factory=dict)


@dataclass
class SampleRequest:
    """What the user wants when they click 'Sample'."""
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 8
    guidance_scale: float = 0.0
    seed: Optional[int] = None
    # Which sampler to use (euler, euler_a, dpm++, etc.)
    sampler: str = "euler"


@dataclass
class SampleResult:
    """Returned after generating samples."""
    images: List[Any]  # PIL Images or tensors
    seeds: List[int]
    prompts: List[str]
    step: int  # training step when sample was taken
    epoch: int


class BaseDiffusionPipeline(ABC):
    """
    Abstract pipeline. Subclass for each model family:
      - ZImageTurboPipeline
      - FluxPipeline
      - SDXLPipeline
      - etc.
    """

    name: str = "base"
    # Metadata for UI
    display_name: str = "Base Pipeline"
    description: str = ""
    # Default sampling params (overridable in config)
    default_num_steps: int = 20
    default_guidance_scale: float = 7.5
    default_sampler: str = "euler"
    # Whether the model uses flow matching vs DDPM
    uses_flow_matching: bool = False

    def __init__(self, components: PipelineComponents, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        self.components = components
        self.device = device
        self.dtype = dtype

    # ── Required implementations ──

    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None
                      ) -> Dict[str, torch.Tensor]:
        """Encode text prompt into conditioning tensors.
        Returns dict with at least 'prompt_embeds' key."""
        ...

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space image [B,C,H,W] into latent space."""
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to pixel space [B,C,H,W] in [0,1]."""
        ...

    @abstractmethod
    def get_noise(self, latent_shape: Tuple[int, ...],
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample noise matching the latent shape."""
        ...

    @abstractmethod
    def get_timesteps(self, batch_size: int,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample random timesteps for training."""
        ...

    @abstractmethod
    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Forward process: add noise to clean latents at given timesteps."""
        ...

    @abstractmethod
    def compute_target(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                       timesteps: torch.Tensor) -> torch.Tensor:
        """What the model should predict (noise, v, velocity, etc.)."""
        ...

    @abstractmethod
    def forward_denoise(self, noisy_latents: torch.Tensor,
                        timesteps: torch.Tensor,
                        condition: Dict[str, torch.Tensor],
                        **kwargs) -> torch.Tensor:
        """Run denoiser, return predicted target."""
        ...

    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                     timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute training loss."""
        ...

    @abstractmethod
    @torch.no_grad()
    def sample(self, request: SampleRequest) -> SampleResult:
        """Full inference loop: prompt → image. Used during training for preview."""
        ...

    # ── Training step (usually not overridden) ──

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        """
        Standard diffusion training step.
        Most pipelines can use this default; override if needed.

        Expected batch keys:
          - 'pixel_values' or 'latents': [B, C, H, W] images or pre-encoded latents
          - 'prompt_embeds' or 'input_ids': text conditioning (pre-encoded or raw)
          - (optional) 'caption': raw text strings for on-the-fly encoding
        """
        # 1. Get or encode latents
        if 'latents' in batch:
            latents = batch['latents'].to(self.device, dtype=self.dtype)
        elif 'pixel_values' in batch:
            latents = self.encode_image(batch['pixel_values'].to(self.device, dtype=self.dtype))
        else:
            raise ValueError("Batch must contain 'latents' or 'pixel_values'")

        # 2. Get or encode conditioning
        if 'prompt_embeds' in batch:
            condition = {k: v.to(self.device) for k, v in batch.items()
                         if k.startswith('prompt') or k.startswith('cond')}
        elif 'caption' in batch:
            # Encode on the fly (less efficient, but flexible)
            captions = batch['caption']
            if isinstance(captions, str):
                captions = [captions]
            # Batch encode
            conditions = [self.encode_prompt(c) for c in captions]
            condition = {k: torch.stack([c[k] for c in conditions])
                         for k in conditions[0]}
        else:
            condition = {}

        bs = latents.shape[0]

        # 3. Sample noise and timesteps
        noise = self.get_noise(latents.shape)
        timesteps = self.get_timesteps(bs)

        # 4. Forward diffusion
        noisy_latents = self.add_noise(latents, noise, timesteps)

        # 5. Predict
        prediction = self.forward_denoise(noisy_latents, timesteps, condition)

        # 6. Target
        target = self.compute_target(latents, noise, timesteps)

        # 7. Loss
        loss = self.compute_loss(prediction, target, timesteps)

        return loss, {
            'loss': loss.item(),
            'timesteps': timesteps,
        }

    # ── Utilities ──

    def get_latent_shape(self, height: int, width: int, batch_size: int = 1
                         ) -> Tuple[int, ...]:
        """Get latent tensor shape for given image dimensions.
        Override per model family (different VAE scale factors)."""
        # Default: 8x downscale, 4 channels
        return (batch_size, 4, height // 8, width // 8)

    def to(self, device: torch.device):
        self.device = device
        for attr in ('denoiser', 'text_encoder', 'vae'):
            mod = getattr(self.components, attr, None)
            if mod is not None:
                mod.to(device)
        for mod in self.components.extras.values():
            mod.to(device)
        return self

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'uses_flow_matching': self.uses_flow_matching,
            'default_num_steps': self.default_num_steps,
            'default_guidance_scale': self.default_guidance_scale,
            'default_sampler': self.default_sampler,
        }
