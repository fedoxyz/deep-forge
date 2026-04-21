"""
Z-Image Turbo training pipeline.

Architecture recap:
  - S3-DiT: 30 transformer layers, hidden=3840, 32 heads, FFN=10240, 6.15B params
  - Text encoder: Qwen3-4B (frozen)
  - VAE: Flux VAE (ae.safetensors, frozen)
  - Single-stream: text tokens + image latent tokens concatenated
  - Flow matching: linear interpolation x_t = (1-t)*x_0 + t*noise
  - Model predicts velocity v = dx/dt
  - Turbo: distilled for 8 NFE, guidance_scale=0.0
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from backend.pipelines.base_pipeline import (
    BaseDiffusionPipeline,
    PipelineComponents,
    SampleRequest,
    SampleResult,
)
from backend.pipelines.registry import register_pipeline
from backend.pipelines.samplers import create_sampler


class ZImageTurboPipeline(BaseDiffusionPipeline):
    """
    Training + inference pipeline for Z-Image Turbo (S3-DiT).

    Components expected in ComponentBundle:
      - 'dit' or 'denoiser': the S3-DiT transformer (6.15B)
      - 'text_encoder': Qwen3-4B
      - 'vae': Flux VAE (ae.safetensors)

    Flow matching formulation:
      x_t = (1 - t) * x_0 + t * eps     (linear interpolation)
      target = eps - x_0                   (velocity = dx/dt)
      loss = ||model(x_t, t, cond) - target||^2
    """

    name = "zimage_turbo"
    display_name = "Z-Image Turbo"
    description = "S3-DiT flow matching pipeline for Z-Image Turbo (6.15B)"
    uses_flow_matching = True
    default_num_steps = 8
    default_guidance_scale = 0.0
    default_sampler = "euler"

    # VAE scale factor (Flux VAE uses 16-channel latents with 8x downscale)
    vae_scale_factor = 8
    vae_latent_channels = 16

    def __init__(self, components: PipelineComponents, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16,
                 # Flow matching params
                 shift: float = 1.0,
                 # Loss weighting
                 loss_type: str = "mse",
                 snr_gamma: Optional[float] = None,
                 max_sequence_length: int = 512,
                 **kwargs
                 ):
        super().__init__(components, device, dtype)
        self.shift = shift
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma
        self.max_sequence_length = max_sequence_length
        self.training_adapter = components.extras.get('training_adapter', None)
        self._low_vram = kwargs.get('low_vram', False)

    # ── Text encoding ──
    def encode_prompt(self, prompt: str, negative_prompt=None):
        te = self.components.text_encoder
        if te is None:
            raise RuntimeError("Text encoder not loaded")
        if not prompt or not prompt.strip():
            prompt = "."
        # ── ADD: move to device if low_vram ──
        if self._low_vram:
            te.to(self.device)
        result = {'prompt_embeds': te.encode(prompt)}
        if negative_prompt is not None:
            result['negative_prompt_embeds'] = te.encode(negative_prompt)
        if self._low_vram:
            te.to(torch.device('cpu'))
            torch.cuda.empty_cache()
        return result

    # ── VAE ──
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")
        # ── ADD: move to device if low_vram ──
        if self._low_vram:
            vae.to(self.device)
        with torch.no_grad():
            enc = vae.encode(image.to(self.device, self.dtype))
            latents = enc.latent_dist.sample() if hasattr(enc, 'latent_dist') else enc
        if self._low_vram:
            vae.to(torch.device('cpu'))
            torch.cuda.empty_cache()
        return latents.to(self.device, self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")
    
        # Do NOT pre-divide — FluxVAEWrapper.decode handles unscaling internally
        actual_device = next(iter(vae.parameters())).device
        with torch.no_grad():
            dec = vae.decode(latents.to(actual_device, self.dtype))
            images = dec.sample if hasattr(dec, 'sample') else dec
    
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.to(self.device)

    # ── Flow matching noise schedule ──

    def get_noise(self, latent_shape: Tuple[int, ...],
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample Gaussian noise."""
        return torch.randn(latent_shape, device=self.device, dtype=self.dtype,
                           generator=generator)

    def get_timesteps(self, batch_size: int,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample random timesteps in [0, 1] for flow matching.

        For distilled models like Turbo, we use uniform sampling.
        The shift parameter can be used for logit-normal or shifted distributions.
        """
        # Uniform sampling in [0, 1]
        t = torch.rand(batch_size, device=self.device, dtype=self.dtype,
                        generator=generator)

        # Apply shift if needed (shifts distribution towards higher noise levels)
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)

        return t

    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Flow matching forward process:
          x_t = (1 - t) * x_0 + t * eps

        timesteps shape: [B] → reshape for broadcasting
        """
        t = timesteps
        while t.dim() < clean_latents.dim():
            t = t.unsqueeze(-1)

        noisy = (1.0 - t) * clean_latents + t * noise
        return noisy

    def compute_target(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                       timesteps: torch.Tensor) -> torch.Tensor:
        """
        Flow matching target: velocity = eps - x_0 = dx/dt
        (derivative of x_t = (1-t)*x_0 + t*eps w.r.t. t)
        """
        return noise - clean_latents

    # ── Denoiser forward ──
    def forward_denoise(self, noisy_latents, timesteps, condition, **kwargs):
        dit = self.components.denoiser
        if dit is None:
            raise RuntimeError("Denoiser (DiT) not loaded")
    
        if isinstance(condition, dict):
            cap_feats = condition.get('prompt_embeds')
            if isinstance(cap_feats, torch.Tensor):
                cap_feats = [cap_feats.to(self.dtype)]  # cast to pipeline dtype
            elif isinstance(cap_feats, list):
                cap_feats = [f.to(self.dtype) for f in cap_feats]
        else:
            cap_feats = condition
    
        return dit(noisy_latents, timesteps, cap_feats)

    # ── Loss ──

    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                     timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute flow matching loss.

        For distilled models, MSE is standard.
        Optional: min-SNR weighting to stabilize training.
        """
        if self.loss_type == "huber":
            loss = F.huber_loss(prediction, target, reduction='none')
        else:
            loss = F.mse_loss(prediction, target, reduction='none')

        # Per-sample loss: mean over spatial dims
        loss = loss.mean(dim=list(range(1, loss.dim())))

        # Optional SNR weighting
        if self.snr_gamma is not None:
            # For flow matching, SNR ≈ (1-t)²/t²
            t = timesteps.clamp(1e-5, 1 - 1e-5)
            snr = ((1 - t) / t) ** 2
            weight = torch.clamp(snr, max=self.snr_gamma) / snr
            loss = loss * weight

        return loss.mean()

    # ── Inference / Sampling ──

    @torch.no_grad()
    def sample(self, request: SampleRequest) -> SampleResult:
        """
        Full inference loop for generating preview samples during training.

        For Z-Image Turbo:
          - 8 steps (NFE)
          - guidance_scale = 0.0 (no CFG, it's distilled)
          - Euler sampler
        """
        # Use defaults if not specified
        num_steps = request.num_steps or self.default_num_steps
        guidance = request.guidance_scale if request.guidance_scale is not None else self.default_guidance_scale
        sampler_name = request.sampler or self.default_sampler

        sampler = create_sampler(sampler_name, num_steps=num_steps, shift=self.shift)
        schedule = sampler.get_schedule(device=self.device)

        all_images = []
        all_seeds = []

        for i, prompt in enumerate(request.prompts):
            seed = (request.seed + i) if request.seed is not None else torch.randint(0, 2**32, (1,)).item()
            gen = torch.Generator(device=self.device).manual_seed(seed)

            # Encode prompt
            cond = self.encode_prompt(prompt, request.negative_prompts[i]
                                      if request.negative_prompts and i < len(request.negative_prompts)
                                      else None)

            # Initial noise
            latent_shape = self.get_latent_shape(request.height, request.width, batch_size=1)
            latents = self.get_noise(latent_shape, generator=gen)

            # Denoise loop
            if hasattr(sampler, 'reset'):
                sampler.reset()

            for step_idx in range(num_steps):
                t_current = schedule[step_idx]
                t_next = schedule[step_idx + 1]

                # Model prediction
                t_tensor = torch.tensor([t_current], device=self.device, dtype=self.dtype)
                pred = self.forward_denoise(latents, t_tensor, cond)

                # Sampler step
                latents = sampler.step(pred, t_current.item(), t_next.item(), latents,
                                       generator=gen) if hasattr(sampler.step, '__code__') and \
                    sampler.step.__code__.co_varnames[:6].__contains__('generator') else \
                    sampler.step(pred, t_current.item(), t_next.item(), latents)

            # Decode
            pixels = self.decode_latents(latents)

            # Convert to PIL
            img = pixels[0].cpu().float().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype('uint8')
            pil_img = Image.fromarray(img)

            all_images.append(pil_img)
            all_seeds.append(seed)

        return SampleResult(
            images=all_images,
            seeds=all_seeds,
            prompts=request.prompts,
            step=0,  # Will be set by caller
            epoch=0,
        )

    # ── Latent shape ──

    def get_latent_shape(self, height: int, width: int, batch_size: int = 1
                         ) -> Tuple[int, ...]:
        """Flux VAE: 16 channels, 8x downscale."""
        return (
            batch_size,
            self.vae_latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Pixel values
        pixels = batch.get('pixel_values', batch.get('input'))
        if pixels is None:
            raise ValueError(f"training_step: no 'pixel_values'/'input' in batch. Keys: {list(batch.keys())}")
        pixels = pixels.to(self.device, self.dtype)
    
        # 2. VAE encode — frozen, already on GPU, runs fine
        latents = self.encode_image(pixels)
        latents = latents.to(self.device, self.dtype)
    
        # 3. Text encode — frozen, already on GPU, runs fine
        if 'prompt_embeds' in batch:
            condition = {'prompt_embeds': batch['prompt_embeds'].to(self.device, self.dtype)}
        else:
            caption = batch.get('caption', batch.get('text', None))
            if not caption:
                caption = '.'
            if isinstance(caption, (list, tuple)):
                caption = caption[0] if caption else '.'
            if not str(caption).strip():
                caption = '.'
            condition = self.encode_prompt(caption)
            if isinstance(condition, dict):
                condition = {
                    k: v.to(self.device, self.dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in condition.items()
                }
    
        # 4. Flow matching — pure math, no model, no GPU memory spike
        noise     = self.get_noise(latents.shape)
        timesteps = self.get_timesteps(latents.shape[0])
        noisy     = self.add_noise(latents, noise, timesteps)
        target    = self.compute_target(latents, noise, timesteps)
    
        # ── 5. FREE frozen model memory before DiT forward ──
        # In low_vram mode: explicitly clear Qwen3 + VAE activations from GPU
        # before the 6.15B DiT forward pass to avoid OOM.
        if getattr(self, '_low_vram', False):
            torch.cuda.empty_cache()
    
        # 6. DiT forward — BundleWrapper._component_on_device handles move if low_vram
        denoiser_device = next(self.components.denoiser.parameters()).device
        noisy     = noisy.to(denoiser_device, self.dtype)
        timesteps = timesteps.to(denoiser_device)
        if isinstance(condition, dict):
            condition = {
                k: v.to(denoiser_device, self.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in condition.items()
            }
    
        prediction = self.forward_denoise(noisy, timesteps, condition)
    
        # 7. Loss
        loss = self.compute_loss(prediction, target, timesteps)
        return {'loss': loss}

# Register
register_pipeline("zimage_turbo", ZImageTurboPipeline)
