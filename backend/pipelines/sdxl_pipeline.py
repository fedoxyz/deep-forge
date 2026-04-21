"""
SDXL training + inference pipeline.

Architecture:
  - UNet: SDXL UNet2DConditionModel (~2.6B params)
  - Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-bigG/14 (both frozen)
  - VAE: SDXL VAE (4 channel latents, 8x downscale)
  - Noise scheduler: DDPM / Euler discrete
  - Uses epsilon prediction (not velocity like flow matching)
  - CFG guidance_scale typically 5.0-9.0
  
Key differences from Z-Image Turbo:
  - Epsilon prediction, not velocity/flow matching
  - Discrete timesteps (0-999), not continuous [0,1]
  - Two text encoders (CLIP + OpenCLIP) → pooled + sequence embeds
  - VAE has 4 channels (not 16)
  - Needs CFG for inference (not distilled)
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


class SDXLPipeline(BaseDiffusionPipeline):
    
    name = "sdxl"
    display_name = "Stable Diffusion XL"
    description = "SDXL UNet pipeline with dual text encoders"
    uses_flow_matching = False
    default_num_steps = 30
    default_guidance_scale = 7.5
    default_sampler = "euler_discrete"

    vae_scale_factor = 8
    vae_latent_channels = 4
    num_train_timesteps = 1000

    def __init__(self, components: PipelineComponents, device: torch.device,
                 dtype: torch.dtype = torch.float16,
                 prediction_type: str = "epsilon",  # "epsilon" | "v_prediction"
                 snr_gamma: Optional[float] = None,
                 offset_noise: float = 0.0,
                 ):
        super().__init__(components, device, dtype)
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.offset_noise = offset_noise
        
        # Precompute noise schedule (linear beta schedule)
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, self.num_train_timesteps) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        
        # Integration consideration: the second text encoder is stored
        # in components.extras['text_encoder_2']. The component config
        # should use role='text_encoder' for CLIP and a second component
        # with name='text_encoder_2' and role='text_encoder'.
        self.text_encoder_2 = components.extras.get('text_encoder_2', None)

    # ── Text encoding ──

    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None
                      ) -> Dict[str, torch.Tensor]:
        """
        Dual text encoder: CLIP ViT-L + OpenCLIP ViT-bigG.
        Returns pooled_embeds + sequence embeds concatenated.
        
        Integration consideration: 
        - text_encoder (CLIP) → hidden_states from penultimate layer
        - text_encoder_2 (OpenCLIP) → hidden_states + pooled output
        - Final prompt_embeds = concat([clip_embeds, openclip_embeds], dim=-1)
        - pooled_prompt_embeds = openclip pooled output
        - These are passed to UNet as encoder_hidden_states + add_text_embeds
        """
        te1 = self.components.text_encoder
        te2 = self.text_encoder_2
        
        result = {}
        
        with torch.no_grad():
            # Integration consideration: actual tokenization depends on
            # whether the text encoders are wrapped with tokenizers.
            # The component loader should handle this based on the model files.
            #
            # For CLIP: use CLIPTokenizer, max_length=77
            # For OpenCLIP: use CLIPTokenizer (openclip variant), max_length=77
            #
            # If using diffusers-wrapped models:
            #   embeds_1 = te1(input_ids, output_hidden_states=True).hidden_states[-2]
            #   out_2 = te2(input_ids_2, output_hidden_states=True)
            #   embeds_2 = out_2.hidden_states[-2]
            #   pooled = out_2[0]  # pooled output
            
            if hasattr(te1, 'encode'):
                embeds_1 = te1.encode(prompt)
            else:
                embeds_1 = te1(prompt)
            
            if te2 is not None:
                if hasattr(te2, 'encode'):
                    out_2 = te2.encode(prompt)
                    if isinstance(out_2, dict):
                        embeds_2 = out_2.get('hidden_states', out_2.get('embeds'))
                        pooled = out_2.get('pooled_output', out_2.get('pooled'))
                    else:
                        embeds_2 = out_2
                        pooled = None
                else:
                    embeds_2 = te2(prompt)
                    pooled = None
                
                # Concat along feature dim
                if embeds_2 is not None:
                    prompt_embeds = torch.cat([embeds_1, embeds_2], dim=-1)
                else:
                    prompt_embeds = embeds_1
                
                if pooled is not None:
                    result['pooled_prompt_embeds'] = pooled
            else:
                prompt_embeds = embeds_1
            
            result['prompt_embeds'] = prompt_embeds

        # Negative prompt encoding (same structure)
        if negative_prompt is not None:
            with torch.no_grad():
                # Same dual-encoder path for negative prompt
                # ... mirror the above logic ...
                pass  # implement same as above with negative_prompt
            
        return result

    # ── VAE ──

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode to 4-channel latent space."""
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")
        
        with torch.no_grad():
            if hasattr(vae, 'encode'):
                enc = vae.encode(image.to(self.dtype))
                if hasattr(enc, 'latent_dist'):
                    latents = enc.latent_dist.sample()
                elif hasattr(enc, 'latents'):
                    latents = enc.latents
                else:
                    latents = enc
            else:
                latents = vae(image.to(self.dtype))
        
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents * vae.config.scaling_factor
        else:
            latents = latents * 0.13025  # SDXL default
        
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")
        
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        else:
            latents = latents / 0.13025
        
        with torch.no_grad():
            if hasattr(vae, 'decode'):
                dec = vae.decode(latents.to(self.dtype))
                images = dec.sample if hasattr(dec, 'sample') else dec
            else:
                images = vae(latents.to(self.dtype))
        
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    # ── DDPM noise schedule ──

    def get_noise(self, latent_shape, generator=None):
        noise = torch.randn(latent_shape, device=self.device, dtype=self.dtype,
                            generator=generator)
        if self.offset_noise > 0:
            noise += self.offset_noise * torch.randn(
                latent_shape[0], latent_shape[1], 1, 1,
                device=self.device, dtype=self.dtype, generator=generator
            )
        return noise

    def get_timesteps(self, batch_size, generator=None):
        """Sample random discrete timesteps in [0, num_train_timesteps)."""
        return torch.randint(
            0, self.num_train_timesteps, (batch_size,),
            device=self.device, generator=generator
        ).long()

    def add_noise(self, clean_latents, noise, timesteps):
        """
        DDPM forward: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps
        """
        sqrt_alpha = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5
        
        while sqrt_alpha.dim() < clean_latents.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * clean_latents + sqrt_one_minus_alpha * noise

    def compute_target(self, clean_latents, noise, timesteps):
        """Target depends on prediction_type."""
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "v_prediction":
            sqrt_alpha = self.alphas_cumprod[timesteps] ** 0.5
            sqrt_one_minus_alpha = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5
            while sqrt_alpha.dim() < clean_latents.dim():
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            return sqrt_alpha * noise - sqrt_one_minus_alpha * clean_latents
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    # ── Denoiser forward ──

    def forward_denoise(self, noisy_latents, timesteps, condition, **kwargs):
        """
        Run SDXL UNet.
        
        Integration consideration: SDXL UNet expects:
          - sample: noisy latents
          - timestep: discrete timesteps
          - encoder_hidden_states: text embeddings (concat of both encoders)
          - added_cond_kwargs: {'text_embeds': pooled_embeds, 'time_ids': time_ids}
          
        time_ids encodes: [original_height, original_width, crop_top, crop_left, 
                           target_height, target_width]
        """
        unet = self.components.denoiser
        if unet is None:
            raise RuntimeError("UNet not loaded")
        
        prompt_embeds = condition.get('prompt_embeds')
        pooled = condition.get('pooled_prompt_embeds')
        
        # Build SDXL micro-conditioning (time_ids)
        # Integration consideration: these should come from the batch/dataset
        # For now, use defaults matching the target resolution
        batch_size = noisy_latents.shape[0]
        add_time_ids = kwargs.get('add_time_ids')
        if add_time_ids is None:
            # Default: assume 1024x1024 original, no crop, 1024x1024 target
            add_time_ids = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]] * batch_size,
                device=self.device, dtype=self.dtype
            )
        
        added_cond_kwargs = {}
        if pooled is not None:
            added_cond_kwargs['text_embeds'] = pooled
        added_cond_kwargs['time_ids'] = add_time_ids
        
        try:
            out = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
                return_dict=False,
            )
            return out[0] if isinstance(out, tuple) else out
        except TypeError:
            out = unet(noisy_latents, timesteps, prompt_embeds)
            return out[0] if isinstance(out, tuple) else out

    # ── Loss ──

    def compute_loss(self, prediction, target, timesteps, **kwargs):
        loss = F.mse_loss(prediction, target, reduction='none')
        loss = loss.mean(dim=list(range(1, loss.dim())))
        
        if self.snr_gamma is not None:
            snr = self.alphas_cumprod[timesteps] / (1.0 - self.alphas_cumprod[timesteps])
            weight = torch.clamp(snr, max=self.snr_gamma) / snr
            if self.prediction_type == "v_prediction":
                weight = weight + 1.0
            loss = loss * weight
        
        return loss.mean()

    # ── Inference ──

    @torch.no_grad()
    def sample(self, request: SampleRequest) -> SampleResult:
        num_steps = request.num_steps or self.default_num_steps
        guidance = request.guidance_scale if request.guidance_scale is not None else self.default_guidance_scale
        sampler_name = request.sampler or self.default_sampler
        
        sampler = create_sampler(sampler_name, num_steps=num_steps,
                                 num_train_timesteps=self.num_train_timesteps)
        
        # Integration consideration: create_sampler needs to support
        # discrete schedulers (EulerDiscreteScheduler, DPMSolverMultistep, etc.)
        # The sampler should return discrete timestep schedules
        
        all_images = []
        all_seeds = []
        
        for i, prompt in enumerate(request.prompts):
            seed = (request.seed + i) if request.seed is not None else torch.randint(0, 2**32, (1,)).item()
            gen = torch.Generator(device=self.device).manual_seed(seed)
            
            # Encode prompt (with negative for CFG)
            neg = (request.negative_prompts[i] 
                   if request.negative_prompts and i < len(request.negative_prompts)
                   else "")
            cond = self.encode_prompt(prompt, neg)
            
            latent_shape = self.get_latent_shape(request.height, request.width, batch_size=1)
            latents = self.get_noise(latent_shape, generator=gen)
            
            # Integration consideration: for SDXL the initial latents
            # are scaled by the scheduler's init_noise_sigma
            
            schedule = sampler.get_schedule(device=self.device)
            
            for step_idx in range(num_steps):
                t = schedule[step_idx]
                
                # CFG: run model twice (conditional + unconditional)
                if guidance > 1.0 and 'negative_prompt_embeds' in cond:
                    latent_input = torch.cat([latents, latents])
                    t_input = torch.tensor([t, t], device=self.device)
                    embeds = torch.cat([
                        cond['negative_prompt_embeds'], cond['prompt_embeds']
                    ])
                    cond_input = {'prompt_embeds': embeds}
                    if 'pooled_prompt_embeds' in cond:
                        # Integration consideration: need negative pooled embeds too
                        cond_input['pooled_prompt_embeds'] = cond.get('pooled_prompt_embeds')
                    
                    pred = self.forward_denoise(latent_input, t_input, cond_input)
                    pred_uncond, pred_cond = pred.chunk(2)
                    pred = pred_uncond + guidance * (pred_cond - pred_uncond)
                else:
                    t_tensor = torch.tensor([t], device=self.device)
                    pred = self.forward_denoise(latents, t_tensor, cond)
                
                # Sampler step
                t_next = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else 0
                latents = sampler.step(pred, t, t_next, latents)
            
            pixels = self.decode_latents(latents)
            img = pixels[0].cpu().float().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype('uint8')
            all_images.append(Image.fromarray(img))
            all_seeds.append(seed)
        
        return SampleResult(
            images=all_images, seeds=all_seeds,
            prompts=request.prompts, step=0, epoch=0,
        )

    def get_latent_shape(self, height, width, batch_size=1):
        return (batch_size, self.vae_latent_channels,
                height // self.vae_scale_factor, width // self.vae_scale_factor)


# ── LoRA presets for SDXL ──

SDXL_LORA_PRESETS = {
    "character_identity": {
        "description": "Target cross-attention in mid and up blocks for identity learning.",
        "target_patterns": [
            r"mid_block.*attn.*\.(to_q|to_k|to_v|to_out)",
            r"up_blocks\.[1-3].*attn.*\.(to_q|to_k|to_v|to_out)",
            r"down_blocks\.[1-3].*attn.*\.(to_q|to_k|to_v|to_out)",
        ],
        "rank": 16,
        "alpha": 16,
        "recommended_lr": 1e-4,
    },
    "style_transfer": {
        "description": "Broader targeting including ResNet blocks for style.",
        "target_patterns": [
            r"attn.*\.(to_q|to_k|to_v|to_out)",
            r"mid_block.*resnets.*conv",
        ],
        "rank": 32,
        "alpha": 32,
        "recommended_lr": 5e-5,
    },
}


register_pipeline("sdxl", SDXLPipeline)
