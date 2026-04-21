"""
Inference samplers for diffusion/flow models.

All samplers work on the same interface:
  - __init__(num_steps, ...)
  - get_schedule() -> list of timesteps
  - step(model_output, timestep, sample, ...) -> denoised sample
"""

import math
from typing import List, Optional, Tuple

import torch


class FlowMatchingEulerSampler:
    """
    Euler sampler for flow matching models (Z-Image, Flux, SD3).

    Flow matching uses linear interpolation:
      x_t = (1 - t) * x_0 + t * noise
    The model predicts velocity v = x_1 - x_0 (or equivalently dx/dt).

    Euler step: x_{t-dt} = x_t - dt * v_predicted
    """

    def __init__(self, num_steps: int = 8, shift: float = 1.0):
        self.num_steps = num_steps
        self.shift = shift

    def get_schedule(self, device: torch.device = None) -> torch.Tensor:
        """Returns timesteps from 1.0 → 0.0 (num_steps + 1 values including endpoints)."""
        # Linearly spaced from 1 to 0
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        # Apply shift (time-shift from Flux/SD3 style)
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        return timesteps

    def step(self, model_output: torch.Tensor, t_current: float,
             t_next: float, sample: torch.Tensor) -> torch.Tensor:
        """Single Euler step."""
        dt = t_next - t_current  # negative (going from 1 → 0)
        return sample + dt * model_output


class FlowMatchingEulerAncestralSampler:
    """Euler Ancestral for flow matching — adds noise at each step for variety."""

    def __init__(self, num_steps: int = 8, shift: float = 1.0, eta: float = 1.0):
        self.num_steps = num_steps
        self.shift = shift
        self.eta = eta

    def get_schedule(self, device=None) -> torch.Tensor:
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        return timesteps

    def step(self, model_output: torch.Tensor, t_current: float,
             t_next: float, sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        dt = t_next - t_current
        # Deterministic part
        denoised = sample + dt * model_output

        # Stochastic part (only if not at final step)
        if t_next > 0 and self.eta > 0:
            sigma = self.eta * abs(dt) * math.sqrt(t_next)
            noise = torch.randn_like(denoised, generator=generator)
            denoised = denoised + sigma * noise

        return denoised


class DPMPPSampler:
    """
    DPM++ 2M style sampler adapted for flow matching.
    Uses 2nd order multistep for improved quality.
    """

    def __init__(self, num_steps: int = 8, shift: float = 1.0):
        self.num_steps = num_steps
        self.shift = shift
        self._prev_output = None

    def get_schedule(self, device=None) -> torch.Tensor:
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        return timesteps

    def step(self, model_output: torch.Tensor, t_current: float,
             t_next: float, sample: torch.Tensor) -> torch.Tensor:
        dt = t_next - t_current

        if self._prev_output is None:
            # First step: plain Euler
            result = sample + dt * model_output
        else:
            # 2nd order: use previous output for correction
            result = sample + dt * (1.5 * model_output - 0.5 * self._prev_output)

        self._prev_output = model_output.clone()
        return result

    def reset(self):
        self._prev_output = None


# ── Registry ──

SAMPLER_REGISTRY = {
    'euler': FlowMatchingEulerSampler,
    'euler_a': FlowMatchingEulerAncestralSampler,
    'dpm++': DPMPPSampler,
}


def create_sampler(name: str, num_steps: int = 8, **kwargs):
    """Create a sampler by name."""
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[name](num_steps=num_steps, **kwargs)
