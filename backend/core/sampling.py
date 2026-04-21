"""
Sampling system for generating preview images during training.

Integrates with UnifiedTrainer via:
  1. Automatic sampling every N steps (config-driven)
  2. Manual sampling triggered by API button
  3. Saves images to run_dir/samples/

The sampler temporarily switches the model to eval mode,
generates images, saves them, then restores train mode.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from backend.pipelines.base_pipeline import BaseDiffusionPipeline, SampleRequest, SampleResult


class TrainingSampler:
    """
    Manages sample generation during training.

    Usage:
        sampler = TrainingSampler(pipeline, config, run_dir)
        # In training loop:
        if sampler.should_sample(step):
            results = sampler.generate(step, epoch)
    """

    def __init__(
        self,
        pipeline: BaseDiffusionPipeline,
        config: Dict[str, Any],
        run_dir: str,
    ):
        self.pipeline = pipeline
        self.run_dir = run_dir
        self.samples_dir = os.path.join(run_dir, 'samples')
        os.makedirs(self.samples_dir, exist_ok=True)

        sc = config.get('sampling', {})
        self.enabled = sc.get('enabled', False)
        self.every_n_steps = sc.get('every_n_steps', 0)
        self.every_n_epochs = sc.get('every_n_epochs', 0)

        # Default prompts for sampling
        self.prompts = sc.get('prompts', [])
        self.negative_prompts = sc.get('negative_prompts', [])

        # Sampling parameters
        self.width = sc.get('width', 1024)
        self.height = sc.get('height', 1024)
        self.num_steps = sc.get('num_steps', pipeline.default_num_steps)
        self.guidance_scale = sc.get('guidance_scale', pipeline.default_guidance_scale)
        self.sampler_name = sc.get('sampler', pipeline.default_sampler)
        self.seed = sc.get('seed', 42)

        # Track what we've generated
        self.sample_log: List[Dict] = []
        self._last_sample_step = -1

    def should_sample(self, step: int, epoch: int = 0, force: bool = False) -> bool:
        """Check if we should sample at this step."""
        if force:
            return True
        if not self.enabled or not self.prompts:
            return False
        if self.every_n_steps > 0 and step > 0 and step % self.every_n_steps == 0:
            if step != self._last_sample_step:
                return True
        return False

    def should_sample_epoch(self, epoch: int) -> bool:
        """Check if we should sample at end of this epoch."""
        if not self.enabled or not self.prompts:
            return False
        return self.every_n_epochs > 0 and (epoch + 1) % self.every_n_epochs == 0

    @torch.no_grad()
    def generate(self, step: int, epoch: int,
                 prompts: Optional[List[str]] = None,
                 request: Optional[SampleRequest] = None) -> SampleResult:
        """
        Generate samples. Caller should ensure model is on correct device.

        Args:
            step: Current training step
            epoch: Current epoch
            prompts: Override default prompts
            request: Full override SampleRequest
        """
        t0 = time.time()

        if request is None:
            request = SampleRequest(
                prompts=prompts or self.prompts,
                negative_prompts=self.negative_prompts or None,
                width=self.width,
                height=self.height,
                num_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
                sampler=self.sampler_name,
            )

        result = self.pipeline.sample(request)
        result.step = step
        result.epoch = epoch

        # Save images
        step_dir = os.path.join(self.samples_dir, f'step_{step:06d}')
        os.makedirs(step_dir, exist_ok=True)

        image_paths = []
        for i, (img, seed, prompt) in enumerate(
            zip(result.images, result.seeds, result.prompts)
        ):
            fname = f'sample_{i:02d}_seed{seed}.png'
            fpath = os.path.join(step_dir, fname)
            if isinstance(img, Image.Image):
                img.save(fpath)
            else:
                # Tensor → PIL
                if img.dim() == 3:
                    arr = img.cpu().float().permute(1, 2, 0).numpy()
                    arr = (arr * 255).clip(0, 255).astype('uint8')
                    Image.fromarray(arr).save(fpath)

            image_paths.append(fpath)

        dt = time.time() - t0

        # Log entry
        entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'prompts': request.prompts,
            'seeds': result.seeds,
            'sampler': request.sampler,
            'num_steps': request.num_steps,
            'guidance_scale': request.guidance_scale,
            'image_paths': image_paths,
            'generation_time': dt,
        }
        self.sample_log.append(entry)
        self._last_sample_step = step

        # Save log
        log_path = os.path.join(self.samples_dir, 'sample_log.json')
        try:
            with open(log_path, 'w') as f:
                json.dump(self.sample_log, f, indent=2, default=str)
        except Exception:
            pass

        print(f"[Sampler] Generated {len(result.images)} samples at step {step} "
              f"in {dt:.1f}s → {step_dir}")

        return result

    def get_latest_samples(self) -> Optional[Dict]:
        """Return info about the most recent samples."""
        if not self.sample_log:
            return None
        return self.sample_log[-1]

    def get_all_sample_paths(self) -> List[str]:
        """Get all generated image paths."""
        paths = []
        for entry in self.sample_log:
            paths.extend(entry.get('image_paths', []))
        return paths
