"""
Layer streaming: automatically offloads individual layers to/from VRAM
during forward pass. Zero changes needed to model code.

How it works:
  - Register pre/post forward hooks on every leaf-ish module
  - pre_hook:  module.to(cuda)
  - post_hook: module.to(cpu), torch.cuda.empty_cache() (optional)

Peak VRAM = largest single layer + activations flowing through it.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from contextlib import contextmanager


class LayerStreamManager:
    """
    Streams model layers through VRAM one at a time.
    
    Usage:
        mgr = LayerStreamManager(model, device='cuda', min_param_bytes=10_000_000)
        with mgr.stream():
            output = model(input)   # each layer auto-moves to GPU then back
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        offload_device: torch.device = None,
        # Only stream modules larger than this (skip tiny bias layers etc)
        min_param_bytes: int = 10_000_000,  # 10MB default
        # Whether to empty_cache after each layer (slower but lower peak)
        aggressive_offload: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.device = device or torch.device('cuda')
        self.offload_device = offload_device or torch.device('cpu')
        self.min_param_bytes = min_param_bytes
        self.aggressive_offload = aggressive_offload
        self.dtype = dtype
        self._hooks: List = []
        self._streaming = False

    def _module_param_bytes(self, module: nn.Module) -> int:
        return sum(p.numel() * p.element_size() 
                   for p in module.parameters(recurse=False))

    def _should_stream(self, module: nn.Module) -> bool:
        """Stream modules that own significant parameters."""
        return self._module_param_bytes(module) >= self.min_param_bytes

    def _pre_hook(self, module: nn.Module, args):
        """Move module to GPU just before its forward."""
        if not self._streaming:
            return
        module.to(self.device)
        # Cast if needed
        if self.dtype is not None:
            module.to(self.dtype)
        return args

    def _post_hook(self, module: nn.Module, args, output):
        """Move module back to CPU right after its forward."""
        if not self._streaming:
            return
        module.to(self.offload_device)
        if self.aggressive_offload:
            torch.cuda.empty_cache()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self._should_stream(module):
                h1 = module.register_forward_pre_hook(
                    lambda m, a: self._pre_hook(m, a)
                )
                h2 = module.register_forward_hook(
                    lambda m, a, o: self._post_hook(m, a, o)
                )
                self._hooks.extend([h1, h2])

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @contextmanager
    def stream(self):
        """Context manager: enables layer streaming for this forward pass."""
        self._register_hooks()
        self._streaming = True
        try:
            yield
        finally:
            self._streaming = False
            self._remove_hooks()
            torch.cuda.empty_cache()

    def estimate_peak_vram(self) -> Dict[str, float]:
        """Estimate peak VRAM needed (largest single layer)."""
        largest_name = ""
        largest_bytes = 0
        for name, module in self.model.named_modules():
            if not self._should_stream(module):
                continue
            b = sum(p.numel() * p.element_size() 
                    for p in module.parameters())
            if b > largest_bytes:
                largest_bytes = b
                largest_name = name
        return {
            'largest_layer': largest_name,
            'largest_layer_gb': largest_bytes / 1024**3,
            'estimated_peak_gb': largest_bytes * 3 / 1024**3,  # weights + grad + optimizer
        }

class TrainingLayerStreamer:
    """
    For training: stream weights CPU→GPU→CPU but keep activations on GPU
    so backward pass works correctly.
    
    The key difference from inference streaming:
    - Weights move: CPU → GPU (pre) → CPU (post)  
    - Activations stay: on GPU (needed for backward)
    - On backward, weights are reloaded for grad computation (like grad checkpointing)
    
    This is what ai-toolkit does internally.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        offload_device: torch.device = None,
        min_param_bytes: int = 50_000_000,  # 50MB — only stream big layers
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.device = device or torch.device('cuda')
        self.offload_device = offload_device or torch.device('cpu')
        self.min_param_bytes = min_param_bytes
        self.dtype = dtype
        self._hooks: List = []

    def _should_stream(self, module: nn.Module) -> bool:
        own_bytes = sum(p.numel() * p.element_size()
                        for p in module.parameters(recurse=False))
        return own_bytes >= self.min_param_bytes

    def _pre_hook(self, module, args):
        """Load weights to GPU before forward."""
        module.to(self.device)
        if self.dtype:
            module.to(self.dtype)

    def _post_hook(self, module, args, output):
        """
        After forward: offload weights but NOT activations.
        Only offload params that require_grad=False (frozen base weights).
        LoRA params (requires_grad=True) stay on GPU.
        """
        for param in module.parameters(recurse=True):   # ← recurse=True to match pre_hook
            if not param.requires_grad:                  # ← only offload frozen params
                param.data = param.data.to(self.offload_device)

    def enable(self):
        """Register hooks — call once before training loop."""
        for name, module in self.model.named_modules():
            if self._should_stream(module):
                h1 = module.register_forward_pre_hook(
                    lambda m, a: self._pre_hook(m, a)
                )
                h2 = module.register_forward_hook(
                    lambda m, a, o: self._post_hook(m, a, o)
                )
                self._hooks.extend([h1, h2])
        print(f"[LayerStream] Registered {len(self._hooks)//2} streaming layers")

    def disable(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

class VRAMMonitor:
    """Lightweight VRAM tracker — attach to trainer for live monitoring."""
    
    def __init__(self, device='cuda', log_every_n_steps=10):
        self.device = device
        self.log_every = log_every_n_steps
        self.peak_allocated = 0
        self._step = 0
    
    def step(self, tag=''):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        self.peak_allocated = max(self.peak_allocated, alloc)
        self._step += 1
        if self._step % self.log_every == 0:
            print(f"[VRAM] {tag} alloc={alloc:.2f}GB "
                  f"reserved={reserv:.2f}GB "
                  f"peak={self.peak_allocated:.2f}GB")
