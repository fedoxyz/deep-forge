"""
Task-specific output heads for deep-forge.
These are output-stage modules that sit on top of a generic encoder+neck.
Currently: FPN neck, dot-product mask head, conditioning slot.
New heads (detection, classification, generation) can be added here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── FPN Neck ──

class FPNNeck(nn.Module):
    """Top-down multi-scale feature fusion over encoder tap tokens.
    Task-agnostic: any downstream head (mask, bbox, class) can consume its output."""
    def __init__(self, in_dims: str = '256,256,256',
                 fpn_dim: int = 256, tap_indices: str = '2,5,7'):
        super().__init__()
        dims = [int(d) for d in in_dims.split(',')]
        self.tap_indices = [int(i) for i in tap_indices.split(',')]
        self.laterals = nn.ModuleList([nn.Linear(d, fpn_dim) for d in dims])
        self.fpn_dim = fpn_dim
        self._grid = None  # (gh, gw) set at build time

    def forward(self, taps: List[torch.Tensor]):
        projected = [lat(t) for lat, t in zip(self.laterals, taps)]
        fused = projected[-1]
        for feat in reversed(projected[:-1]):
            fused = fused + feat
        return fused  # (B, N, fpn_dim)


# ── Conditioning Slot ──

class ConditioningSlot(nn.Module):
    """Prompt interface node. V1: learned null token.
    Future modes: text embedding, image-ref crop, bounding box prior."""
    def __init__(self, dim: int = 256, mode: str = 'null'):
        super().__init__()
        self.mode = mode
        self.null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x=None):
        B = x.shape[0] if x is not None else 1
        return self.null_token.expand(B, -1, -1)


# ── Mask Head ──

class MaskHead(nn.Module):
    """Dot-product mask generation from query tokens against FPN features,
    followed by bilinear upsampling to full resolution.
    Suited for instance/semantic/binary segmentation."""
    def __init__(self, dim: int = 256, fpn_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(dim, fpn_dim) if dim != fpn_dim else nn.Identity()
        self._patch_size = 16  # set by the assembling model

    def forward(self, queries, fpn_tokens, grid_h, grid_w):
        q = self.proj(queries)                                      # (B, 1, fpn_dim)
        logits = torch.bmm(q, fpn_tokens.transpose(1, 2))          # (B, 1, N)
        logits = logits.reshape(logits.shape[0], 1, grid_h, grid_w)
        logits = F.interpolate(logits, scale_factor=self._patch_size,
                               mode='bilinear', align_corners=False)
        return {'mask_logits': logits, 'masks': torch.sigmoid(logits)}
