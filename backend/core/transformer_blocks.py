"""
Universal transformer and vision encoder building blocks for deep-forge.
These primitives are task-agnostic and can be composed for segmentation,
detection, classification, generation, or any other vision/multimodal task.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ── 2D RoPE helpers ──

def build_2d_rope_cache(seq_len: int, head_dim: int, grid_h: int, grid_w: int,
                         device=None, dtype=None):
    assert grid_h * grid_w == seq_len, f"grid {grid_h}x{grid_w} != seq_len {seq_len}"
    half = head_dim // 2
    quarter = half // 2

    rows = torch.arange(grid_h, device=device, dtype=torch.float32)
    cols = torch.arange(grid_w, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, quarter, device=device, dtype=torch.float32) / quarter))

    row_emb = torch.outer(rows, inv_freq).unsqueeze(1).expand(-1, grid_w, -1).reshape(seq_len, quarter)
    col_emb = torch.outer(cols, inv_freq).unsqueeze(0).expand(grid_h, -1, -1).reshape(seq_len, quarter)

    emb = torch.cat([row_emb, col_emb], dim=-1)  # (N, half)
    sin = torch.cat([emb.sin(), emb.sin()], dim=-1)  # (N, head_dim)
    cos = torch.cat([emb.cos(), emb.cos()], dim=-1)
    return sin, cos


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, heads, N, head_dim). sin, cos: (N, head_dim)."""
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ── Drop Path (stochastic depth) ──

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype).floor_().div_(keep)
        return x * mask


# ── SwiGLU FFN ──

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── SDPA Self-Attention ──

class SDPASelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, drop_path: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self._grid: Optional[Tuple[int, int]] = None
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _get_rope(self, N, device, dtype):
        if self._grid is None:
            return None, None
        gh, gw = self._grid
        if (self._rope_cache is None or
                self._rope_cache[0].shape[0] != N or
                self._rope_cache[0].device != device):
            sin, cos = build_2d_rope_cache(N, self.head_dim, gh, gw, device=device, dtype=dtype)
            self._rope_cache = (sin, cos)
        return self._rope_cache

    def forward(self, x):
        B, N, C = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        sin, cos = self._get_rope(N, x.device, x.dtype)
        if sin is not None:
            q, k = apply_rope(q, k, sin, cos)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return residual + self.drop_path(self.proj(out))


# ── Cross-Attention ──

class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q_tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_tokens.shape
        residual = q_tokens
        q  = self.q_proj(self.norm_q(q_tokens))
        kv = self.kv_proj(self.norm_kv(ctx))
        k, v = kv.chunk(2, dim=-1)

        def split_heads(t):
            return t.reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return residual + self.out_proj(out)


# ── RoPE2D standalone palette primitive ──

class RoPE2D(nn.Module):
    """Stateless palette primitive. RoPE is applied inside SDPASelfAttention;
    this class exists so the primitive appears as a wirable node in the builder."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self._grid = None

    def forward(self, x):
        return x  # no-op when used standalone


# ── Patch Embed ──

class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)
        self.grid_h = None
        self.grid_w = None

    def forward(self, x):
        B, C, H, W = x.shape
        self.grid_h = H // self.patch_size
        self.grid_w = W // self.patch_size
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        return self.norm(x)


# ── RoPE Encoder Block ──

class RoPEEncoderBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8,
                 mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.attn = SDPASelfAttention(dim, heads, drop_path)
        self.ffn  = SwiGLUFFN(dim, mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.last_output = None  # tapped by FPNNeck or any other neck

    def set_grid(self, gh: int, gw: int):
        self.attn._grid = (gh, gw)
        self.attn._rope_cache = None

    def forward(self, x):
        x = self.attn(x)
        residual = x
        x = residual + self.drop_path(self.ffn(self.norm(x)))
        self.last_output = x
        return x


# ── Decoder Block ──

class DecoderBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.self_attn  = SDPASelfAttention(dim, heads)
        self.cross_cond = CrossAttention(dim, heads)
        self.cross_img  = CrossAttention(dim, heads)
        self.ffn  = SwiGLUFFN(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, cond_tokens, img_tokens):
        queries = self.self_attn(queries)
        queries = self.cross_cond(queries, cond_tokens)
        queries = self.cross_img(queries, img_tokens)
        return queries + self.ffn(self.norm(queries))


# ── Encoder with intermediate taps ──

class EncoderWithTaps(nn.Module):
    """Wraps a sequence of encoder blocks and collects outputs at specified indices.
    Tap indices are consumed by any neck (FPN, simple linear, etc.)."""
    def __init__(self, blocks: List[RoPEEncoderBlock], tap_indices: List[int]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.tap_indices = tap_indices

    def forward(self, x):
        taps = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.tap_indices:
                taps.append(x)
        return x, taps
