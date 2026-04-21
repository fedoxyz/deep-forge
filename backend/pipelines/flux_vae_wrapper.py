"""
Flux VAE wrapper for Z-Image Turbo ae.safetensors.

Architecture verified against actual ae.safetensors weight shapes via test_ae.py.
Keys match exactly — strict=True load, no remapping needed.

Verified facts:
  - Encoder: 2 resnets per down block, nin_shortcut (not conv_shortcut)
  - Decoder: 3 resnets per up block
  - Decoder up.0 has NO upsample, up.1/2/3 have upsample
  - Decoder forward runs reversed: up.3 → up.2 → up.1 → up.0
  - No quant_conv / post_quant_conv
  - scaling_factor=0.3611, shift_factor=0.1159
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def swish(x):
    return x * torch.sigmoid(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=32, eps=1e-6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        h = swish(self.norm1(x))
        h = self.conv1(h)
        h = swish(self.norm2(h))
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """ldm-style attention with Conv2d q/k/v — matches attn_1.q/k/v/proj_out keys."""
    def __init__(self, in_channels, groups=32, eps=1e-6):
        super().__init__()
        self.norm     = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.q        = nn.Conv2d(in_channels, in_channels, 1)
        self.k        = nn.Conv2d(in_channels, in_channels, 1)
        self.v        = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        w = F.softmax(torch.bmm(q, k) * (C ** -0.5), dim=2)
        v = v.reshape(B, C, H * W)
        h = torch.bmm(v, w.permute(0, 2, 1)).reshape(B, C, H, W)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), value=0))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


class MidBlock(nn.Module):
    """Matches encoder.mid / decoder.mid: block_1, attn_1, block_2."""
    def __init__(self, channels, groups=32):
        super().__init__()
        self.block_1 = ResnetBlock2D(channels, channels, groups=groups)
        self.attn_1  = AttnBlock(channels, groups=groups)
        self.block_2 = ResnetBlock2D(channels, channels, groups=groups)

    def forward(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class DownBlock(nn.Module):
    """Matches encoder.down.{i}: block.0, block.1, downsample.conv."""
    def __init__(self, in_ch, out_ch, n_layers, has_downsample, groups=32):
        super().__init__()
        self.block = nn.ModuleList([
            ResnetBlock2D(in_ch if j == 0 else out_ch, out_ch, groups=groups)
            for j in range(n_layers)
        ])
        self.downsample = Downsample(out_ch) if has_downsample else None

    def forward(self, x):
        for b in self.block:
            x = b(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Matches decoder.up.{i}: block.0, block.1, block.2, upsample.conv."""
    def __init__(self, in_ch, out_ch, n_layers, has_upsample, groups=32):
        super().__init__()
        self.block = nn.ModuleList([
            ResnetBlock2D(in_ch if j == 0 else out_ch, out_ch, groups=groups)
            for j in range(n_layers)
        ])
        self.upsample = Upsample(out_ch) if has_upsample else None

    def forward(self, x):
        for b in self.block:
            x = b(x)
        if self.upsample:
            x = self.upsample(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder / AE
# ─────────────────────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self, in_channels=3, z_channels=16,
                 block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.down = nn.ModuleList()
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            self.down.append(DownBlock(
                in_ch, out_ch,
                n_layers=layers_per_block,
                has_downsample=(i < len(block_out_channels) - 1),
                groups=groups,
            ))
            in_ch = out_ch
        self.mid      = MidBlock(in_ch, groups=groups)
        self.norm_out = nn.GroupNorm(groups, in_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.down:
            x = b(x)
        x = self.mid(x)
        x = swish(self.norm_out(x))
        return self.conv_out(x)


class _Decoder(nn.Module):
    def __init__(self, out_channels=3, z_channels=16,
                 block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, groups=32):
        super().__init__()
        top_ch = block_out_channels[-1]
        self.conv_in = nn.Conv2d(z_channels, top_ch, 3, padding=1)
        self.mid = MidBlock(top_ch, groups=groups)

        n = len(block_out_channels)
        self.up = nn.ModuleList()
        for i in range(n):
            out_ch = block_out_channels[i]
            in_ch  = block_out_channels[i + 1] if i < n - 1 else block_out_channels[-1]
            self.up.append(UpBlock(
                in_ch, out_ch,
                n_layers=layers_per_block + 1,  # 3 resnets per decoder block
                has_upsample=(i > 0),            # up.0 has no upsample
                groups=groups,
            ))

        self.norm_out = nn.GroupNorm(groups, block_out_channels[0], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        for block in reversed(self.up):   # up.3 → up.2 → up.1 → up.0
            x = block(x)
        x = swish(self.norm_out(x))
        return self.conv_out(x)


class _AE(nn.Module):
    SCALING_FACTOR = 0.3611
    SHIFT_FACTOR   = 0.1159

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)

        shift = self.SHIFT_FACTOR
        scale = self.SCALING_FACTOR

        class _Dist:
            def __init__(self, m, s):
                self._mean = (m - shift) * scale
                self._std  = s * scale
            def sample(self):
                return self._mean + self._std * torch.randn_like(self._mean)

        class _Out:
            def __init__(self, m, s):
                self.latent_dist = _Dist(m, s)

        return _Out(mean, std)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        z = z / self.SCALING_FACTOR + self.SHIFT_FACTOR
        dec = self.decoder(z)

        if return_dict:
            class _Out:
                def __init__(self, s): self.sample = s
            return _Out(dec)
        return (dec,)

    def forward(self, x: torch.Tensor):
        enc = self.encode(x)
        z = enc.latent_dist.sample()
        return self.decode(z)


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper — drop-in for ZImageTurboPipeline
# ─────────────────────────────────────────────────────────────────────────────

class FluxVAEWrapper(nn.Module):
    """
    Wraps _AE to match the interface ZImageTurboPipeline expects:
      - .encode(x)  → obj with .latent_dist.sample()
      - .decode(z)  → obj with .sample
      - .config.scaling_factor
    """

    class _Config:
        scaling_factor = _AE.SCALING_FACTOR
        shift_factor   = _AE.SHIFT_FACTOR

    config = _Config()

    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        super().__init__()
        self._dtype  = dtype
        self._device = device or torch.device('cpu')
        self._vae    = self._load(path)

    def _load(self, path: str) -> _AE:
        vae = _AE()
        print(f"[FluxVAE] Loading weights from '{path}'")
        sd = load_file(path)
        vae.load_state_dict(sd, strict=True)   # strict — keys verified
        print(f"[FluxVAE] Loaded — {sum(p.numel() for p in vae.parameters()):,} params")
        vae = vae.to(self._device, self._dtype).eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    def encode(self, x: torch.Tensor):
        actual_device = next(self._vae.parameters()).device
        return self._vae.encode(x.to(actual_device, self._dtype))
    
    def decode(self, z: torch.Tensor):
        actual_device = next(self._vae.parameters()).device
        return self._vae.decode(z.to(actual_device, self._dtype))

    def forward(self, x: torch.Tensor):
        return self._vae(x)
