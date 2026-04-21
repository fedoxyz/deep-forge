"""
VAE debug script — run inside container to test ae.safetensors loading.
Usage: python main.py [--path /data/models/ae.safetensors] [--device cpu]

Tests:
  1. Weight loading (zero missing/unexpected keys)
  2. Encode: random image → latents
  3. Decode: latents → image
  4. Round-trip: encode then decode, check shape
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import re

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — matches ae.safetensors keys exactly
# ─────────────────────────────────────────────────────────────────────────────

def swish(x):
    return x * torch.sigmoid(x)

# ResnetBlock2D — rename conv_shortcut back to nin_shortcut
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=32, eps=1e-6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = (             # ← was conv_shortcut
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        h = swish(self.norm1(x))
        h = self.conv1(h)
        h = swish(self.norm2(h))
        h = self.conv2(h)
        if self.nin_shortcut is not None:  # ← was conv_shortcut
            x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    """Original ldm-style attention with conv q/k/v — matches ae.safetensors keys."""
    def __init__(self, in_channels, groups=32, eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.q        = nn.Conv2d(in_channels, in_channels, 1)
        self.k        = nn.Conv2d(in_channels, in_channels, 1)
        self.v        = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H*W).permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(B, C, H*W)
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
    """encoder.mid / decoder.mid — block_1, attn_1, block_2"""
    def __init__(self, channels, groups=32):
        super().__init__()
        self.block_1 = ResnetBlock2D(channels, channels, groups=groups)
        self.attn_1  = AttnBlock(channels, groups=groups)
        self.block_2 = ResnetBlock2D(channels, channels, groups=groups)

    def forward(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class DownBlock(nn.Module):
    """encoder.down.{i} — matches keys: block.0, block.1, downsample.conv"""
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
    """decoder.up.{i} — matches keys: block.0, block.1, block.2, upsample.conv"""
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


class Encoder(nn.Module):
    """
    Matches encoder.* keys in ae.safetensors exactly.
    block_out_channels = [128, 256, 512, 512], 2 resnets per down block.
    """
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

        self.mid = MidBlock(in_ch, groups=groups)
        self.norm_out = nn.GroupNorm(groups, in_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.down:
            x = b(x)
        x = self.mid(x)
        x = swish(self.norm_out(x))
        return self.conv_out(x)


class Decoder(nn.Module):
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
            has_upsample = (i > 0)
            self.up.append(UpBlock(
                in_ch, out_ch,
                n_layers=layers_per_block + 1,
                has_upsample=has_upsample,
                groups=groups,
            ))

        # ← these were missing
        self.norm_out = nn.GroupNorm(groups, block_out_channels[0], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        for block in reversed(self.up):
            x = block(x)
        x = swish(self.norm_out(x))
        return self.conv_out(x)

class AE(nn.Module):
    """
    Full autoencoder. Keys match ae.safetensors exactly — no remapping needed.
    scaling_factor and shift_factor from Flux VAE config.
    """
    SCALING_FACTOR = 0.3611
    SHIFT_FACTOR   = 0.1159

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        # Apply scaling
        z = (z - self.SHIFT_FACTOR) * self.SCALING_FACTOR
        return z

    def decode(self, z):
        # Reverse scaling
        z = z / self.SCALING_FACTOR + self.SHIFT_FACTOR
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# Debug runner
# ─────────────────────────────────────────────────────────────────────────────

def load_vae(path: str, device: str) -> AE:
    print(f"\n{'='*60}")
    print(f"Loading ae.safetensors from: {path}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    vae = AE()
    total = sum(p.numel() for p in vae.parameters())
    print(f"Model parameters: {total:,}")

    sd = load_file(path)
    print(f"Safetensors keys: {len(sd)}")
    print(f"Model keys:       {len(vae.state_dict())}")

    # Check for key mismatches before loading
    model_keys = set(vae.state_dict().keys())
    sd_keys    = set(sd.keys())

    missing    = model_keys - sd_keys
    unexpected = sd_keys - model_keys

    if missing:
        print(f"\n❌ MISSING keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"   {k}")
    if unexpected:
        print(f"\n❌ UNEXPECTED keys ({len(unexpected)}):")
        for k in sorted(unexpected)[:10]:
            print(f"   {k}")
    if not missing and not unexpected:
        print(f"\n✅ All keys match perfectly!")

    result = vae.load_state_dict(sd, strict=True)
    print(f"\n✅ load_state_dict done")

    vae = vae.to(device).eval()
    return vae


def test_encode(vae: AE, device: str, H=64, W=64):
    print(f"\n{'─'*40}")
    print(f"Test: encode  [{1}, 3, {H}, {W}] image → latents")
    x = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        z = vae.encode(x)
    print(f"  Input  shape: {x.shape}")
    print(f"  Latent shape: {z.shape}  (expected [1, 16, {H//8}, {W//8}])")
    print(f"  Latent mean:  {z.mean().item():.4f}")
    print(f"  Latent std:   {z.std().item():.4f}")
    assert z.shape == (1, 16, H//8, W//8), f"Wrong latent shape: {z.shape}"
    print(f"  ✅ encode OK")
    return z


def test_decode(vae: AE, z: torch.Tensor, H=64, W=64):
    print(f"\n{'─'*40}")
    print(f"Test: decode  {z.shape} latents → image")
    with torch.no_grad():
        out = vae.decode(z)
    print(f"  Output shape: {out.shape}  (expected [1, 3, {H}, {W}])")
    print(f"  Output min:   {out.min().item():.4f}")
    print(f"  Output max:   {out.max().item():.4f}")
    assert out.shape == (1, 3, H, W), f"Wrong output shape: {out.shape}"
    print(f"  ✅ decode OK")


def test_roundtrip(vae: AE, device: str, H=128, W=128):
    print(f"\n{'─'*40}")
    print(f"Test: round-trip encode→decode  [{1}, 3, {H}, {W}]")
    x = torch.rand(1, 3, H, W, device=device)  # [0,1] range
    with torch.no_grad():
        out = vae(x)
    print(f"  Input  shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  ✅ round-trip OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',   default='/data/models/ae.safetensors')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size',   type=int, default=64,
                        help='Spatial size for encode/decode test (multiple of 8)')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")

    try:
        vae = load_vae(args.path, args.device)
    except Exception as e:
        print(f"\n❌ LOAD FAILED: {e}")
        sys.exit(1)

    try:
        z   = test_encode(vae, args.device, args.size, args.size)
        test_decode(vae, z, args.size, args.size)
        test_roundtrip(vae, args.device, args.size * 2, args.size * 2)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS PASSED — ae.safetensors loads correctly")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
