import re
import torch
from typing import Optional
from safetensors.torch import load_file, save_file


class ModelFormatNormalizer:
    """
    Detects, normalizes, and denormalizes model/LoRA state dict formats.
    
    Canonical internal format:
      - No top-level prefix
      - Attention always split: attention.to_q, attention.to_k, attention.to_v, attention.to_out
      - Everything else unchanged
    
    Supported formats:
      - diffusers_split   : HuggingFace diffusers (to_q/to_k/to_v/to_out, may have prefix)
      - civitai_fused     : ComfyUI/single-file (qkv fused, attention.out, no prefix)
      - unknown           : Pass-through, no conversion
    
    Usage:
        normalizer = ModelFormatNormalizer()
        canonical = normalizer.normalize(raw_state_dict)
        # ... inject LoRA, train ...
        lora_sd = normalizer.denormalize(canonical_lora_sd)
        save_file(lora_sd, "lora.safetensors")
    """

    # ── Prefix detection ──────────────────────────────────────────────────────

    KNOWN_PREFIXES = [
        "diffusion_model.",
        "model.diffusion_model.",
        "network.",
        "transformer.",
        "base_model.model.",  # some PEFT-style saves
    ]

    # ── Attention pattern detection ───────────────────────────────────────────

    # civitai/fused patterns
    _RE_FUSED_QKV = re.compile(r"^(.*?)\.attention\.qkv$")
    _RE_FUSED_OUT = re.compile(r"^(.*?)\.attention\.out$")

    # diffusers/split patterns
    _RE_SPLIT_Q   = re.compile(r"^(.*?)\.attention\.to_q$")
    _RE_SPLIT_K   = re.compile(r"^(.*?)\.attention\.to_k$")
    _RE_SPLIT_V   = re.compile(r"^(.*?)\.attention\.to_v$")
    _RE_SPLIT_OUT = re.compile(r"^(.*?)\.attention\.to_out(?:\.0)?$")

    # LoRA suffix — matches lora_A.weight, lora_B.weight, etc.
    _RE_LORA_SUFFIX = re.compile(r"\.(lora_[AB](?:\.\w+)*)$")

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.detected_format: str = "unknown"
        self.detected_prefix: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(self, state_dict: dict) -> dict:
        """
        Convert any supported format → canonical.
        Remembers detected format and prefix for denormalize().
        """
        sd, prefix = self._strip_prefix(state_dict)
        self.detected_prefix = prefix
        self.detected_format = self._detect_format(sd)

        print(f"[Normalizer] Detected format='{self.detected_format}' prefix='{prefix}' "
              f"({len(state_dict)} keys)")

        if self.detected_format == "civitai_fused":
            sd = self._fused_to_split(sd)
        # diffusers_split and unknown pass through as-is

        return sd

    def denormalize(self, canonical: dict) -> dict:
        """
        Convert canonical → original detected format.
        Call this before saving LoRA weights.
        """
        if self.detected_format == "civitai_fused":
            sd = self._split_to_fused(canonical)
        else:
            sd = dict(canonical)

        if self.detected_prefix:
            sd = {f"{self.detected_prefix}{k}": v for k, v in sd.items()}

        return sd

    def convert_lora_to_model_format(self, adapter_sd: dict) -> dict:
        """
        Convert a LoRA adapter into whatever format matches THIS model.
        Used for training adapters that need to inject into the base model.
        
        If model is civitai_fused:  adapter split keys → fused keys
        If model is diffusers_split: adapter fused keys → split keys
        """
        # First normalize adapter to canonical (split)
        tmp = ModelFormatNormalizer(strict=self.strict)
        canonical = tmp.normalize(adapter_sd)
        
        # Then convert canonical → this model's format
        if self.detected_format == "civitai_fused":
            # Model uses fused QKV — convert adapter split → fused
            return self._split_to_fused(canonical)
        else:
            # Model uses split or unknown — canonical is fine
            return canonical

    @staticmethod
    def detect_format_of(state_dict: dict) -> str:
        """Stateless format detection — doesn't affect instance state."""
        tmp = ModelFormatNormalizer()
        sd, _ = tmp._strip_prefix(state_dict)
        return tmp._detect_format(sd)

    # ── Format detection ──────────────────────────────────────────────────────

    def _detect_format(self, sd: dict) -> str:
        """
        Detect format from (already prefix-stripped) state dict.
        Checks a sample of keys for attention naming patterns.
        """
        # Sample keys — check up to 100, bias toward attention-related ones
        all_keys = list(sd.keys())
        attn_keys = [k for k in all_keys if "attention" in k]
        sample = attn_keys[:50] or all_keys[:100]

        has_to_q  = any("attention.to_q"  in k for k in sample)
        has_to_k  = any("attention.to_k"  in k for k in sample)
        has_qkv   = any("attention.qkv"   in k for k in sample)
        has_fused_out = any(
            bool(self._RE_FUSED_OUT.search(self._strip_lora_suffix(k)[0]))
            for k in sample
        )

        if has_to_q or has_to_k:
            return "diffusers_split"
        if has_qkv or has_fused_out:
            return "civitai_fused"
        return "unknown"

    # ── Prefix handling ───────────────────────────────────────────────────────

    def _strip_prefix(self, sd: dict) -> tuple[dict, str]:
        """
        Detect and strip a known top-level prefix.
        Returns (stripped_dict, prefix_found).
        Handles mixed-prefix dicts gracefully.
        """
        if not sd:
            return sd, ""

        # Find which prefix (if any) most keys start with
        prefix_counts = {p: 0 for p in self.KNOWN_PREFIXES}
        sample = list(sd.keys())[:200]
        for k in sample:
            for p in self.KNOWN_PREFIXES:
                if k.startswith(p):
                    prefix_counts[p] += 1
        
        best_prefix = max(prefix_counts, key=lambda p: prefix_counts[p])
        if prefix_counts[best_prefix] == 0:
            return sd, ""

        # Verify it covers the majority of keys (avoid false positives)
        coverage = prefix_counts[best_prefix] / len(sample)
        if coverage < 0.5:
            return sd, ""

        stripped = {}
        for k, v in sd.items():
            stripped[k[len(best_prefix):] if k.startswith(best_prefix) else k] = v

        return stripped, best_prefix

    # ── civitai_fused → canonical (split) ────────────────────────────────────

    def _fused_to_split(self, sd: dict) -> dict:
        """
        Convert fused QKV and out keys to split to_q/to_k/to_v/to_out.
        Handles both plain weights and LoRA A/B weights.
        
        Examples:
          layers.0.attention.qkv
            → layers.0.attention.to_q
            → layers.0.attention.to_k  
            → layers.0.attention.to_v

          layers.0.attention.qkv.lora_A.weight
            → layers.0.attention.to_q.lora_A.weight
            → layers.0.attention.to_k.lora_A.weight
            → layers.0.attention.to_v.lora_A.weight

          layers.0.attention.out
            → layers.0.attention.to_out

          layers.0.attention.out.lora_A.weight
            → layers.0.attention.to_out.lora_A.weight
        """
        new_sd = {}

        for k, v in sd.items():
            base_key, lora_suffix = self._strip_lora_suffix(k)

            # ── fused QKV ────────────────────────────────────────────────────
            m = self._RE_FUSED_QKV.match(base_key)
            if m:
                layer_path = m.group(1)
                if lora_suffix:
                    # LoRA weight — split A/B differently
                    # lora_A: [rank, in_dim] → split along in_dim? No.
                    # lora_A shape: [rank, in]  lora_B shape: [out, rank]
                    # For fused QKV:
                    #   lora_A: [rank, in_dim]        → same for q/k/v (shared input)
                    #   lora_B: [3*out_dim, rank]     → split into thirds
                    if "lora_A" in lora_suffix:
                        # lora_A is shared input projection — duplicate for q/k/v
                        # This is an approximation; exact split requires knowing
                        # the original separate ranks. Best we can do.
                        new_sd[f"{layer_path}.attention.to_q.{lora_suffix}"] = v
                        new_sd[f"{layer_path}.attention.to_k.{lora_suffix}"] = v
                        new_sd[f"{layer_path}.attention.to_v.{lora_suffix}"] = v
                    elif "lora_B" in lora_suffix:
                        # lora_B output dim — split into thirds
                        dim = v.shape[0] // 3
                        new_sd[f"{layer_path}.attention.to_q.{lora_suffix}"] = v[:dim].contiguous()
                        new_sd[f"{layer_path}.attention.to_k.{lora_suffix}"] = v[dim:2*dim].contiguous()
                        new_sd[f"{layer_path}.attention.to_v.{lora_suffix}"] = v[2*dim:].contiguous()
                    else:
                        # Unknown LoRA param — pass through
                        new_sd[k] = v
                else:
                    # Plain weight — split into thirds along dim 0
                    dim = v.shape[0] // 3
                    new_sd[f"{layer_path}.attention.to_q"] = v[:dim].contiguous()
                    new_sd[f"{layer_path}.attention.to_k"] = v[dim:2*dim].contiguous()
                    new_sd[f"{layer_path}.attention.to_v"] = v[2*dim:].contiguous()
                continue

            # ── fused out ────────────────────────────────────────────────────
            m = self._RE_FUSED_OUT.match(base_key)
            if m:
                layer_path = m.group(1)
                new_key = f"{layer_path}.attention.to_out"
                new_sd[f"{new_key}.{lora_suffix}" if lora_suffix else new_key] = v
                continue

            # ── everything else ───────────────────────────────────────────────
            new_sd[k] = v

        return new_sd

    # ── canonical (split) → civitai_fused ────────────────────────────────────

    def _split_to_fused(self, sd: dict) -> dict:
        """
        Convert split to_q/to_k/to_v/to_out back to fused QKV and out.
        Handles both plain weights and LoRA A/B weights.
        Buffers Q/K/V and fuses when all three are present.
        """
        new_sd = {}
        # Buffer: (layer_path, lora_suffix) → {'q': t, 'k': t, 'v': t}
        qkv_buffer: dict[tuple, dict] = {}

        for k, v in sd.items():
            base_key, lora_suffix = self._strip_lora_suffix(k)

            # ── to_q ─────────────────────────────────────────────────────────
            m = self._RE_SPLIT_Q.match(base_key)
            if m:
                buf_key = (m.group(1), lora_suffix)
                qkv_buffer.setdefault(buf_key, {})['q'] = v
                continue

            # ── to_k ─────────────────────────────────────────────────────────
            m = self._RE_SPLIT_K.match(base_key)
            if m:
                buf_key = (m.group(1), lora_suffix)
                qkv_buffer.setdefault(buf_key, {})['k'] = v
                continue

            # ── to_v ─────────────────────────────────────────────────────────
            m = self._RE_SPLIT_V.match(base_key)
            if m:
                buf_key = (m.group(1), lora_suffix)
                qkv_buffer.setdefault(buf_key, {})['v'] = v
                continue

            # ── to_out ───────────────────────────────────────────────────────
            m = self._RE_SPLIT_OUT.match(base_key)
            if m:
                layer_path = m.group(1)
                new_key = f"{layer_path}.attention.out"
                new_sd[f"{new_key}.{lora_suffix}" if lora_suffix else new_key] = v
                continue

            new_sd[k] = v

        # ── Flush QKV buffer ─────────────────────────────────────────────────
        for (layer_path, lora_suffix), qkv in qkv_buffer.items():
            fused_key = f"{layer_path}.attention.qkv"
            if lora_suffix:
                fused_key = f"{fused_key}.{lora_suffix}"

            if all(x in qkv for x in ('q', 'k', 'v')):
                if "lora_A" in (lora_suffix or ""):
                    # lora_A was duplicated during split — just take one copy
                    new_sd[fused_key] = qkv['q'].contiguous()
                else:
                    # Plain weight or lora_B — concatenate along dim 0
                    new_sd[fused_key] = torch.cat(
                        [qkv['q'], qkv['k'], qkv['v']], dim=0
                    ).contiguous()
            else:
                # Incomplete QKV — keep split (shouldn't happen in normal flow)
                present = list(qkv.keys())
                print(f"[Normalizer] WARNING: Incomplete QKV at '{layer_path}' "
                      f"lora='{lora_suffix}', found={present} — keeping split")
                for letter, tensor in qkv.items():
                    name = {'q': 'to_q', 'k': 'to_k', 'v': 'to_v'}[letter]
                    fallback = f"{layer_path}.attention.{name}"
                    new_sd[f"{fallback}.{lora_suffix}" if lora_suffix else fallback] = tensor

        return new_sd

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _strip_lora_suffix(self, key: str) -> tuple[str, str]:
        # Find lora_A or lora_B boundary
        for marker in [".lora_A", ".lora_B"]:
            idx = key.find(marker)
            if idx != -1:
                return key[:idx], key[idx + 1:]
        return key, ""
