"""
Z-Image Turbo DiT wrapper.
Uses Z-Image repo's ZImageTransformer2DModel �~@~T no reimplementation.
Config hardcoded from official config.json �~@~T no external file needed.
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file


# Official config from Tongyi-MAI/Z-Image-Turbo transformer/config.json
_DEFAULT_CONFIG = {
    "all_patch_size":    (2,),
    "all_f_patch_size":  (1,),
    "in_channels":       16,
    "dim":               3840,
    "n_layers":          30,
    "n_refiner_layers":  2,
    "n_heads":           30,
    "n_kv_heads":        30,
    "norm_eps":          1e-5,
    "qk_norm":           True,
    "cap_feat_dim":      2560,
    "rope_theta":        256.0,
    "t_scale":           1000.0,
    "axes_dims":         [32, 48, 48],
    "axes_lens":         [1536, 512, 512],
}


class ZImageDiTWrapper(nn.Module):

    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        super().__init__()
        self._dtype = dtype
        self._device = device or torch.device('cpu')
        self.model = self._load(path)

    def _load(self, path: str):
        try:
            from zimage.transformer import ZImageTransformer2DModel
        except ImportError:
            raise RuntimeError(
                "Cannot import zimage.transformer.\n"
                "Add to Dockerfile:\n"
                "  RUN git clone https://github.com/Tongyi-MAI/Z-Image /app/z_image_repo && "
                "pip install -e /app/z_image_repo"
            )

        print(f"[ZImageDiT] Building ZImageTransformer2DModel...")
        with torch.device("meta"):
            model = ZImageTransformer2DModel(**_DEFAULT_CONFIG).to(self._dtype)

        print(f"[ZImageDiT] Loading weights from '{path}'")
        sd = load_file(path, device="cpu")
        converted = self._convert_comfy_to_native(sd)
        del sd

        model.to_empty(device="cpu")
        missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
        del converted

        if len(missing) > 0:
            print(f"[ZImageDiT] Missing keys ({len(missing)}): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"[ZImageDiT] Unexpected keys ({len(unexpected)}): {list(unexpected)[:10]}")
        if len(unexpected) > 20:
            raise RuntimeError(
                f"[ZImageDiT] Too many unexpected keys ({len(unexpected)}) �~@~T "
                "conversion produced wrong key names."
            )

        print(f"[ZImageDiT] Moving to {self._device}...")
        model = model.to(self._device).eval()
        self._fix_all_tensors(model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        self._fix_all_tensors(self.model)
    
        hidden_states = hidden_states.to(device=self._device, dtype=self._dtype)
        timestep = timestep.to(device=self._device, dtype=self._dtype)
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [f.to(device=self._device, dtype=self._dtype)
                                      for f in encoder_hidden_states]
    
        if hasattr(self.model, 'cap_pad_token') and self.model.cap_pad_token is not None:
            if self.model.cap_pad_token.dtype != self._dtype:
                self.model.cap_pad_token = self.model.cap_pad_token.to(self._dtype)
    
        # ── FIX: patch cap_embedder (or whatever produces cap_feats) output dtype ──
        # The layer producing cap_feats has float32 weights that modules() misses.
        # We hook its output to force-cast to our dtype.
        hook_handle = None
        cap_embedder = self._find_cap_embedder()
        if cap_embedder is not None:
            def _cast_hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.dtype != self._dtype:
                    return output.to(self._dtype)
            hook_handle = cap_embedder.register_forward_hook(_cast_hook)
    
        try:
            x_list = [h.unsqueeze(1) for h in hidden_states.unbind(0)]
            out_list, _ = self.model(x_list, timestep, encoder_hidden_states)
            result = torch.stack([o.squeeze(1) for o in out_list], dim=0)
        finally:
            if hook_handle is not None:
                hook_handle.remove()
    
        return result
    
    def _find_cap_embedder(self):
        """Find the layer that produces cap_feats (float32 source of the dtype mismatch)."""
        # Common attribute names in dit architectures
        for attr_name in ('cap_embedder', 'caption_embedder', 'text_embedder',
                          'cap_proj', 'caption_projection', 'context_embedder'):
            layer = getattr(self.model, attr_name, None)
            if layer is not None and isinstance(layer, nn.Module):
                return layer
        # Also check one level down
        for child_name, child in self.model.named_children():
            for attr_name in ('cap_embedder', 'caption_embedder', 'cap_proj'):
                layer = getattr(child, attr_name, None)
                if layer is not None and isinstance(layer, nn.Module):
                    return layer
        return None

    def _convert_comfy_to_native(self, sd: dict) -> dict:
        """
        Reverse the z_image_convert_original_to_comfy.py conversion.
        
        ComfyUI format �~F~R Native format:
          x_embedder              �~F~R all_x_embedder.2-1
          final_layer             �~F~R all_final_layer.2-1
          .attention.out          �~F~R .attention.to_out.0  (rename only)
          .attention.k_norm       �~F~R .attention.norm_k    (rename only)
          .attention.q_norm       �~F~R .attention.norm_q    (rename only)
          .attention.qkv.weight   �~F~R split into to_q / to_k / to_v  (unfuse)
        """
        out = {}
        for k, v in sd.items():

            # �~T~@�~T~@ Unfuse qkv �~F~R split into to_q, to_k, to_v �~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@�~T~@
            if k.endswith('.attention.qkv.weight'):
                prefix = k[:-len('.attention.qkv.weight')]
                # qkv was cat([q, k, v], dim=0) �~@~T split into thirds
                dim = v.shape[0] // 3
                out[f'{prefix}.attention.to_q.weight'] = v[:dim].contiguous()
                out[f'{prefix}.attention.to_k.weight'] = v[dim:2*dim].contiguous()
                out[f'{prefix}.attention.to_v.weight'] = v[2*dim:].contiguous()
                continue

            new_k = k

            # �~T~@�~T~@ Simple renames (reverse of replace_keys in conversion script) �~T~@�~T~@
            new_k = new_k.replace('x_embedder.',           'all_x_embedder.2-1.')
            new_k = new_k.replace('final_layer.',           'all_final_layer.2-1.')
            new_k = new_k.replace('.attention.out.weight',  '.attention.to_out.0.weight')
            new_k = new_k.replace('.attention.out.bias',    '.attention.to_out.0.bias')
            new_k = new_k.replace('.attention.k_norm.',     '.attention.norm_k.')
            new_k = new_k.replace('.attention.q_norm.',     '.attention.norm_q.')

            out[new_k] = v

        return out

    def _fix_all_tensors(self, model):
        """
        Cast all floating tensors to self._device / self._dtype.
        Handles: nn.Parameter, registered buffers, unregistered tensor attrs,
        AND modules stored in plain Python lists/dicts (missed by model.modules()).
        """
        visited = set()
    
        def _fix_module(module):
            mid = id(module)
            if mid in visited:
                return
            visited.add(mid)
    
            # Registered parameters
            for name, param in list(module._parameters.items()):
                if param is not None and (param.device != self._device or param.dtype != self._dtype):
                    module._parameters[name] = nn.Parameter(
                        param.data.to(self._device, self._dtype),
                        requires_grad=param.requires_grad
                    )
    
            # Registered buffers
            for name, buf in list(module._buffers.items()):
                if buf is not None and buf.is_floating_point():
                    if buf.device != self._device or buf.dtype != self._dtype:
                        module._buffers[name] = buf.to(self._device, self._dtype)
    
            # Unregistered tensor attributes on this module's __dict__
            for name, attr in list(module.__dict__.items()):
                if isinstance(attr, torch.Tensor) and not isinstance(attr, nn.Parameter):
                    if attr.is_floating_point() and (attr.device != self._device or attr.dtype != self._dtype):
                        object.__setattr__(module, name, attr.to(self._device, self._dtype))
                        print(f"[_fix_all_tensors] Fixed unregistered tensor: "
                              f"{type(module).__name__}.{name} {attr.dtype}→{self._dtype}")
    
                # ── NEW: recurse into plain list/dict of modules ──
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, nn.Module):
                            _fix_module(item)
                elif isinstance(attr, dict):
                    for item in attr.values():
                        if isinstance(item, nn.Module):
                            _fix_module(item)
    
            # Recurse into registered children
            for child in module.children():
                _fix_module(child)
    
        _fix_module(model)
    
        # ── DEBUG: audit anything still wrong after fixing ──
        bad_params = [
            (n, p.dtype, tuple(p.shape))
            for n, p in model.named_parameters()
            if p.is_floating_point() and p.dtype != self._dtype
        ]
        bad_bufs = [
            (n, b.dtype)
            for n, b in model.named_buffers()
            if b.is_floating_point() and b.dtype != self._dtype
        ]
        if bad_params:
            print(f"[_fix_all_tensors] ⚠ Still wrong dtype after fix — params: {bad_params[:8]}")
        if bad_bufs:
            print(f"[_fix_all_tensors] ⚠ Still wrong dtype after fix — buffers: {bad_bufs[:8]}")
        if not bad_params and not bad_bufs:
            print(f"[_fix_all_tensors] ✓ All floating tensors confirmed {self._dtype}")
