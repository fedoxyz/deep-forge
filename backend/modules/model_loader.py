"""
Model loader module.

Loads models from single files (.safetensors, .ckpt, .pt) and
provides layer introspection for LoRA targeting.
Designed to support diffusion models from CivitAI and similar sources.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def load_state_dict_from_file(path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load state dict from .safetensors, .ckpt, or .pt file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.safetensors':
        from safetensors.torch import load_file
        return load_file(path, device=device)
    elif ext in ('.ckpt', '.pt', '.pth', '.bin'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                return ckpt['state_dict']
            elif 'model' in ckpt:
                return ckpt['model']
            return ckpt
        return ckpt
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def inspect_model_layers(
    state_dict: Dict[str, torch.Tensor],
    show_shapes: bool = False,
) -> Dict[str, List[str]]:
    """
    Analyze a state dict and group layers by component.
    Useful for deciding which layers to target with LoRA.

    Returns dict of component groups -> list of layer names.
    """
    groups: Dict[str, List[str]] = {}

    for key in sorted(state_dict.keys()):
        # Try to identify the component
        parts = key.split('.')
        if len(parts) >= 2:
            group = parts[0]
            if len(parts) >= 3:
                group = f"{parts[0]}.{parts[1]}"
        else:
            group = 'root'

        if group not in groups:
            groups[group] = []

        info = key
        if show_shapes:
            info = f"{key} {list(state_dict[key].shape)}"
        groups[group].append(info)

    return groups


def get_linear_layer_names(model: nn.Module) -> List[str]:
    """Get all nn.Linear layer names in a model."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names


def get_conv_layer_names(model: nn.Module) -> List[str]:
    """Get all nn.Conv2d layer names in a model."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            names.append(name)
    return names


def get_attention_layer_names(model: nn.Module) -> List[str]:
    """Get attention-related layer names (common LoRA targets)."""
    attention_keywords = [
        'to_q', 'to_k', 'to_v', 'to_out',
        'query', 'key', 'value',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'attn', 'attention',
    ]
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(kw in name.lower() for kw in attention_keywords):
                names.append(name)
    return names


# Pre-defined LoRA target patterns for common model architectures
LORA_TARGET_PRESETS: Dict[str, Dict[str, List[str]]] = {
    'sdxl': {
        'unet_attention': [
            r'unet.*attn.*to_q',
            r'unet.*attn.*to_k',
            r'unet.*attn.*to_v',
            r'unet.*attn.*to_out',
        ],
        'unet_all_linear': [
            r'unet.*\.linear',
            r'unet.*\.proj',
        ],
        'text_encoder_1': [
            r'text_encoder\..*\.self_attn\.(q|k|v|out)_proj',
        ],
        'text_encoder_2': [
            r'text_encoder_2\..*\.self_attn\.(q|k|v|out)_proj',
        ],
    },
    'flux': {
        'transformer_attention': [
            r'transformer.*attn.*to_q',
            r'transformer.*attn.*to_k',
            r'transformer.*attn.*to_v',
            r'transformer.*attn.*to_out',
        ],
        'transformer_mlp': [
            r'transformer.*mlp.*',
        ],
        'text_encoder_clip': [
            r'text_encoder\..*self_attn\.(q|k|v|out)_proj',
        ],
        'text_encoder_t5': [
            r'text_encoder_2\..*SelfAttention\.(q|k|v|o)',
        ],
    },
    'sd15': {
        'unet_attention': [
            r'unet.*attn.*to_q',
            r'unet.*attn.*to_k',
            r'unet.*attn.*to_v',
            r'unet.*attn.*to_out',
        ],
        'text_encoder': [
            r'text_encoder\..*\.self_attn\.(q|k|v|out)_proj',
        ],
    },
    'generic': {
        'all_attention': [
            r'.*attn.*to_q',
            r'.*attn.*to_k',
            r'.*attn.*to_v',
            r'.*q_proj',
            r'.*k_proj',
            r'.*v_proj',
        ],
        'all_linear': [
            r'.*\.linear',
            r'.*\.proj',
            r'.*\.fc',
        ],
    },
}


class ModelWrapper(nn.Module):
    """
    Wrapper that loads a model from a single file and provides
    LoRA-relevant layer inspection.
    """

    def __init__(self, model: nn.Module, model_type: str = 'generic'):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_targetable_layers(self) -> Dict[str, List[str]]:
        """Get layer groups that can be targeted by LoRA."""
        result = {
            'linear_layers': get_linear_layer_names(self.model),
            'conv_layers': get_conv_layer_names(self.model),
            'attention_layers': get_attention_layer_names(self.model),
        }

        # Add preset patterns if available
        if self.model_type in LORA_TARGET_PRESETS:
            result['presets'] = LORA_TARGET_PRESETS[self.model_type]

        return result

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing where possible."""
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
            elif hasattr(module, '_set_gradient_checkpointing'):
                module._set_gradient_checkpointing(True)
