"""
Model registry for pretrained and loadable models.

Supports:
- Loading .safetensors single-file models (CivitAI style)
- Loading HuggingFace-style models if available
- Reconstructing architecture from state dict shape analysis
- Freezing/unfreezing selected layers for fine-tuning
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def load_state_dict_from_file(path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load state dict from any supported format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.safetensors':
        from safetensors.torch import load_file
        return load_file(path, device=device)
    elif ext in ('.ckpt', '.pt', '.pth', '.bin'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ('state_dict', 'model', 'model_state_dict'):
                if key in ckpt:
                    return ckpt[key]
            return ckpt
        return ckpt
    raise ValueError(f"Unsupported format: {ext}")


def analyze_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze a state dict structure for the frontend."""
    total_params = sum(t.numel() for t in state_dict.values())
    groups = {}
    linear_layers = []
    conv_layers = []
    all_layers = []

    for key, tensor in sorted(state_dict.items()):
        parts = key.split('.')
        group = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        if group not in groups:
            groups[group] = []
        groups[group].append({'key': key, 'shape': list(tensor.shape), 'params': tensor.numel()})

        if 'weight' in key:
            if len(tensor.shape) == 2:
                linear_layers.append(key)
            elif len(tensor.shape) == 4:
                conv_layers.append(key)

        all_layers.append({
            'key': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'params': tensor.numel(),
        })

    return {
        'total_params': total_params,
        'total_params_human': _human_number(total_params),
        'num_keys': len(state_dict),
        'linear_layers': linear_layers,
        'conv_layers': conv_layers,
        'groups': {k: v[:10] for k, v in groups.items()},
        'group_names': sorted(groups.keys()),
        'all_layers': all_layers[:500],
    }

def reconstruct_model_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Reconstruct a model from a state dict.
    
    - If keys have meaningful named hierarchy (e.g. layers.0.attention.to_q.weight)
      → builds a proper named module tree preserving full path structure
    - If keys are flat/simple (e.g. 0.weight, 1.weight, fc.weight)
      → falls back to nn.Sequential (original behavior)
    """
    if _has_named_hierarchy(state_dict):
        return _reconstruct_named(state_dict)
    else:
        return _reconstruct_sequential(state_dict)


def _has_named_hierarchy(state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Detect whether state dict has meaningful named structure worth preserving.
    A flat sequential model has keys like '0.weight', '1.weight'.
    A named model has keys like 'layers.0.attention.to_q.weight'.
    """
    for key in list(state_dict.keys())[:20]:
        parts = key.split('.')
        # If any non-terminal part is a non-numeric string → named hierarchy
        non_param_parts = parts[:-1]  # exclude the last part (weight/bias)
        if any(not p.isdigit() for p in non_param_parts):
            return True
    return False


def _reconstruct_sequential(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Original behavior — flat nn.Sequential for simple models."""
    layers = []
    layer_map: Dict[str, Dict[str, torch.Tensor]] = {}

    for key in sorted(state_dict.keys()):
        parts = key.rsplit('.', 1)
        prefix, param_name = (parts[0], parts[1]) if len(parts) == 2 else (key, 'weight')
        if prefix not in layer_map:
            layer_map[prefix] = {}
        layer_map[prefix][param_name] = state_dict[key]

    for prefix in sorted(layer_map.keys()):
        module = _make_leaf_module(layer_map[prefix])
        if module is not None:
            layers.append(module)

    return nn.Sequential(*layers)


def _reconstruct_named(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Build a named module hierarchy preserving full key path structure."""
    param_groups: Dict[str, Dict[str, torch.Tensor]] = {}

    PARAM_SUFFIXES = {
        'weight', 'bias', 'running_mean', 'running_var',
        'num_batches_tracked', 'weight_g', 'weight_v'
    }

    for key, tensor in state_dict.items():
        parts = key.split('.')
        # Find where the path ends and param name begins
        # Walk from the end — first part that's a known param suffix is the split
        split_idx = len(parts) - 1
        if parts[-1] in PARAM_SUFFIXES:
            split_idx = len(parts) - 1
        
        prefix = '.'.join(parts[:split_idx])
        param_name = parts[split_idx]

        if not prefix:
            prefix = key
            param_name = 'weight'

        if prefix not in param_groups:
            param_groups[prefix] = {}
        param_groups[prefix][param_name] = tensor

    tree = _NamedModuleTree()
    for prefix, params in param_groups.items():
        module = _make_leaf_module(params)
        if module is not None:
            tree.set_module(prefix, module)

    return tree.build()


def _make_leaf_module(params: Dict[str, torch.Tensor]) -> nn.Module | None:
    """Create the appropriate nn.Module for a group of parameters."""
    if 'weight' not in params:
        return None

    w = params['weight']
    has_bias = 'bias' in params

    if len(w.shape) == 2:
        layer = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
        layer.weight.data.copy_(w)
        if has_bias:
            layer.bias.data.copy_(params['bias'])
        return layer

    if len(w.shape) == 4:
        layer = nn.Conv2d(
            w.shape[1], w.shape[0],
            kernel_size=(w.shape[2], w.shape[3]),
            bias=has_bias,
        )
        layer.weight.data.copy_(w)
        if has_bias:
            layer.bias.data.copy_(params['bias'])
        return layer

    if len(w.shape) == 1:
        if 'running_mean' in params:
            layer = nn.BatchNorm1d(w.shape[0])
            layer.weight.data.copy_(w)
            if has_bias:
                layer.bias.data.copy_(params['bias'])
            if 'running_mean' in params:
                layer.running_mean.data.copy_(params['running_mean'])
            if 'running_var' in params:
                layer.running_var.data.copy_(params['running_var'])
            return layer
        else:
            # LayerNorm, embedding, or other 1D — generic parameter container
            container = nn.Module()
            for pname, tensor in params.items():
                container.register_parameter(
                    pname, nn.Parameter(tensor, requires_grad=False)
                )
            return container

    return None


class _NamedModuleTree:
    """
    Builds a nested nn.Module hierarchy from dotted path assignments.
    String keys → named children via add_module.
    All-numeric sibling keys → nn.ModuleList.
    Mixed → named children with numeric keys kept as strings.
    """

    def __init__(self):
        self._tree: dict = {}

    def set_module(self, path: str, module: nn.Module) -> None:
        parts = path.split('.')
        node = self._tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            elif isinstance(node[part], nn.Module):
                # Collision — a leaf was placed where a subtree is needed
                # Wrap existing module and continue
                node[part] = {'_leaf': node[part]}
            node = node[part]
        node[parts[-1]] = module

    def build(self) -> nn.Module:
        return self._build_node(self._tree)

    def _build_node(self, node: dict) -> nn.Module:
        if isinstance(node, nn.Module):
            return node

        keys = list(node.keys())
        all_numeric = all(k.isdigit() for k in keys)

        container = nn.Module()
        if all_numeric:
            # Use ModuleList but also add_module with string keys
            # so named_modules() still returns 'layers.0.attention...' paths
            for k in sorted(keys, key=int):
                container.add_module(k, self._build_node(node[k]))
        else:
            for k in sorted(keys):
                container.add_module(k, self._build_node(node[k]))

        return container

def freeze_model(model: nn.Module):
    """Freeze all parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_layers(model: nn.Module, patterns: List[str]):
    """Unfreeze layers matching patterns."""
    compiled = [re.compile(p) for p in patterns]
    unfrozen = 0
    for name, param in model.named_parameters():
        for pattern in compiled:
            if pattern.search(name):
                param.requires_grad = True
                unfrozen += 1
                break
    return unfrozen


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of a model's architecture."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    layer_info = []
    for name, module in model.named_modules():
        if name == '':
            continue
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params > 0:
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'params': n_params,
                'trainable': any(p.requires_grad for p in module.parameters(recurse=False)),
            })

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'total_human': _human_number(total),
        'trainable_human': _human_number(trainable),
        'layers': layer_info,
    }


LORA_TARGET_PRESETS = {
    'sdxl': {
        'unet_attention': [r'unet.*attn.*to_[qkvo]'],
        'unet_all_linear': [r'unet.*\.(linear|proj)'],
        'text_encoder_1': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
        'text_encoder_2': [r'text_encoder_2\..*self_attn\.[qkvo]_proj'],
    },
    'flux': {
        'transformer_attention': [r'transformer.*attn.*to_[qkvo]'],
        'transformer_mlp': [r'transformer.*mlp.*'],
        'text_encoder_clip': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
        'text_encoder_t5': [r'text_encoder_2\..*SelfAttention\.[qkvo]'],
    },
    'sd15': {
        'unet_attention': [r'unet.*attn.*to_[qkvo]'],
        'text_encoder': [r'text_encoder\..*self_attn\.[qkvo]_proj'],
    },
    'generic': {
        'all_attention': [r'.*attn.*to_[qkv]', r'.*[qkv]_proj'],
        'all_linear': [r'.*\.(linear|proj|fc)'],
    },
}


def _human_number(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
