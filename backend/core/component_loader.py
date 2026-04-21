"""
Universal multi-component model loader.

Every training run produces a ComponentBundle, even single-file models.

Key concepts:
  - Every model source (file, spec, empty) becomes a ModelComponent
  - Components have an execution_order for pipeline sequencing
  - Components can declare inputs/outputs for explicit data flow
  - A ComponentBundle is an ordered collection ready for pipeline execution
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from backend.modules.model_registry import load_state_dict_from_file


# ═══════════════════════════════════════════════════════════════
# ModelComponent
# ═══════════════════════════════════════════════════════════════

class ModelComponent:
    """
    A single loaded component (vae, clip_l, unet, dit_high, custom_adapter, etc).

    Each component knows:
      - Its identity (name, role)
      - Its weights (state_dict or built module)
      - Its training strategy (from config)
      - Its position in the pipeline (execution_order)
      - What it consumes/produces (input_keys / output_keys)
    """

    def __init__(
        self,
        name: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        module: Optional[nn.Module] = None,
        role: str = 'generic',
        trainable: bool = False,
        dtype: Optional[str] = None,
        device: str = 'cpu',
        config: Optional[Dict] = None,
    ):
        self.name = name
        self.state_dict = state_dict or {}
        self.module = module
        self.role = role
        self.trainable = trainable
        self.dtype = dtype
        self.device = device
        self.config = config or {}
        self.param_count = sum(t.numel() for t in self.state_dict.values()) if self.state_dict else 0

        if self.module is not None and self.param_count == 0:
            self.param_count = sum(p.numel() for p in self.module.parameters())

        # Pipeline position (from config)
        self.execution_order = self.config.get('execution_order', 0)

        # Output cache (for components marked cache_output: true)
        self._cached_output = None

    def to(self, device):
        self.device = str(device)
        if self.module is not None:
            self.module.to(device)
        return self

    def freeze(self):
        if self.module:
            for p in self.module.parameters():
                p.requires_grad = False
        self.trainable = False

    def unfreeze(self, patterns: Optional[List[str]] = None):
        if self.module is None:
            return
        if patterns:
            compiled = [re.compile(p) for p in patterns]
            for pname, param in self.module.named_parameters():
                for pat in compiled:
                    if pat.search(pname):
                        param.requires_grad = True
                        break
        else:
            for p in self.module.parameters():
                p.requires_grad = True
        self.trainable = True

    def get_info(self) -> Dict[str, Any]:
        trainable_count = 0
        if self.module:
            trainable_count = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        return {
            'name': self.name,
            'role': self.role,
            'trainable': self.trainable,
            'param_count': self.param_count,
            'trainable_param_count': trainable_count,
            'dtype': self.dtype,
            'device': self.device,
            'has_module': self.module is not None,
            'strategy': self.config.get('training', {}).get('strategy', 'frozen'),
            'execution_order': self.execution_order,
        }


# ═══════════════════════════════════════════════════════════════
# ComponentBundle
# ═══════════════════════════════════════════════════════════════

class ComponentBundle:
    """
    Ordered collection of model components.
    Iteration yields components in execution_order.
    """

    def __init__(self):
        self.components: Dict[str, ModelComponent] = {}

    def add(self, component: ModelComponent):
        self.components[component.name] = component

    def get(self, name: str) -> Optional[ModelComponent]:
        return self.components.get(name)

    def get_by_role(self, role: str) -> List[ModelComponent]:
        return [c for c in self.ordered() if c.role == role]

    def get_trainable(self) -> List[ModelComponent]:
        return [c for c in self.components.values() if c.trainable]

    def get_all_trainable_params(self) -> List[torch.nn.Parameter]:
        params = []
        for comp in self.get_trainable():
            if comp.module:
                params.extend([p for p in comp.module.parameters() if p.requires_grad])
        return params

    def ordered(self) -> List[ModelComponent]:
        """Components sorted by execution_order."""
        return sorted(self.components.values(), key=lambda c: c.execution_order)

    def to(self, device):
        for c in self.components.values():
            c.to(device)
        return self

    def info(self) -> Dict[str, Any]:
        return {
            'components': {n: c.get_info() for n, c in self.components.items()},
            'total_params': sum(c.param_count for c in self.components.values()),
            'trainable_components': [c.name for c in self.get_trainable()],
            'execution_order': [c.name for c in self.ordered()],
        }

    def __getitem__(self, key: str) -> ModelComponent:
        return self.components[key]

    def __contains__(self, key: str) -> bool:
        return key in self.components

    def __iter__(self):
        return iter(self.ordered())

    def __len__(self):
        return len(self.components)


# ═══════════════════════════════════════════════════════════════
# Loading functions
# ═══════════════════════════════════════════════════════════════

def _load_model_spec_by_name(spec_name: str) -> Dict[str, Any]:
    """
    Load a saved model spec JSON by name.
    model_builder.build_model() takes a spec dict, not a name string.
    This bridges the gap.
    """
    search_dirs = [
        './saved_model_specs',
        './model_specs',
        os.path.join(os.path.dirname(__file__), '..', 'saved_model_specs'),
    ]
    for search_dir in search_dirs:
        for ext in ('.json', '.yaml', '.yml'):
            path = os.path.join(search_dir, f'{spec_name}{ext}')
            if os.path.exists(path):
                with open(path) as f:
                    if ext == '.json':
                        return json.load(f)
                    else:
                        try:
                            import yaml
                            return yaml.safe_load(f) or {}
                        except ImportError:
                            continue
    raise FileNotFoundError(f"Model spec '{spec_name}' not found. Searched: {search_dirs}")

def _try_special_loader(role: str, path: str, spec: dict) -> Optional[nn.Module]:
    """
    Return a fully constructed nn.Module for components that need
    custom loading (VAE, text encoder). Returns None to fall through
    to generic state dict loading.
    """
    import torch
    filename = os.path.basename(path).lower()

    dtype_str = spec.get('dtype', 'bfloat16')
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Flux VAE ─────────────────────────────────────────────────────────────
    if role == 'vae' or 'ae.safetensors' in filename:
        from backend.pipelines.flux_vae_wrapper import FluxVAEWrapper
        print(f"[Loader] Special load: Flux VAE from '{path}'")
        return FluxVAEWrapper(path, dtype=dtype, device=device)

    # ── Qwen3 text encoder ────────────────────────────────────────────────────
    if role == 'text_encoder' or 'qwen' in filename:
        from backend.pipelines.qwen3_text_encoder import Qwen3TextEncoder
        print(f"[Loader] Special load: Qwen3 text encoder from '{path}'")
        return Qwen3TextEncoder(path, dtype=dtype, device=device)

    if role == 'denoiser' or any(x in filename for x in ['z_image', 'zimage']):
        from backend.pipelines.zimage_dit_wrapper import ZImageDiTWrapper
        print(f"[Loader] Special load: Z-Image DiT from '{path}'")
        return ZImageDiTWrapper(path, dtype=dtype, device=device)

    return None

def load_component_from_spec(spec: Dict[str, Any], base_dir: str = '') -> ModelComponent:
    """
    Load a single component from a config spec dict.

    source: "file"  — load state_dict from path
    source: "spec"  — build nn.Module from model spec (model_builder.build_model)
    source: "empty" — no weights, module set externally
    """
    name = spec['name']
    source = spec.get('source', 'file')

    sd = {}
    module = None

    if source == 'spec':
        spec_ref = spec.get('spec_name') or spec.get('path', '')
        if not spec_ref:
            raise ValueError(f"Component '{name}' with source='spec' needs 'spec_name'")

        from backend.core.model_builder import build_model

        if isinstance(spec_ref, dict):
            model_spec_dict = spec_ref
        else:
            model_spec_dict = _load_model_spec_by_name(spec_ref)

        module = build_model(model_spec_dict)

    elif source == 'empty':
        pass

    else:  # source == "file"
        path = spec.get('path', '')
        if not path:
            raise ValueError(f"Component '{name}' with source='file' needs 'path'")
        if not os.path.isabs(path) and base_dir:
            path = os.path.join(base_dir, path)

        # ── Special loaders by role or filename ──────────────────────────
        role = spec.get('role', 'generic')
        _special = _try_special_loader(role, path, spec)
        if _special is not None:
            return ModelComponent(
                name=name,
                state_dict={},
                module=_special,
                role=role,
                trainable=spec.get('trainable', False),
                dtype=spec.get('dtype'),
                config=spec,
            )

        sd = load_state_dict_from_file(path)

        key_filter = spec.get('key_filter')
        if key_filter:
            pattern = re.compile(key_filter)
            sd = {k: v for k, v in sd.items() if pattern.search(k)}

        strip_prefix = spec.get('strip_prefix')
        if strip_prefix:
            sd = {
                k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v
                for k, v in sd.items()
            }

        key_prefix = spec.get('key_prefix')
        if key_prefix:
            sd = {f"{key_prefix}{k}": v for k, v in sd.items()}

        dtype_str = spec.get('dtype')
        if dtype_str:
            dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
            dt = dtype_map.get(dtype_str)
            if dt:
                sd = {k: v.to(dt) for k, v in sd.items()}

    return ModelComponent(
        name=name,
        state_dict=sd,
        module=module,
        role=spec.get('role', 'generic'),
        trainable=spec.get('trainable', False),
        dtype=spec.get('dtype'),
        config=spec,
    )


def load_component_bundle(components_config: List[Dict[str, Any]],
                          base_dir: str = '') -> ComponentBundle:
    """Load a multi-component model bundle from a config list."""
    bundle = ComponentBundle()

    for i, spec in enumerate(components_config):
        # Auto-assign execution_order if not set
        if 'execution_order' not in spec:
            spec['execution_order'] = i

        comp = load_component_from_spec(spec, base_dir=base_dir)
        bundle.add(comp)

    return bundle


# ═══════════════════════════════════════════════════════════════
# Normalization — everything becomes a components list
# ═══════════════════════════════════════════════════════════════

def normalize_single_model_to_bundle(model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Wrap model.path as a one-component list."""
    path = model_config.get('path', '')
    if not path:
        return []
    return [{
        'name': 'model',
        'source': 'file',
        'path': path,
        'role': 'generic',
        'trainable': True,
        'dtype': model_config.get('dtype', 'float16'),
        'execution_order': 0,
    }]


def normalize_model_spec_to_bundle(spec_name) -> List[Dict[str, Any]]:
    """Wrap model_spec as a one-component list."""
    if not spec_name:
        return []
    return [{
        'name': 'model',
        'source': 'spec',
        'spec_name': spec_name,
        'role': 'generic',
        'trainable': True,
        'execution_order': 0,
    }]
