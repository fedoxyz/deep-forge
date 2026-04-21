"""
Configuration management.

Component training.strategy controls everything:
  - "frozen"   : no training fields apply (no lr, no grad norm, nothing)
  - "lora"     : lora sub-config applies (rank, alpha, target_patterns)
  - "finetune" : unfreeze_patterns, lr, weight_decay, max_grad_norm apply
  - "full"     : lr, weight_decay, max_grad_norm apply
  - "adapter"  : like full, but component is a custom adapter module (from model builder)
  - null       : infer from general.mode
"""

import os
import json
import copy
from typing import Any, Dict, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Component Schema
# ═══════════════════════════════════════════════════════════════

COMPONENT_SCHEMA = {
    'name': '',                    # unique identifier
    'source': 'file',              # 'file' | 'spec' | 'empty'
    'path': '',                    # for source: file
    'spec_name': '',               # for source: spec (references saved model spec)
    'role': 'generic',             # denoiser | vae | text_encoder | adapter | generic | training_adapter
    'dtype': 'float16',

    # Pipeline ordering and data flow
    'execution_order': 0,          # lower = runs first
    'forward': {
        'input_key': None,         # batch key to read (null = previous component's output)
        'output_key': None,        # batch key to write (null = component name)
        'no_grad': False,          # run in torch.no_grad
        'cache_output': False,     # cache output across steps (frozen encoders)
        'dtype_override': None,    # cast to this dtype during forward
    },

    # Training — which fields matter depends on strategy:
    #   frozen:   NOTHING below matters
    #   lora:     lora.*, lr, weight_decay, max_grad_norm, freeze_epochs
    #   finetune: unfreeze_patterns, lr, weight_decay, max_grad_norm, freeze_epochs
    #   full:     lr, weight_decay, max_grad_norm, freeze_epochs
    #   adapter:  lr, weight_decay, max_grad_norm, freeze_epochs
    'training': {
        'strategy': None,          # 'frozen' | 'lora' | 'finetune' | 'full' | 'adapter' | null
        'lr': None,                # null = use global
        'weight_decay': None,
        'max_grad_norm': None,
        'freeze_epochs': 0,        # delay training for N epochs
        'unfreeze_patterns': [],   # for strategy: finetune
        'lora': None,  # {rank, alpha, dropout, target_patterns, conv_rank, conv_alpha}
    },
}


# ═══════════════════════════════════════════════════════════════
# Default Config
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'pipeline': {
        'name': None,
        'params': {},
    },

    'sampling': {
        'enabled': False,
        'every_n_steps': 500,
        'every_n_epochs': 0,
        'prompts': [],
        'negative_prompts': [],
        'width': 1024,
        'height': 1024,
        'num_steps': 8,
        'guidance_scale': 0.0,
        'sampler': 'euler',
        'seed': 42,
    },

    'general': {
        'mode': 'lora',            # hint for default strategy. Per-component strategy overrides.
    },

    'model': {
        'path': '',                # shortcut: single file (auto-wrapped as 1 component)
        'dtype': 'float16',
        'base_dir': '',
        'components': [],          # THE primary way to define models
    },

    'model_spec': None,            # shortcut: build from model builder spec

    # Global fallbacks — used by components that don't specify their own
    'finetune': {
        'unfreeze_patterns': [],
    },
    'lora': {
        'rank': 16,
        'alpha': 16,
        'dropout': 0.0,
        'init_reversed': True,
        'target_layers': [],
        'target_patterns': [],
        'target_components': [],   # which component names get LoRA by default
        'conv_rank': None,    # None = don't train conv layers
        'conv_alpha': None,   # None = fall back to conv_rank
    },

    'dataset': {
        'path': '',
        'builtin': None,
        'batch_size': 1,
        'num_workers': 4,
        'center_crop': True,
        'random_flip': 0.5,
        'buckets': None,
        'validation_split': 0.1,
    },

    'training': {
        'epochs': 10,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'mixed_precision': 'fp16',
        'gradient_checkpointing': True,
        'save_every_n_steps': 500,
        'save_every_n_epochs': 0,
        'eval_every_n_steps': 100,
        'seed': 42,
        'keep_last_n_checkpoints': 5,
        'save_on_interrupt': True,
    },

    'optimizer': {
        'name': 'adamw',
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
    },

    'scheduler': {
        'name': 'cosine_warmup',
        'warmup_steps': 100,
        'total_steps': None,
    },

    'loss': {
        'name': 'mse',
        'params': {},
    },

    'logging': {
        'tensorboard': True,
        'tensorboard_dir': os.environ.get('TB_LOG_DIR', '/data/logs/tensorboard'),
        'json_log': True,
        'json_log_path': os.environ.get('JSON_LOG_PATH', '/data/logs/training_log.json'),
        'print_every': 10,
        'run_name': None,
    },

    'output': {
        'dir': './outputs',
        'save_format': 'safetensors',
    },

    'optuna': {
        'enabled': False,
        'n_trials': 20,
        'direction': 'minimize',
        'pruner': 'median',
        'sampler': 'tpe',
    },
}


# ═══════════════════════════════════════════════════════════════
# Component normalization
# ═══════════════════════════════════════════════════════════════

COMPONENT_TRAINING_DEFAULTS = {
    'strategy': None,
    'lr': None,
    'weight_decay': None,
    'max_grad_norm': None,
    'freeze_epochs': 0,
    'unfreeze_patterns': [],
    'lora': None,
}

COMPONENT_FORWARD_DEFAULTS = {
    'input_key': None,
    'output_key': None,
    'no_grad': False,
    'cache_output': False,
    'dtype_override': None,
}


def normalize_component_config(comp: dict) -> dict:
    """Ensure component config has all fields with defaults."""
    comp.setdefault('source', 'file')
    comp.setdefault('execution_order', 0)
    comp.setdefault('training', {})
    comp.setdefault('forward', {})
    for k, v in COMPONENT_TRAINING_DEFAULTS.items():
        comp['training'].setdefault(k, copy.deepcopy(v) if isinstance(v, (list, dict)) else v)
    for k, v in COMPONENT_FORWARD_DEFAULTS.items():
        comp['forward'].setdefault(k, v)
    return comp


# ═══════════════════════════════════════════════════════════════
# Config operations
# ═══════════════════════════════════════════════════════════════

def deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML required. pip install pyyaml")
    elif ext == '.json':
        with open(path) as f:
            user_config = json.load(f)
    else:
        raise ValueError(f"Unsupported: {ext}. Use .yaml or .json")
    return deep_merge(DEFAULT_CONFIG, user_config)


def save_config(config: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            json_path = path.rsplit('.', 1)[0] + '.json'
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def validate_config(config: Dict[str, Any]) -> list:
    errors = []

    has_path = bool(config.get('model', {}).get('path'))
    has_spec = bool(config.get('model_spec'))
    has_comps = bool(config.get('model', {}).get('components'))

    if not has_path and not has_comps and not has_spec:
        errors.append("No model source. Provide model.path, model.components, or model_spec.")

    valid_strategies = ('lora', 'finetune', 'full', 'frozen', 'adapter', None)

    if has_comps:
        components = config.get('model', {}).get('components', [])
        names_seen = set()
        for i, comp in enumerate(components):
            name = comp.get('name')
            if not name:
                errors.append(f"components[{i}]: name required")
            elif name in names_seen:
                errors.append(f"components[{i}]: duplicate name '{name}'")
            else:
                names_seen.add(name)

            source = comp.get('source', 'file')
            if source == 'file' and not comp.get('path'):
                errors.append(f"components[{i}]: path required for source='file'")
            elif source == 'spec' and not comp.get('spec_name'):
                errors.append(f"components[{i}]: spec_name required for source='spec'")

            strategy = (comp.get('training') or {}).get('strategy')
            if strategy and strategy not in valid_strategies:
                errors.append(f"components[{i}]: strategy must be one of {valid_strategies}")

            if strategy == 'lora':
                comp_lora = (comp.get('training') or {}).get('lora')
                if comp_lora:
                    if comp_lora.get('rank', 1) <= 0:
                        errors.append(f"components[{i}]: lora.rank must be positive")
                    if not comp_lora.get('target_patterns') and not comp_lora.get('target_layers'):
                        errors.append(f"components[{i}]: lora needs target_patterns or target_layers")

    ds = config.get('dataset', {})
    if not ds.get('path') and not ds.get('builtin'):
        errors.append("dataset.path or dataset.builtin required")

    if config.get('training', {}).get('epochs', 1) <= 0:
        errors.append("training.epochs must be positive")

    if config.get('optimizer', {}).get('lr', 0.001) <= 0:
        errors.append("optimizer.lr must be positive")

    return errors


def get_flat_config(config: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    flat.update(config.get('training', {}))
    flat['output_dir'] = config.get('output', {}).get('dir', './outputs')
    return flat


class ConfigManager:
    def __init__(self, configs_dir: str = './saved_configs'):
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def list_configs(self) -> list:
        configs = []
        for f in self.configs_dir.iterdir():
            if f.suffix in ('.yaml', '.yml', '.json'):
                configs.append({'name': f.stem, 'path': str(f), 'format': f.suffix})
        return sorted(configs, key=lambda x: x['name'])

    def save(self, name: str, config: Dict[str, Any], format: str = 'json'):
        ext = '.json' if format == 'json' else '.yaml'
        path = self.configs_dir / f"{name}{ext}"
        save_config(config, str(path))
        return str(path)

    def load(self, name: str) -> Dict[str, Any]:
        for ext in ('.json', '.yaml', '.yml'):
            path = self.configs_dir / f"{name}{ext}"
            if path.exists():
                return load_config(str(path))
        raise FileNotFoundError(f"Config '{name}' not found")

    def delete(self, name: str) -> bool:
        for ext in ('.json', '.yaml', '.yml'):
            path = self.configs_dir / f"{name}{ext}"
            if path.exists():
                path.unlink()
                return True
        return False

    def get_default(self) -> Dict[str, Any]:
        return copy.deepcopy(DEFAULT_CONFIG)
