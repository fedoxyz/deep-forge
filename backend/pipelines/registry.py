"""
Pipeline registry.

Pipelines register themselves here. The training task looks up the pipeline
by name from the config's `pipeline.name` field.
"""

from typing import Dict, Type, List, Optional

_REGISTRY: Dict[str, Type] = {}


def register_pipeline(name: str, cls: Type):
    """Register a pipeline class under a name."""
    _REGISTRY[name] = cls


def get_pipeline(name: str) -> Type:
    """Get pipeline class by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_pipelines() -> List[Dict]:
    """List all registered pipelines with metadata."""
    result = []
    for name, cls in _REGISTRY.items():
        result.append({
            'name': name,
            'display_name': getattr(cls, 'display_name', name),
            'description': getattr(cls, 'description', ''),
            'uses_flow_matching': getattr(cls, 'uses_flow_matching', False),
            'default_num_steps': getattr(cls, 'default_num_steps', 20),
            'default_guidance_scale': getattr(cls, 'default_guidance_scale', 7.5),
            'default_sampler': getattr(cls, 'default_sampler', 'euler'),
        })
    return result
