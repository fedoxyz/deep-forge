from typing import Any, Dict, Optional
from backend.core.lora import LoRAInjector  # your existing LoRA class


def setup_per_component_lora(
    bundle,
    global_lora_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inject LoRA into components that request it.
    
    Returns dict of {component_name: LoRAInjector}
    """
    injectors = {}
    
    for comp in bundle:
        comp_lora = (comp.config.get('training', {}) or {}).get('lora')
        
        if comp_lora is None and not comp.config.get('trainable', False):
            continue
        
        if comp.module is None:
            continue
            
        if comp_lora is not None:
            # Per-component LoRA config
            injector = LoRAInjector(
                model=comp.module,
                rank=comp_lora.get('rank', global_lora_config.get('rank', 16)),
                alpha=comp_lora.get('alpha', global_lora_config.get('alpha', 16)),
                dropout=comp_lora.get('dropout', global_lora_config.get('dropout', 0.0)),
                target_patterns=comp_lora.get('target_patterns', []),
                conv_rank=comp_lora.get('conv_rank', global_lora_config.get('conv_rank', None)),
                conv_alpha=comp_lora.get('conv_alpha', global_lora_config.get('conv_alpha', None)),
            )
            injectors[comp.name] = injector
    
    return injectors
