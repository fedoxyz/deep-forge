# New file or add to a utils module, e.g. backend/core/optimizer_builder.py

from typing import Any, Dict, List, Optional
import torch


def build_param_groups(
    bundle,          # ComponentBundle
    global_config: Dict[str, Any],
    lora_injectors: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build optimizer param groups with per-component overrides.
    
    Returns list suitable for torch.optim.AdamW(param_groups, lr=default_lr)
    """
    global_lr = global_config.get('optimizer', {}).get('lr', 1e-4)
    global_wd = global_config.get('optimizer', {}).get('weight_decay', 0.01)
    
    param_groups = []
    
    for comp in bundle:
        comp_training = comp.config.get('training', {}) if hasattr(comp, 'config') else {}
        
        # Determine which parameters to collect
        params = []
        
        # Check if this component has LoRA
        if lora_injectors and comp.name in lora_injectors:
            params = list(lora_injectors[comp.name].get_trainable_parameters())
        elif comp.module is not None:
            params = [p for p in comp.module.parameters() if p.requires_grad]
        
        if not params:
            continue
        
        group = {
            'params': params,
            'lr': comp_training.get('lr') or global_lr,
            'weight_decay': comp_training.get('wd') or comp_training.get('weight_decay') or global_wd,
            '_component_name': comp.name,  # metadata for logging
            '_max_grad_norm': comp_training.get('max_grad_norm'),  # per-component clip
            '_freeze_epochs': comp_training.get('freeze_epochs', 0),
        }
        param_groups.append(group)
    
    return param_groups
