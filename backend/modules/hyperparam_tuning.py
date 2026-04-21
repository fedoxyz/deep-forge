"""
Optuna-based hyperparameter tuning module.

Searches over:
- Learning rate
- LoRA rank & alpha
- Optimizer settings
- Scheduler settings
- Loss function parameters
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional

import torch


def create_optuna_study(
    objective_fn: Callable,
    n_trials: int = 20,
    direction: str = 'minimize',
    study_name: str = 'lora_hparam_search',
    storage: Optional[str] = None,
    pruner: Optional[str] = 'median',
    sampler: Optional[str] = 'tpe',
):
    """
    Create and run an Optuna study.

    Args:
        objective_fn: Function that takes an optuna.Trial and returns the metric to optimize
        n_trials: Number of trials
        direction: 'minimize' or 'maximize'
        study_name: Name for the study
        storage: Optional database URL for persistent storage
        pruner: Pruner type ('median', 'hyperband', 'none')
        sampler: Sampler type ('tpe', 'random', 'cmaes')
    """
    import optuna

    # Setup pruner
    pruner_obj = None
    if pruner == 'median':
        pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner == 'hyperband':
        pruner_obj = optuna.pruners.HyperbandPruner()

    # Setup sampler
    sampler_obj = None
    if sampler == 'tpe':
        sampler_obj = optuna.samplers.TPESampler()
    elif sampler == 'random':
        sampler_obj = optuna.samplers.RandomSampler()
    elif sampler == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        pruner=pruner_obj,
        sampler=sampler_obj,
        load_if_exists=True,
    )

    study.optimize(objective_fn, n_trials=n_trials)
    return study


def default_lora_search_space(trial) -> Dict[str, Any]:
    """
    Default hyperparameter search space for LoRA training.

    Returns a dict of suggested hyperparameters.
    """
    import optuna

    params = {
        # LoRA params
        'lora_rank': trial.suggest_categorical('lora_rank', [4, 8, 16, 32, 64]),
        'lora_alpha': trial.suggest_categorical('lora_alpha', [1, 4, 8, 16, 32, 64]),
        'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.3, step=0.05),

        # Training params
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
        'batch_size': trial.suggest_categorical('batch_size', [1, 2, 4]),
        'gradient_accumulation_steps': trial.suggest_categorical(
            'gradient_accumulation_steps', [1, 2, 4, 8]
        ),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0, step=0.5),

        # Scheduler
        'scheduler': trial.suggest_categorical(
            'scheduler', ['cosine_warmup', 'constant_warmup', 'one_cycle']
        ),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 500, step=50),

        # Loss
        'loss_fn': trial.suggest_categorical('loss_fn', ['mse', 'huber', 'snr_weighted']),
    }

    return params


def save_study_results(study, output_dir: str):
    """Save Optuna study results to JSON."""
    import optuna

    os.makedirs(output_dir, exist_ok=True)

    results = {
        'best_trial': {
            'number': study.best_trial.number,
            'value': study.best_trial.value,
            'params': study.best_trial.params,
        },
        'all_trials': [],
    }

    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value if trial.value is not None else None,
            'state': str(trial.state),
            'params': trial.params,
        }
        results['all_trials'].append(trial_data)

    output_path = os.path.join(output_dir, 'optuna_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Optuna] Results saved to {output_path}")
    print(f"[Optuna] Best trial #{study.best_trial.number}: "
          f"value={study.best_trial.value:.6f}")
    print(f"[Optuna] Best params: {json.dumps(study.best_trial.params, indent=2)}")

    return results
