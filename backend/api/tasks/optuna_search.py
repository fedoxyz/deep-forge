"""Background task: Optuna hyperparameter search with configurable search space."""

import json
import traceback
from typing import Any, Dict, Optional

from backend.api.state.training_state import training_state, optuna_state


DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"type": "float_log", "low": 1e-5, "high": 1e-2},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
}


def run_optuna_task(
    config: Dict[str, Any],
    n_trials: int,
    mode: str,
    search_space: Optional[Dict[str, Any]] = None,
):
    try:
        import torch
        import optuna
        from backend.modules.hyperparam_tuning import create_optuna_study, save_study_results

        space = search_space or DEFAULT_SEARCH_SPACE

        def objective(trial):
            optuna_state["current_trial"] = trial.number + 1
            training_state["current_step"] = trial.number + 1
            training_state["current_epoch"] = trial.number + 1

            params = {k: _suggest_param(trial, k, v) for k, v in space.items()}
            trial_config = _build_trial_config(config, params)
            val_loss = _run_single_trial(trial_config, params)

            _record_trial(trial, params, val_loss)
            _maybe_prune(trial, val_loss)

            return val_loss

        study = create_optuna_study(objective_fn=objective, n_trials=n_trials)
        save_study_results(study, "./logs/optuna")

        optuna_state.update({
            "status": "completed",
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
        })
        training_state["status"] = "completed"
        training_state["loss"] = study.best_trial.value

    except Exception as e:
        optuna_state["status"] = "error"
        optuna_state["error"] = str(e)
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _suggest_param(trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float_log":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    return spec.get("default", 0)


def _build_trial_config(base_config: dict, params: dict) -> dict:
    """Deep-copy base config and override with trial hyperparameters."""
    tc = json.loads(json.dumps(base_config))

    tc.setdefault("optimizer", {})["lr"] = params.get("learning_rate", 1e-3)
    tc.setdefault("dataset", {})["batch_size"] = int(params.get("batch_size", 64))
    tc.setdefault("training", {})["epochs"] = tc.get("training", {}).get("optuna_epochs", 3)

    if "weight_decay" in params:
        tc["optimizer"]["weight_decay"] = params["weight_decay"]

    # Patch dropout layers in model spec
    if tc.get("model_spec"):
        for layer in tc["model_spec"].get("layers", []):
            if layer.get("type") == "dropout" and "dropout" in params:
                layer["p"] = params["dropout"]

    return tc


def _run_single_trial(trial_config: dict, params: dict) -> float:
    """Train a single trial and return the best validation loss."""
    import torch
    from backend.core.model_builder import build_model
    from backend.core.unified_trainer import UnifiedTrainer
    from backend.modules.optimizers import create_optimizer
    from backend.datasets.builtin_datasets import create_builtin_dataloaders

    model_spec = trial_config.get("model_spec")
    if not model_spec:
        return float("inf")

    model = build_model(model_spec)
    ds_name = trial_config.get("dataset", {}).get("builtin", "mnist")
    batch_size = int(params.get("batch_size", 64))
    train_dl, val_dl, _ = create_builtin_dataloaders(ds_name, batch_size=batch_size, val_split=0.1)

    class CrossEntropyWrapper:
        def __init__(self):
            self.fn = torch.nn.CrossEntropyLoss()

        def compute(self, predictions, targets, **kwargs):
            return self.fn(predictions, targets)

    optimizer = create_optimizer(
        trial_config.get("optimizer", {}).get("name", "adamw"),
        list(model.parameters()),
        lr=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 0.01),
    )

    trainer = UnifiedTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        loss_fn=CrossEntropyWrapper(),
        config=trial_config.get("training", {}),
        full_config=trial_config,
        mode="train_custom",
    )
    result = trainer.train()

    val_losses = [v["loss"] for v in result.get("val_losses", []) if "loss" in v]
    return min(val_losses) if val_losses else float("inf")


def _record_trial(trial, params: dict, val_loss: float):
    """Update shared state with trial results."""
    training_state["loss"] = val_loss
    training_state["smoothed_loss"] = val_loss
    training_state["loss_history"].append({
        "step": trial.number + 1,
        "loss": val_loss,
        "smoothed": val_loss,
    })

    optuna_state["trials"].append({
        "number": trial.number,
        "value": val_loss,
        "params": params,
        "state": "complete",
    })

    if optuna_state["best_value"] is None or val_loss < optuna_state["best_value"]:
        optuna_state["best_value"] = val_loss
        optuna_state["best_params"] = params


def _maybe_prune(trial, val_loss: float):
    import optuna
    if val_loss != float("inf"):
        trial.report(val_loss, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
