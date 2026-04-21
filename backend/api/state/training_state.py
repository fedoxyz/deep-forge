"""
Shared mutable state for training and Optuna hyperparameter search.

Kept in a dedicated module so routes and background tasks can import
the same dictionaries without circular dependencies.
"""

import threading

stop_event = threading.Event()

training_state: dict = {
    "status": "idle",
    "current_step": 0,
    "total_steps": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "smoothed_loss": 0.0,
    "val_loss": None,
    "lr": 0.0,
    "start_time": None,
    "error": None,
    "config_name": None,
    "loss_history": [],
    "val_loss_history": [],
    "lr_history": [],
    "accuracy": None,
    "val_accuracy": None,
    "mode": None,
    "checkpoints": [],
    "last_checkpoint": None, 
    "run_dir": None,
    "samples": [],           # List of sample generation results
    "last_sample_step": 0,
    "_trainer_ref": None,    # Reference to active trainer (not serialized)
    "checkpoint_path": None, # For resume
}

optuna_state: dict = {
    "status": "idle",
    "current_trial": 0,
    "n_trials": 0,
    "best_value": None,
    "best_params": None,
    "trials": [],
    "start_time": None,
    "error": None,
    "search_space": None,
    "mode": None,
}

def reset_stop_event():
    stop_event.clear()
