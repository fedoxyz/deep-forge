"""
Persistent run registry. Each training run gets a record saved to disk.
Survives server restarts. Single source of truth for all runs.
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

RUNS_DIR = os.environ.get('RUNS_DIR', './outputs')
REGISTRY_FILE = os.path.join(RUNS_DIR, 'run_registry.json')


def _load_registry() -> Dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": {}}


def _save_registry(registry: Dict):
    os.makedirs(RUNS_DIR, exist_ok=True)
    tmp = REGISTRY_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    os.replace(tmp, REGISTRY_FILE)  # atomic write


def create_run(run_name: str, config_name: Optional[str], config: Dict, mode: str) -> Dict:
    """Register a new run. Called when training starts."""
    registry = _load_registry()
    run = {
        "run_name": run_name,
        "config_name": config_name,
        "mode": mode,
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "run_dir": os.path.join(RUNS_DIR, run_name),
        "checkpoints": [],          # {tag, path, step, epoch, timestamp}
        "last_checkpoint": None,
        "error": None,
        "total_steps": 0,
        "total_epochs": config.get("training", {}).get("epochs", 0),
    }
    registry["runs"][run_name] = run
    _save_registry(registry)
    return run


def update_run(run_name: str, **kwargs):
    """Patch fields on an existing run record."""
    registry = _load_registry()
    if run_name not in registry["runs"]:
        return
    registry["runs"][run_name].update(kwargs)
    _save_registry(registry)


def add_checkpoint(run_name: str, tag: str, path: str, step: int, epoch: int):
    """Record a checkpoint on a run."""
    registry = _load_registry()
    if run_name not in registry["runs"]:
        return
    entry = {
        "tag": tag,
        "path": path,
        "step": step,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
    }
    registry["runs"][run_name]["checkpoints"].append(entry)
    registry["runs"][run_name]["last_checkpoint"] = path
    _save_registry(registry)


def get_run(run_name: str) -> Optional[Dict]:
    return _load_registry()["runs"].get(run_name)


def list_runs() -> List[Dict]:
    """Return all runs, newest first."""
    registry = _load_registry()
    runs = list(registry["runs"].values())
    return sorted(runs, key=lambda r: r.get("started_at", ""), reverse=True)


def get_latest_run() -> Optional[Dict]:
    runs = list_runs()
    return runs[0] if runs else None


def scan_run_dir(run_dir: str) -> List[Dict]:
    """
    Rebuild checkpoint list from disk for a run whose registry entry is missing.
    Used for recovery after server crash.
    """
    checkpoints = []
    pattern = os.path.join(run_dir, "checkpoints", "*_meta.json")
    for meta_path in sorted(glob.glob(pattern)):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Reconstruct the .pt path from the meta path
            pt_path = meta_path.replace('_meta.json', '.pt')
            checkpoints.append({
                "tag": meta.get("tag", ""),
                "path": pt_path,
                "step": meta.get("global_step", 0),
                "epoch": meta.get("epoch", 0),
                "timestamp": meta.get("timestamp", ""),
            })
        except Exception:
            pass
    return checkpoints
