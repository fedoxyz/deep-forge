"""Routes for Optuna hyperparameter search: start, status, streaming, results."""

import asyncio
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

from backend.api.models import OptunaRequest
from backend.api.state.training_state import training_state, optuna_state
from backend.api.dependencies import get_config_manager
from backend.api.tasks.optuna_search import run_optuna_task

router = APIRouter(prefix="/api/optuna", tags=["optuna"])
config_manager = get_config_manager()

DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"type": "float_log", "low": 1e-5, "high": 1e-2},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1, "step": 0.01},
    "lora_rank": {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
    "lora_alpha": {"type": "categorical", "choices": [1, 4, 8, 16, 32, 64]},
    "scheduler": {"type": "categorical", "choices": ["cosine_warmup", "constant_warmup", "one_cycle"]},
}


@router.post("/start")
async def start_optuna(req: OptunaRequest, bg: BackgroundTasks):
    if optuna_state["status"] == "training":
        raise HTTPException(409, "Optuna search in progress")
    if training_state["status"] == "training":
        raise HTTPException(409, "Training in progress")

    config = _resolve_config(req)

    optuna_state.update({
        "status": "training",
        "current_trial": 0,
        "n_trials": req.n_trials,
        "best_value": None,
        "best_params": None,
        "trials": [],
        "start_time": datetime.now().isoformat(),
        "error": None,
        "search_space": req.search_space,
        "mode": req.mode,
    })
    bg.add_task(run_optuna_task, config, req.n_trials, req.mode, req.search_space)
    return {"status": "started", "n_trials": req.n_trials}


@router.get("/default_space")
async def optuna_default_space():
    """Return the default search space definition so the frontend can show/edit it."""
    return {"search_space": DEFAULT_SEARCH_SPACE}


@router.get("/status")
async def get_optuna_status():
    return optuna_state


@router.get("/stream")
async def optuna_stream():
    async def generate():
        last_trial = -1
        while True:
            if optuna_state["current_trial"] != last_trial or optuna_state["status"] != "training":
                yield f"data: {json.dumps(optuna_state, default=str)}\n\n"
                last_trial = optuna_state["current_trial"]
            if optuna_state["status"] in ("completed", "error", "idle"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/results")
async def get_optuna_results():
    path = "./logs/optuna/optuna_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"best_trial": None, "all_trials": []}


# ── Helpers ──

def _resolve_config(req: OptunaRequest) -> dict:
    if req.config_name:
        return config_manager.load(req.config_name)
    elif req.config:
        return req.config
    raise HTTPException(400, "Provide config_name or config")
