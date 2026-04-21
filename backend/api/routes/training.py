"""Routes for training lifecycle: start, stop, status, streaming, logs."""

import asyncio
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

from backend.api.models import TrainRequest, SampleRequest, SaveRequest
from backend.api.state.training_state import training_state, stop_event
from backend.api.dependencies import get_config_manager
from backend.api.tasks.train import run_unified_training
from backend.api.state.log_buffer import log_buffer
from backend.api.state.run_registry import list_runs, get_run, scan_run_dir

router = APIRouter(prefix="/api/training", tags=["training"])
config_manager = get_config_manager()

@router.get("/runs")
async def get_runs():
    """List all training runs from registry."""
    runs = list_runs()
    return {"runs": runs}

@router.get("/runs/{run_name}")
async def get_run_detail(run_name: str):
    run = get_run(run_name)
    if not run:
        raise HTTPException(404, f"Run '{run_name}' not found")
    # If checkpoints list is empty, try scanning disk (crash recovery)
    if not run["checkpoints"] and run.get("run_dir"):
        run["checkpoints"] = scan_run_dir(run["run_dir"])
    return run

@router.get("/runs/{run_name}/checkpoints")
async def get_run_checkpoints(run_name: str):
    run = get_run(run_name)
    if not run:
        raise HTTPException(404)
    checkpoints = run.get("checkpoints", [])
    if not checkpoints and run.get("run_dir"):
        checkpoints = scan_run_dir(run["run_dir"])
    return {"checkpoints": checkpoints, "run_name": run_name}

@router.get("/status")
async def get_training_status():
    return {k: v for k, v in training_state.items() if not k.startswith("_")}


@router.post("/start")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    if training_state["status"] == "training":
        raise HTTPException(409, "Training in progress")

    config = _resolve_config(req)

    stop_event.clear()
    training_state.update({
        "status": "training",
        "current_step": 0,
        "current_epoch": 0,
        "total_epochs": config.get("training", {}).get("epochs", 10),
        "loss": 0.0,
        "smoothed_loss": 0.0,
        "val_loss": None,
        "accuracy": None,
        "val_accuracy": None,
        "lr": config.get("optimizer", {}).get("lr", 1e-4),
        "start_time": datetime.now().isoformat(),
        "error": None,
        "config_name": req.config_name,
        "mode": req.mode,
        "run_dir": None,
        "loss_history": [],
        "val_loss_history": [],
        "lr_history": [],
    })
    bg.add_task(run_unified_training, config, req.mode)
    return {"status": "started", "mode": req.mode}

@router.post("/stop")
async def stop_training():
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")
    stop_event.set()
    training_state["status"] = "stopping"
    
    # Optionally also poke the trainer directly as a fallback
    trainer = training_state.get("_trainer_ref")
    if trainer and hasattr(trainer, '_save_requested'):
        pass  # stop_event is sufficient, trainer checks it each step
    
    return {"status": "stopping"}

@router.get("/stream")
async def training_stream():
    async def generate():
        last_step = -1
        while True:
            if training_state["current_step"] != last_step or training_state["status"] != "training":
                yield f"data: {json.dumps(training_state, default=str)}\n\n"
                last_step = training_state["current_step"]
            if training_state["status"] in ("completed", "error", "idle"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/sample")
async def request_sample(req: SampleRequest = None):
    """Request sample generation at next step boundary."""
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")

    trainer = training_state.get("_trainer_ref")
    if trainer is None:
        raise HTTPException(400, "Trainer not initialized")

    request_data = None
    if req:
        request_data = req.dict(exclude_none=True)

    trainer.request_sample(request_data)
    return {"status": "sample_requested", "message": "Will generate at next step boundary"}


@router.post("/save_now")
async def request_save():
    """Request checkpoint save at next step boundary."""
    if training_state["status"] != "training":
        raise HTTPException(400, "Not currently training")

    trainer = training_state.get("_trainer_ref")
    if trainer is None:
        raise HTTPException(400, "Trainer not initialized")

    trainer.request_save()
    return {"status": "save_requested", "message": "Will save at next step boundary"}

@router.post("/resume")
async def resume_training(req: TrainRequest, bg: BackgroundTasks):
    if training_state["status"] == "training":
        raise HTTPException(409, "Training in progress")

    config = _resolve_config(req)
    checkpoint_path = req.checkpoint_path

    # If no explicit checkpoint, find it from the run registry
    if not checkpoint_path:
        if req.run_name:
            # Resume specific run by name
            run = get_run(req.run_name)
            if run and run.get("last_checkpoint"):
                checkpoint_path = run["last_checkpoint"]
        else:
            # Fall back to current session's last checkpoint
            checkpoint_path = training_state.get("last_checkpoint")

    if not checkpoint_path or not os.path.exists(
        checkpoint_path.replace('.pt', '_trainer.pt')
    ):
        raise HTTPException(400, "No valid checkpoint found. Provide run_name or checkpoint_path.")

    # Restore the run_name so TensorBoard continues on same log dir
    run_name = req.run_name or training_state.get("run_name")
    if run_name:
        config.setdefault("logging", {})
        config["logging"]["run_name"] = run_name

    stop_event.clear()
    training_state.update({
        "status": "training",
        "error": None,
        "mode": req.mode,
        "run_name": run_name,
        "checkpoints": [],  # will repopulate as training continues
    })

    bg.add_task(run_unified_training, config, req.mode,
                resume_checkpoint=checkpoint_path)
    return {"status": "resuming", "checkpoint": checkpoint_path, "run_name": run_name}

# In training_routes.py:
@router.get("/checkpoints")
async def list_checkpoints():
    """List all checkpoints from current or last run."""
    run_dir = training_state.get("run_dir")
    checkpoints = training_state.get("checkpoints", [])
    
    # If we have in-memory list, use it
    if checkpoints:
        return {"checkpoints": checkpoints, "run_dir": run_dir}
    
    # Fallback: scan disk for a resumed/previous run
    if run_dir:
        import glob
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        metas = sorted(glob.glob(os.path.join(ckpt_dir, "*_meta.json")))
        found = []
        for meta_path in metas:
            with open(meta_path) as f:
                import json
                found.append(json.load(f))
        return {"checkpoints": found, "run_dir": run_dir}
    
    return {"checkpoints": [], "run_dir": None}

@router.get("/samples")
async def get_samples():
    """Get list of generated samples."""
    trainer = training_state.get("_trainer_ref")
    if trainer is None or trainer.sampler is None:
        return {"samples": []}

    return {"samples": trainer.sampler.sample_log}


@router.get("/samples/latest")
async def get_latest_samples():
    """Get the most recent sample images."""
    trainer = training_state.get("_trainer_ref")
    if trainer is None or trainer.sampler is None:
        return {"sample": None}

    latest = trainer.sampler.get_latest_samples()
    return {"sample": latest}


@router.get("/pipelines")
async def list_available_pipelines():
    """List registered diffusion pipelines."""
    from backend.pipelines.registry import list_pipelines
    return {"pipelines": list_pipelines()}


@router.get("/pipelines/{name}/presets")
async def get_pipeline_presets(name: str):
    """Get LoRA presets for a pipeline."""
    if name == "zimage_turbo":
        from backend.pipelines.zimage_turbo_pipeline import ZIMAGE_TURBO_LORA_PRESETS
        return {"presets": ZIMAGE_TURBO_LORA_PRESETS}
    return {"presets": {}}

@router.get("/logs/stream")
async def stream_logs(since: int = 0):
    queue = asyncio.Queue(maxsize=500)
    
    # Subscribe FIRST before reading history — prevents the gap
    log_buffer.subscribe(queue)

    async def generate():
        try:
            # Send history the client hasn't seen yet
            history = log_buffer.get_history()
            sent = set()
            for i, line in enumerate(history[since:], start=since):
                sent.add(line)  # track to deduplicate queue
                yield f"data: {line}\n\n"

            # Stream live lines, skip any already sent from history
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=15.0)
                    if line in sent:
                        sent.discard(line)  # allow future identical lines
                        continue
                    yield f"data: {line}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if training_state["status"] in ("completed", "error", "idle"):
                        break
        finally:
            log_buffer.unsubscribe(queue)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

@router.get("/logs/history")
async def get_log_history():
    """Returns buffered log lines as JSON. For initial page load."""
    return {"lines": log_buffer.get_history()}

@router.get("/logs")
async def get_training_logs():
    path = "./logs/training_log.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"steps": [], "epochs": []}


# ── Helpers ──

def _resolve_config(req: TrainRequest) -> dict:
    if req.config_name:
        try:
            return config_manager.load(req.config_name)
        except FileNotFoundError:
            raise HTTPException(404)
    elif req.config:
        return req.config
    raise HTTPException(400, "Provide config_name or config")
