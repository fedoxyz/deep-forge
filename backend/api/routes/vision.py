"""Routes for vision model management, captioning, and LLM concept extraction."""

import asyncio
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/api/vision", tags=["vision"])


# ── Request models ──

class LoadModelRequest(BaseModel):
    backend: str = "qwen3vl"
    model_id: str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None  # "flash_attention_2"

class CaptionSingleRequest(BaseModel):
    image_path: str
    prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7

class CaptionBatchRequest(BaseModel):
    dataset_id: str
    indices: Optional[List[int]] = None  # None = all uncaptioned
    prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7
    overwrite: bool = False  # overwrite existing captions?

class LLMConceptRequest(BaseModel):
    dataset_id: str
    categories: List[str] = ["attributes", "actions", "settings", "style", "composition"]
    prompt_template: Optional[str] = None
    max_new_tokens: int = 1024


# ── Captioning job state ──
_caption_job = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_file": "",
    "results": [],
    "error": None,
}

_load_job = {
    "running": False,
    "status": "idle",  # idle | loading | loaded | error
    "error": None,
    "info": None,
}


# ── Routes ──
@router.post("/load")
async def load_model(req: LoadModelRequest, background_tasks: BackgroundTasks):
    if _load_job["running"]:
        raise HTTPException(status_code=409, detail="A load job is already running")

    import torch
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(req.dtype, torch.bfloat16)
    kwargs = {"dtype": dtype}
    if req.attn_implementation:
        kwargs["attn_implementation"] = req.attn_implementation

    _load_job.update({"running": True, "status": "loading", "error": None, "info": None})
    background_tasks.add_task(_run_load_model, req.backend, req.model_id, kwargs)

    return {"status": "loading_started"}


def _run_load_model(backend_name: str, model_id: str, kwargs: dict):
    from backend.modules.vision_service import load_backend
    try:
        backend = load_backend(backend_name, model_id, **kwargs)
        _load_job.update({"running": False, "status": "loaded", "info": backend.get_info()})
    except Exception as e:
        _load_job.update({"running": False, "status": "error", "error": str(e)})


@router.post("/unload")
async def unload_model():
    from backend.modules.vision_service import unload_backend
    unload_backend()
    return {"status": "unloaded"}

@router.get("/load/status")
async def load_status():
    return {
        "running": _load_job["running"],
        "status": _load_job["status"],
        "error": _load_job["error"],
        "info": _load_job["info"],
    }

@router.get("/status")
async def model_status():
    from backend.modules.vision_service import get_backend
    b = get_backend()
    if b and b.is_loaded():
        return {"loaded": True, **b.get_info()}
    return {"loaded": False}


@router.post("/caption")
async def caption_single(req: CaptionSingleRequest):
    """Caption a single image."""
    from backend.modules.caption_pipelines import caption_single as _caption, DEFAULT_CAPTION_PROMPT
    prompt = req.prompt or DEFAULT_CAPTION_PROMPT
    try:
        caption = _caption(req.image_path, prompt, req.max_new_tokens, req.temperature)
        return {"caption": caption}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/caption-batch")
async def caption_batch(req: CaptionBatchRequest, background_tasks: BackgroundTasks):
    """Start batch captioning as a background job."""
    from backend.datasets.dataset_manager import get_loaded_dataset
    from backend.modules.caption_pipelines import DEFAULT_CAPTION_PROMPT

    ds = get_loaded_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    if _caption_job["running"]:
        raise HTTPException(status_code=409, detail="A captioning job is already running")

    # Determine which images to caption
    if req.indices is not None:
        target_indices = [i for i in req.indices if 0 <= i < len(ds.entries)]
    elif req.overwrite:
        target_indices = list(range(len(ds.entries)))
    else:
        target_indices = [
            i for i, e in enumerate(ds.entries)
            if not e.has_caption_file or not e.caption
        ]

    if not target_indices:
        return {"status": "nothing_to_caption", "count": 0}

    prompt = req.prompt or DEFAULT_CAPTION_PROMPT

    _caption_job.update({
        "running": True, "progress": 0, "total": len(target_indices),
        "current_file": "", "results": [], "error": None,
    })

    background_tasks.add_task(
        _run_caption_batch, req.dataset_id, target_indices, prompt,
        req.max_new_tokens, req.temperature,
    )

    return {"status": "started", "count": len(target_indices)}


def _run_caption_batch(dataset_id, indices, prompt, max_tokens, temperature):
    """Background task for batch captioning."""
    from backend.datasets.dataset_manager import get_loaded_dataset, update_caption
    from backend.modules.caption_pipelines import caption_single

    ds = get_loaded_dataset(dataset_id)
    if not ds:
        _caption_job["running"] = False
        _caption_job["error"] = "Dataset not found"
        return

    for i, idx in enumerate(indices):
        entry = ds.entries[idx]
        _caption_job["current_file"] = entry.filename
        _caption_job["progress"] = i

        try:
            cap = caption_single(entry.image_path, prompt, max_tokens, temperature)
            update_caption(dataset_id, idx, cap)
            _caption_job["results"].append({
                "index": idx, "filename": entry.filename, "caption": cap, "error": None,
            })
        except Exception as e:
            _caption_job["results"].append({
                "index": idx, "filename": entry.filename, "caption": "", "error": str(e),
            })

    _caption_job["progress"] = len(indices)
    _caption_job["running"] = False


@router.get("/caption-batch/status")
async def caption_batch_status():
    """Poll captioning job progress."""
    return {
        "running": _caption_job["running"],
        "progress": _caption_job["progress"],
        "total": _caption_job["total"],
        "current_file": _caption_job["current_file"],
        "completed": len(_caption_job["results"]),
        "error": _caption_job["error"],
    }


@router.get("/caption-batch/results")
async def caption_batch_results():
    """Get results from the last/current captioning job."""
    return {"results": _caption_job["results"]}


@router.post("/caption-batch/stop")
async def caption_batch_stop():
    """Signal the batch job to stop (checked between images)."""
    # For now we can't interrupt mid-generation, but we set a flag
    _caption_job["running"] = False
    return {"status": "stop_requested"}


@router.post("/extract-concepts")
async def extract_concepts_llm(req: LLMConceptRequest):
    """Use LLM to extract structured concepts from dataset captions."""
    from backend.datasets.dataset_manager import get_loaded_dataset
    from backend.modules.caption_pipelines import extract_concepts_llm as _extract
    from backend.modules.vision_service import ConceptExtractionRequest

    ds = get_loaded_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    captions = [e.caption for e in ds.entries if e.caption]
    if not captions:
        return {"concepts": [], "message": "No captions to analyze"}

    try:
        concepts = _extract(
            ConceptExtractionRequest(
                captions=captions,
                categories=req.categories,
                prompt_template=req.prompt_template,
            ),
            max_new_tokens=req.max_new_tokens,
        )
        return {"concepts": concepts, "total_captions": len(captions)}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
