"""Routes for inspecting pretrained model files and components."""

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.api.models import InspectRequest

router = APIRouter(prefix="/api/model", tags=["model"])


@router.post("/inspect")
async def inspect_model(req: InspectRequest):
    from backend.modules.model_registry import load_state_dict_from_file, analyze_state_dict

    if not os.path.exists(req.model_path):
        raise HTTPException(404)
    try:
        return analyze_state_dict(load_state_dict_from_file(req.model_path))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/presets")
async def get_lora_presets():
    from backend.modules.model_registry import LORA_TARGET_PRESETS
    return {"presets": LORA_TARGET_PRESETS}

