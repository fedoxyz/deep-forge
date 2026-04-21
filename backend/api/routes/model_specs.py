"""Routes for saved model spec architectures."""

import json
import os

from fastapi import APIRouter, HTTPException

from backend.api.models import ModelSpecSave
from backend.api.dependencies import MODEL_SPECS_DIR

router = APIRouter(prefix="/api/model_specs", tags=["model_specs"])


@router.get("")
async def list_model_specs():
    specs = [
        {"name": f[:-5], "path": os.path.join(MODEL_SPECS_DIR, f)}
        for f in sorted(os.listdir(MODEL_SPECS_DIR))
        if f.endswith(".json")
    ]
    return {"specs": specs}


@router.get("/{name}")
async def get_model_spec(name: str):
    path = os.path.join(MODEL_SPECS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404)
    with open(path) as f:
        return {"name": name, "spec": json.load(f)}


@router.post("")
async def save_model_spec(req: ModelSpecSave):
    path = os.path.join(MODEL_SPECS_DIR, f"{req.name}.json")
    with open(path, "w") as f:
        json.dump(req.spec, f, indent=2)
    return {"name": req.name, "path": path}


@router.delete("/{name}")
async def delete_model_spec(name: str):
    path = os.path.join(MODEL_SPECS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.unlink(path)
        return {"deleted": True}
    raise HTTPException(404)
