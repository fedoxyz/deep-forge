"""Routes for saved hyperparameter search configurations."""

import json
import os

from fastapi import APIRouter, HTTPException

from backend.api.models import HparamConfigSave
from backend.api.dependencies import HPARAM_CONFIGS_DIR

router = APIRouter(prefix="/api/hparam_configs", tags=["hparam_configs"])


@router.get("")
async def list_hparam_configs():
    configs = [
        {"name": f[:-5]}
        for f in sorted(os.listdir(HPARAM_CONFIGS_DIR))
        if f.endswith(".json")
    ]
    return {"configs": configs}


@router.get("/{name}")
async def get_hparam_config(name: str):
    path = os.path.join(HPARAM_CONFIGS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404)
    with open(path) as f:
        return {"name": name, "config": json.load(f)}


@router.post("")
async def save_hparam_config(req: HparamConfigSave):
    path = os.path.join(HPARAM_CONFIGS_DIR, f"{req.name}.json")
    with open(path, "w") as f:
        json.dump(req.config, f, indent=2)
    return {"name": req.name, "path": path}


@router.delete("/{name}")
async def delete_hparam_config(name: str):
    path = os.path.join(HPARAM_CONFIGS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.unlink(path)
        return {"deleted": True}
    raise HTTPException(404)
