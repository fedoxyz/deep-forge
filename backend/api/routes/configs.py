"""Routes for saved training configurations."""

from fastapi import APIRouter, HTTPException
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

import json

from backend.api.models import ConfigCreate, ConfigUpdate
from backend.api.dependencies import get_config_manager

router = APIRouter(prefix="/api/configs", tags=["configs"])
config_manager = get_config_manager()


@router.get("")
async def list_configs():
    return {"configs": config_manager.list_configs()}


@router.get("/default")
async def get_default_config():
    return {"config": config_manager.get_default()}


@router.get("/{name}")
async def get_config(name: str):
    try:
        return {"config": config_manager.load(name), "name": name}
    except FileNotFoundError:
        raise HTTPException(404, f"Config '{name}' not found")


@router.post("")
async def create_config(req: ConfigCreate):
    path = config_manager.save(req.name, req.config)
    return {"name": req.name, "path": path}


@router.put("/{name}")
async def update_config(name: str, req: ConfigUpdate):
    path = config_manager.save(name, req.config)
    return {"name": name, "path": path}


@router.delete("/{name}")
async def delete_config(name: str):
    if config_manager.delete(name):
        return {"deleted": True}
    raise HTTPException(404)

@router.post("/import")
async def import_config(file: UploadFile = File(...)):
    content = await file.read()
    try:
        config = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    
    name = file.filename.replace('.json', '').replace('.yaml', '')
    path = config_manager.save(name, config)
    return {"name": name, "path": path, "config": config}
