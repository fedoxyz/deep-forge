"""Pydantic request and response models for the LoRA Forge API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ConfigCreate(BaseModel):
    name: str
    config: Dict[str, Any]


class ConfigUpdate(BaseModel):
    config: Dict[str, Any]

class TrainRequest(BaseModel):
    config_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    config_override: Optional[dict] = None
    mode: str = "lora"
    checkpoint_path: Optional[str] = None
    run_name: Optional[str] = None

class SampleRequest(BaseModel):
    prompts: Optional[List[str]] = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 8
    guidance_scale: float = 0.0
    sampler: str = "euler"
    seed: Optional[int] = 42

class SaveRequest(BaseModel):
    tag: Optional[str] = None

class ModelBuildRequest(BaseModel):
    spec: Dict[str, Any]


class ModelValidateRequest(BaseModel):
    spec: Dict[str, Any]
    input_shape: Optional[List[int]] = None


class InspectRequest(BaseModel):
    model_path: str


class ModelSpecSave(BaseModel):
    name: str
    spec: Dict[str, Any]


class OptunaRequest(BaseModel):
    config_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    n_trials: int = 20
    mode: str = "train_custom"
    search_space: Optional[Dict[str, Any]] = None


class HparamConfigSave(BaseModel):
    name: str
    config: Dict[str, Any]
