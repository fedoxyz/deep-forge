"""Routes for system health, available modules, and bucket config."""

from fastapi import APIRouter

from backend.api.dependencies import TB_LOG_DIR

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/modules")
async def list_modules():
    from backend.modules.optimizers import OPTIMIZER_REGISTRY
    from backend.modules.schedulers import SCHEDULER_REGISTRY
    from backend.modules.losses import LOSS_REGISTRY

    return {
        "optimizers": list(OPTIMIZER_REGISTRY.keys()),
        "schedulers": list(SCHEDULER_REGISTRY.keys()),
        "losses": list(LOSS_REGISTRY.keys()),
    }


@router.get("/buckets")
async def get_buckets():
    from backend.datasets.image_caption import DEFAULT_BUCKETS
    return {"buckets": DEFAULT_BUCKETS}


@router.get("/health")
async def health():
    import torch
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tb_log_dir": TB_LOG_DIR,
    }
