from .configs import router as configs_router
from .model_specs import router as model_specs_router
from .hparam_configs import router as hparam_configs_router
from .training import router as training_router
from .builder import router as builder_router
from .datasets import router as datasets_router
from .inspection import router as inspection_router
from .optuna import router as optuna_router
from .system import router as system_router
from .vision import router as vision_router

all_routers = [
    configs_router,
    model_specs_router,
    hparam_configs_router,
    training_router,
    builder_router,
    datasets_router,
    inspection_router,
    optuna_router,
    system_router,
    vision_router
]
