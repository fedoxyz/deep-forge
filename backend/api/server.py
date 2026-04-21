"""
LoRA Forge API — application entry point.

All route handlers, background tasks, and shared state have been
split into dedicated sub-modules under backend/api/:

    routes/          — one module per resource (configs, training, builder, …)
    state/           — shared mutable dicts (training_state, optuna_state)
    tasks/           — background training & Optuna task functions
    models.py        — Pydantic request/response schemas
    dependencies.py  — singletons, directory constants
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.middleware.auth import PasswordMiddleware

from backend.api.routes import all_routers


def create_app() -> FastAPI:
    app = FastAPI(title="LoRA Forge API", version="3.0.0")
    app.add_middleware(PasswordMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in all_routers:
        app.include_router(router)

    @app.on_event("startup")
    async def startup():
        from backend.datasets.dataset_manager import restore_loaded_datasets
        restore_loaded_datasets()

    return app


app = create_app()
