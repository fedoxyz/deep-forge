"""
Shared dependencies: singletons, directory paths, and factory functions
used across route modules and background tasks.
"""

import os

from backend.configs.config_manager import ConfigManager

# ── Directory paths ──
TB_LOG_DIR = os.environ.get("TB_LOG_DIR", "/data/logs/tensorboard")
os.makedirs(TB_LOG_DIR, exist_ok=True)

MODEL_SPECS_DIR = os.environ.get("MODEL_SPECS_DIR", "./saved_model_specs")
os.makedirs(MODEL_SPECS_DIR, exist_ok=True)

HPARAM_CONFIGS_DIR = os.environ.get("HPARAM_CONFIGS_DIR", "./saved_hparam_configs")
os.makedirs(HPARAM_CONFIGS_DIR, exist_ok=True)

# ── Singletons ──
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(
            os.environ.get("CONFIGS_DIR", "./saved_configs")
        )
    return _config_manager
