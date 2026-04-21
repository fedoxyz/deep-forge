#!/bin/bash
set -e
# Use /workspace if it exists (RunPod), otherwise /data (local Docker)
if [ -d "/workspace" ]; then
    export DATASET_BASE_DIR=${DATASET_BASE_DIR:-/workspace/dataset}
    export DATASET_PERSIST_FILE=${DATASET_PERSIST_FILE:-/workspace/dataset/.loaded_datasets.json}
    export HF_HOME=${HF_HOME:-/workspace/hf_cache}
    export CONFIGS_DIR=${CONFIGS_DIR:-/workspace/saved_configs}
    export TB_LOG_DIR=${TB_LOG_DIR:-/workspace/logs/tensorboard}
    mkdir -p /workspace/dataset /workspace/hf_cache \
             /workspace/logs/tensorboard /workspace/outputs \
             /workspace/saved_configs
else
    export DATASET_BASE_DIR=${DATASET_BASE_DIR:-/data/dataset}
    export DATASET_PERSIST_FILE=${DATASET_PERSIST_FILE:-/data/dataset/.loaded_datasets.json}
    export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
    export CONFIGS_DIR=${CONFIGS_DIR:-/app/saved_configs}
    export TB_LOG_DIR=${TB_LOG_DIR:-/data/logs/tensorboard}
    mkdir -p /data/logs/tensorboard  # ← was missing
fi
echo "=== LoRA Forge ==="
echo "Starting TensorBoard on port 6006..."
tensorboard --logdir="${TB_LOG_DIR}" --host=0.0.0.0 --port=6006 &  # ← use env var
echo "Starting API server on port 8000..."
cd /app
python scripts/start_server.py &
echo "All services started."
wait
