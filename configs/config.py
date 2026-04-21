import os

# dataset_manager.py or a central config.py
DATASET_BASE_DIR = os.environ.get("DATASET_BASE_DIR", "/data/dataset")
PERSIST_FILE = os.environ.get("DATASET_PERSIST_FILE", "/data/dataset/.loaded_datasets.json")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
