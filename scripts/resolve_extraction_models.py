"""Resolve model metadata only; no weights are downloaded."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.io import read_json
from neurosym.model_registry import lock_models

if __name__ == "__main__":
    config = read_json(ROOT / "configs/extraction.json")
    lock_models(config, ROOT / config["model_lock"])
