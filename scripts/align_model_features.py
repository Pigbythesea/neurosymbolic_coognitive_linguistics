"""Compile all completed story states for one model onto the real scan clock."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.io import read_json
from neurosym.model_features import align_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--model-index", type=int)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/extraction.json")
    if args.model_index is not None and not 0 <= args.model_index < len(config["models"]):
        parser.error("Model array index is outside the configured panel.")
    name = args.model if args.model is not None else config["models"][args.model_index]["id"]
    align_model(ROOT, name)
