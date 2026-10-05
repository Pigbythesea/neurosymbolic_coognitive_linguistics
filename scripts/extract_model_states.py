"""Extract all real story word/unit states for one pinned frozen checkpoint."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.extraction import extract_model
from neurosym.io import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--model-index", type=int)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/extraction.json")
    if args.model_index is not None and not 0 <= args.model_index < len(config["models"]):
        parser.error("Model array index is outside the configured panel.")
    model_id = args.model if args.model is not None else config["models"][args.model_index]["id"]
    try:
        extract_model(ROOT, model_id)
    except KeyboardInterrupt:
        raise SystemExit("Stopped; completed stories remain reusable. The interrupted story is recomputed on resume.")
