"""Download and hash-verify the complete locked model panel on a compute node."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.io import read_json, save_json
from neurosym.model_registry import read_lock, verify_snapshot


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional registered model ID; default downloads the complete panel.")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/extraction.json")
    lock = read_lock(config, ROOT / config["model_lock"])
    models = [m for m in lock["models"] if args.model is None or m["id"] == args.model]
    if not models:
        raise ValueError("Unknown model ID.")
    for model in models:
        print("DOWNLOAD", model["id"], model["revision"], flush=True)
        snapshot = Path(snapshot_download(model["repo"], revision=model["revision"],
                         cache_dir=ROOT / config["model_cache"], allow_patterns=[f["name"] for f in model["files"]],
                         max_workers=4))
        verified = verify_snapshot(snapshot, model)
        verified["lock_hash"] = lock["content_hash"]
        verified["file_stats"] = {f["name"]: {"size": (snapshot / f["name"]).stat().st_size,
                                               "mtime_ns": (snapshot / f["name"]).stat().st_mtime_ns}
                                  for f in model["files"]}
        save_json(ROOT / "artifacts/model-verification" / (model["id"] + ".json"), verified)
        print("VERIFIED", model["id"], verified["bytes"], "bytes", flush=True)


if __name__ == "__main__":
    main()
