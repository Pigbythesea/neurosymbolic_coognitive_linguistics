"""Check every real word and unit prefix using each checkpoint's actual tokenizer."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.extraction_inputs import checked_index, checked_story
from neurosym.io import immutable_json, read_json, object_hash
from neurosym.model_registry import read_lock, snapshot_path
from neurosym.tokenization import tokenization_plan


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/extraction.json")
    lock = read_lock(config, ROOT / config["model_lock"])
    corpus = ROOT / config["corpus"]
    index = checked_index(corpus)
    models = [m for m in lock["models"] if args.model is None or m["id"] == args.model]
    if not models:
        raise ValueError("Unknown model ID.")
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path(ROOT, config, model), local_files_only=True,
                                                 trust_remote_code=False, use_fast=True)
        entries = []
        for entry in index["stories"]:
            plan = tokenization_plan(tokenizer, checked_story(corpus, entry), model)
            immutable_json(ROOT / config["tokenization"] / model["id"] / (entry["id"] + ".json"), plan)
            entries.append({"story_id": entry["id"], "plan_hash": plan["content_hash"],
                            "tokens": len(plan["input_ids"]), "independent_prefixes": len(plan["independent_prefixes"])})
            print(model["id"], entry["id"], entries[-1], flush=True)
        report = {"corpus_hash": index["content_hash"], "model_lock_hash": lock["content_hash"], "stories": entries}
        report["content_hash"] = object_hash(report)
        immutable_json(ROOT / config["tokenization"] / model["id"] / "index.json", report)


if __name__ == "__main__":
    main()
