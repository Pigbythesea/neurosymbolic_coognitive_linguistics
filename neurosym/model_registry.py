"""Exact public model snapshots, explicit file hashes, and offline loading paths."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import urllib.request
import json

from .io import immutable_json, object_hash, read_json


METADATA_FILES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model",
                  "model.safetensors.index.json", "README.md", "LICENSE", "LICENSE.txt"}


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "neurosym-research/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def lock_models(config: dict, destination: Path) -> dict:
    if destination.exists():
        return read_lock(config, destination)
    models = []
    for model in config["models"]:
        info = fetch_json("https://huggingface.co/api/models/" + model["repo"] + "?blobs=true")
        revision = info["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model revision is not a Git commit SHA.")
        source = fetch_json(f"https://huggingface.co/{model['repo']}/resolve/{revision}/config.json")
        text_config = source.get("text_config", source)
        if source.get("quantization_config") or text_config.get("quantization_config"):
            raise ValueError("Quantized checkpoint cannot enter the full-precision panel.")
        files = []
        for item in info["siblings"]:
            name = item["rfilename"]
            if name not in METADATA_FILES and not ("/" not in name and name.endswith(".safetensors")):
                continue
            if PurePosixPath(name).name != name or ".." in name:
                raise ValueError("Unexpected checkpoint filename.")
            lfs = item.get("lfs") or {}
            files.append({"name": name, "bytes": item.get("size", lfs.get("size")),
                          "sha256": lfs.get("sha256"), "git_blob": item.get("blobId")})
        if not any(f["name"].endswith(".safetensors") for f in files):
            raise ValueError("No safetensors weights in " + model["repo"])
        if any(f["bytes"] is None or not (f["sha256"] or f["git_blob"]) for f in files):
            raise ValueError("Incomplete file integrity metadata for " + model["repo"])
        models.append({**model, "revision": revision, "config": source,
                       "num_hidden_layers": text_config["num_hidden_layers"],
                       "hidden_size": text_config["hidden_size"],
                       "max_position_embeddings": text_config["max_position_embeddings"],
                       "files": sorted(files, key=lambda f: f["name"])})
        print(f"LOCKED {model['id']} {revision} {sum(f['bytes'] for f in files)/2**30:.2f} GiB", flush=True)
    lock = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "models": models}
    lock["content_hash"] = object_hash(lock)
    immutable_json(destination, lock)
    return lock


def read_lock(config: dict, path: Path) -> dict:
    lock = read_json(path)
    if lock["content_hash"] != object_hash({k: v for k, v in lock.items() if k != "content_hash"}):
        raise ValueError("Model lock hash changed.")
    if [{k: m[k] for k in ("id", "repo", "family", "stage", "loader")} for m in lock["models"]] != config["models"]:
        raise ValueError("Model panel differs from its lock; use an explicit new lock for a changed panel.")
    return lock


def select_model(lock: dict, name: str) -> dict:
    matches = [m for m in lock["models"] if m["id"] == name]
    if len(matches) != 1:
        raise ValueError("Unknown model ID: " + name)
    return matches[0]


def verify_snapshot(snapshot: Path, model: dict) -> dict:
    for item in model["files"]:
        path = snapshot / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing or incomplete checkpoint file: " + str(path))
        digest = hashlib.sha256() if item["sha256"] else hashlib.sha1()
        if not item["sha256"]:
            digest.update(f"blob {item['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != (item["sha256"] or item["git_blob"]):
            raise ValueError("Checkpoint hash mismatch: " + str(path))
    return {"revision": model["revision"], "files": len(model["files"]),
            "bytes": sum(f["bytes"] for f in model["files"])}


def snapshot_path(root: Path, config: dict, model: dict) -> Path:
    return root / config["model_cache"] / ("models--" + model["repo"].replace("/", "--")) / "snapshots" / model["revision"]
