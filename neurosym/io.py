"""Deterministic artifact I/O shared by corpus, annotations, and analyses."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def immutable_json(path: Path, value: object) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Artifact differs from the current inputs/configuration: {path}")
    else:
        save_json(path, value)
