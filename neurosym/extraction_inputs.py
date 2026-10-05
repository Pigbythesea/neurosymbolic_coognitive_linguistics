"""Read the existing corpus contract without importing annotation machinery."""

from pathlib import Path
import re

from .io import object_hash, read_json


def checked_index(corpus: Path) -> dict:
    index = read_json(corpus / "index.json")
    if index["content_hash"] != object_hash({k: v for k, v in index.items() if k != "content_hash"}):
        raise ValueError("Corpus index content hash changed.")
    return index


def checked_story(corpus: Path, entry: dict) -> dict:
    story = read_json(corpus / entry["path"])
    content = {k: v for k, v in story.items() if k != "content_hash"}
    if story["content_hash"] != entry["content_hash"] or object_hash(content) != entry["content_hash"]:
        raise ValueError("Corpus story content hash changed: " + entry["id"])
    return story


def stimulus_text(story: dict) -> str:
    # Same-length masking preserves the corpus's character and word offsets.
    return re.sub(r"\{[^}]*\}", lambda m: " " * len(m.group()), story["text"])


def story_folds(index: dict) -> dict:
    development = [s["id"] for s in index["stories"] if s["split"] == "development"]
    heldout = [s["id"] for s in index["stories"] if s["split"] == "heldout"]
    if not development or not heldout or set(development) & set(heldout):
        raise ValueError("Invalid story-level evaluation split.")
    return {"format_version": 1, "corpus_hash": index["content_hash"],
            "development": development, "heldout": heldout,
            "selection_folds": [{"train": [x for x in development if x != val], "validation": [val]}
                                for val in development],
            "outer_folds": [{"test": [test], "train": [x for x in development if x != test],
                             "inner_folds": [{"train": [x for x in development if x not in (test, val)],
                                              "validation": [val]} for val in development if val != test]}
                            for test in development],
            "policy": "Fit normalization, dimensionality reduction, vocabulary and hyperparameters on training stories only. Heldout story repeats stay together; neither repeat selects models or voxels."}
