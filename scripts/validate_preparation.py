"""Check the complete prepared real corpus, inspected layouts, and annotation configuration.

No neural responses, annotations, or model outputs are fabricated or generated.
"""

import argparse
import ast
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import annotation_passes, codex_command, resolve_codex, validate_cli, verified_story
from neurosym.dataset import DenizReader
from neurosym.graphs import graph_schema
from neurosym.io import object_hash, read_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-codex", action="store_true", help="Read-only CLI and saved-login checks; no inference.")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/annotation.json")
    selected_passes = annotation_passes(config)
    annotation_passes(config, "blind")
    command = codex_command(resolve_codex(), config, ROOT / "artifacts/annotation-context",
                            ROOT / config["schema"], ROOT / "artifacts/annotation-command-check.json")
    if 'model_reasoning_effort="' + config["reasoning_effort"] + '"' not in command:
        raise ValueError("Annotation command does not preserve the configured reasoning effort.")
    corpus = ROOT / config["corpus"]
    index = read_json(corpus / "index.json")
    contract = read_json(ROOT / "data/processed/deniz/contract.json")
    if index["dataset_manifest_sha256"] != contract["manifest_sha256"]:
        raise ValueError("Corpus and fMRI contract use different source manifests.")
    if index["content_hash"] != object_hash({key: value for key, value in index.items() if key != "content_hash"}):
        raise ValueError("Invalid corpus content hash.")
    counts = Counter()
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        words = story["words"]
        for i, word in enumerate(words):
            if word["index"] != i or story["text"][word["char_start"]:word["char_end"]] != word["text"]:
                raise ValueError("Word provenance differs from its actual transcript.")
            if word["alignment"] == "unresolved":
                if word["onset_seconds"] is not None or word["offset_seconds"] is not None:
                    raise ValueError("Unresolved words must not receive fabricated timings.")
            elif word["onset_seconds"] > word["offset_seconds"] or not word["textgrid_interval_indices"]:
                raise ValueError("Invalid observed timing interval.")
            counts[word["alignment"]] += 1
        cursor = 0
        for unit in story["units"]:
            end = unit["end_token"]
            if unit["start_token"] != cursor or not cursor < end <= len(words):
                raise ValueError("Annotation units overlap, omit words, or exceed the source.")
            next_word = words[end]["char_start"] if end < len(words) else len(story["text"])
            if unit["prefix_char_end"] != next_word:
                raise ValueError("Prefix contains future text or omits endpoint punctuation.")
            cursor = end
            counts["units"] += 1
        if cursor != len(words):
            raise ValueError("Corpus units do not cover the full story.")
    reader = DenizReader(ROOT / "data/raw/deniz", ROOT / "data/processed/deniz/contract.json")
    for story in contract["stories"]:
        arrays = reader.features(story)
        expected = contract["feature_shapes"][story]["english1000"][0] - 15
        if any(value.shape[0] != expected for value in arrays.values()):
            raise ValueError("Real released features do not match the trim contract.")
        for subject in contract["subjects"]:
            spec = contract["subjects"][subject][story]
            if spec["timepoints"] - 20 != expected or spec["repeats"] != (2 if story == "story_11" else 1):
                raise ValueError("Inspected brain layout does not match real feature rows/repeats.")
    schema = graph_schema()
    if read_json(ROOT / config["schema"]) != schema:
        raise ValueError("Exported schema is stale.")
    response = schema
    if config.get("protocol") == "joint-source-v3":
        from neurosym.annotation_joint import response_schema
        response = response_schema()
    if config.get("protocol") in ("joint-source-v4", "joint-source-v5"):
        from neurosym.annotation_source import response_schema
        response = response_schema()
    for definition in [response, *response.get("$defs", {}).values()]:
        if definition.get("type") == "object":
            if definition.get("additionalProperties") is not False or set(definition["required"]) != set(definition["properties"]):
                raise ValueError("Structured-output objects must be closed and require every field.")
    for folder in ("neurosym", "scripts"):
        for path in (ROOT / folder).glob("*.py"):
            ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
    result = {"status": "complete", "scope": "real corpus, released features, recorded cluster response layouts, schema, and Python 3.11 syntax",
              "stories": len(index["stories"]), "subjects": len(contract["subjects"]), "counts": dict(counts),
              "annotation_configuration": {"model": config["model"], "reasoning_effort": config["reasoning_effort"],
                                           "default_passes": selected_passes,
                                           "protocol": config.get("protocol", "ground-and-bind-v1"),
                                           "planned_requests_without_corrections": ({"joint-source-v5": 1, "joint-source-v4": 1, "joint-source-v3": 1, "source-anchored-v2": 3}.get(config.get("protocol"), 2))
                                               * counts["units"] * len(selected_passes)},
              "annotation_inference_run": False, "full_neural_loader_executed_locally": False}
    if args.check_codex:
        result["codex"] = validate_cli(resolve_codex())
    save_json(ROOT / "artifacts/preparation-validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
