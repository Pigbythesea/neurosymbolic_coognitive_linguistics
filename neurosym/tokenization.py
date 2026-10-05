"""Verify each extraction endpoint against independently tokenized causal text."""

import hashlib
from pathlib import Path
from .extraction_inputs import stimulus_text
from .io import object_hash


def tokenization_plan(tokenizer, story: dict, model: dict) -> dict:
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with source offsets is required.")
    text = stimulus_text(story)
    encoded = tokenizer(text.rstrip(" \t"), add_special_tokens=False, return_offsets_mapping=True)
    # Explicit BOS, when the checkpoint defines one; no EOS or chat template.
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids = bos + encoded["input_ids"]
    offsets = [[0, 0]] * len(bos) + [list(x) for x in encoded["offset_mapping"]]
    if not ids or len(ids) > model["max_position_embeddings"]:
        raise ValueError("Full story exceeds the checkpoint context capacity; no silent truncation.")
    endpoints, overrides = [], {}
    boundaries = [("words", i, word["char_end"]) for i, word in enumerate(story["words"])]
    boundaries += [("units", i, unit["prefix_char_end"]) for i, unit in enumerate(story["units"])]
    cached = {}
    for kind, row, end in boundaries:
        # Removing trailing horizontal whitespace introduces no new text and avoids a dangling
        # whitespace token merging with a word that has not yet appeared.
        end = len(text[:end].rstrip(" \t"))
        if end not in cached:
            prefix = bos + tokenizer(text[:end], add_special_tokens=False)["input_ids"]
            if not prefix:
                raise ValueError("Empty prefix at a real extraction endpoint.")
            if len(prefix) > model["max_position_embeddings"]:
                raise ValueError("An independently tokenized prefix exceeds checkpoint context capacity.")
            stable = ids[:len(prefix)] == prefix
            cached[end] = (len(prefix) - 1, stable)
            if not stable:
                overrides[str(end)] = prefix
        token_index, stable = cached[end]
        endpoints.append({"kind": kind, "row": row, "char_end": end,
                          "token_index": token_index, "canonical_prefix": stable})
    plan = {"format_version": 1, "story_id": story["story_id"], "story_hash": story["content_hash"],
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_revision": model["revision"], "input_ids": ids, "offset_mapping": offsets,
            "endpoints": endpoints, "independent_prefixes": overrides,
            "text_sha256": object_hash(text), "bos_token_ids": bos,
            "policy": "No prompts, chat template, EOS, annotations or future tokenization at endpoints. Nonstable BPE prefixes are separately evaluated with a fresh cache."}
    plan["content_hash"] = object_hash(plan)
    return plan
