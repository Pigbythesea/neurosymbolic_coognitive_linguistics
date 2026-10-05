"""Causal event-to-volume alignment, within-story FIR delays, and timing masks."""

import hashlib
from pathlib import Path
import numpy as np

from .extraction_inputs import checked_index, checked_story, story_folds
from .io import immutable_json, object_hash, read_json


def first_available_bin(seconds: float, tr: float, leading: int, sample_position: float) -> int:
    return int(np.ceil(seconds / tr + leading - sample_position))


def event_bins(story: dict, kind: str, n_rows: int, config: dict, tr: float) -> tuple[list[int], list[bool]]:
    items = story[kind]
    bins = []
    known = np.ones(n_rows, dtype=bool)
    to_bin = lambda seconds: first_available_bin(seconds, tr, config["leading_silence_trs"],
                                                  config["sample_position_within_tr"])
    for item in items:
        endpoint = item[config["availability"]]
        if endpoint is not None:
            slot = to_bin(endpoint)
            if not 0 <= slot < n_rows:
                raise ValueError(f"Observed event outside scan: {story['story_id']} {kind} {slot}")
            bins.append(slot)
        else:
            bins.append(-1)
            word = item if kind == "words" else story["words"][item["end_token"] - 1]
            left, right = word["previous_anchor_seconds"], word["following_anchor_seconds"]
            first = max(0, to_bin(left)) if left is not None else 0
            last = min(n_rows - 1, to_bin(right)) if right is not None else n_rows - 1
            if last < first:
                raise ValueError("Unresolved timing anchors are reversed.")
            known[first:last + 1] = False
    return bins, known.tolist()


def causal_bin_features(values: np.ndarray, bins: list[int], n_rows: int, *, reduction: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Aggregate event vectors only at or after availability; missing events stay absent."""
    values = np.asarray(values)
    slots = np.asarray(bins, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(slots) or reduction not in ("sum", "mean"):
        raise ValueError("Event feature rows/reduction do not match the alignment contract.")
    if np.any(slots < -1) or np.any(slots >= n_rows) or not np.isfinite(values).all():
        raise ValueError("Invalid event values or bin indices.")
    selected = slots >= 0
    result = np.zeros((n_rows, values.shape[1]), dtype=np.float32)
    counts = np.bincount(slots[selected], minlength=n_rows)
    np.add.at(result, slots[selected], values[selected])
    if reduction == "mean":
        np.divide(result, counts[:, None], out=result, where=counts[:, None] > 0)
    return result, counts


def fir_design(full_story: np.ndarray, raw_rows: list[int], delays: list[int],
               known_rows: list[bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """X[t] = concat(feature[t-d] for d in delays). Never concatenate stories first."""
    values = np.asarray(full_story)
    rows = np.asarray(raw_rows, dtype=np.int64)
    if values.ndim != 2 or not delays or any(type(d) is not int or d < 0 for d in delays):
        raise ValueError("FIR requires a feature matrix and nonnegative integer delays.")
    if len(set(delays)) != len(delays) or np.any(rows < 0) or np.any(rows >= len(values)):
        raise ValueError("Duplicate delays or out-of-range response rows.")
    known = np.ones(len(values), dtype=bool) if known_rows is None else np.asarray(known_rows, dtype=bool)
    if known.shape != (len(values),):
        raise ValueError("Timing-validity mask has the wrong length.")
    blocks, valid = [], np.ones(len(rows), dtype=bool)
    for delay in delays:
        source = rows - delay
        inside = source >= 0
        block = np.zeros((len(rows), values.shape[1]), dtype=values.dtype)
        block[inside] = values[source[inside]]
        valid &= inside
        valid[inside] &= known[source[inside]]
        blocks.append(block)
    return np.concatenate(blocks, axis=1), valid


def build_alignment(root: Path, config_path: Path) -> dict:
    config = read_json(config_path)
    if (config["format_version"] != 1 or config["availability"] != "offset_seconds"
            or config["resampling"] != "causal_bin_mean" or config["sample_position_within_tr"] != 0.5
            or config["leading_silence_trs"] != 5 or config["response_trim_start"] != 10
            or config["response_trim_end"] != 10):
        raise ValueError("Unsupported change to the inspected Deniz timing convention.")
    corpus = root / config["corpus"]
    index = checked_index(corpus)
    contract = read_json(root / config["data_contract"])
    if index["dataset_manifest_sha256"] != contract["manifest_sha256"]:
        raise ValueError("Timing contract and corpus use different source releases.")
    tr = contract["tr_seconds"]
    output = root / config["output"]
    result = {"format_version": 1, "config": config, "corpus_hash": index["content_hash"],
              "data_contract_hash": object_hash(contract), "tr_seconds": tr,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stories": []}
    for entry in index["stories"]:
        story = checked_story(corpus, entry)
        layouts = [s[entry["id"]] for s in contract["subjects"].values()]
        n_rows = layouts[0]["timepoints"]
        if any(s["timepoints"] != n_rows for s in layouts):
            raise ValueError("Subjects disagree on story length.")
        rows = list(range(config["response_trim_start"], n_rows - config["response_trim_end"]))
        if len(rows) != contract["feature_shapes"][entry["id"]]["english1000"][0] - 15:
            raise ValueError("Response and released feature trimming no longer agree.")
        record = {"format_version": 1, "story_id": entry["id"], "story_hash": entry["content_hash"],
                  "split": entry["split"], "raw_response_rows": n_rows, "response_raw_indices": rows,
                  "tr_centers_story_seconds": [((i + 0.5) - config["leading_silence_trs"]) * tr for i in rows],
                  "fir_delays_trs": config["fir_delays_trs"],
                  "fir_delays_seconds": [d * tr for d in config["fir_delays_trs"]]}
        for kind in ("words", "units"):
            bins, known = event_bins(story, kind, n_rows, config, tr)
            record[kind] = {"bin_indices": bins, "known_raw_rows": known,
                            "unresolved_events": sum(b < 0 for b in bins)}
        record["content_hash"] = object_hash(record)
        immutable_json(output / "stories" / (entry["id"] + ".json"), record)
        result["stories"].append({"id": entry["id"], "path": "stories/" + entry["id"] + ".json",
                                  "content_hash": record["content_hash"], "response_rows": len(rows),
                                  "unresolved_words": record["words"]["unresolved_events"],
                                  "unresolved_units": record["units"]["unresolved_events"]})
    result["content_hash"] = object_hash(result)
    immutable_json(output / "index.json", result)
    immutable_json(output / "splits.json", story_folds(index))
    print("ALIGNMENT READY:", len(result["stories"]), "stories;",
          sum(s["response_rows"] for s in result["stories"]), "response rows")
    return result


def checked_alignment(root: Path, story_id: str, corpus_hash: str) -> dict:
    index = read_json(root / "index.json")
    if (index["content_hash"] != object_hash({k: v for k, v in index.items() if k != "content_hash"})
            or index["corpus_hash"] != corpus_hash):
        raise ValueError("Alignment index changed or belongs to another corpus.")
    entry = next(s for s in index["stories"] if s["id"] == story_id)
    record = read_json(root / entry["path"])
    if record["content_hash"] != entry["content_hash"] or object_hash({k:v for k,v in record.items() if k != "content_hash"}) != entry["content_hash"]:
        raise ValueError("Alignment story content changed.")
    return record
