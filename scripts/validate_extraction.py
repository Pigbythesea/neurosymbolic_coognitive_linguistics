"""Validate the actual corpus, timing, released features, splits and locked panel. No inference."""
import ast
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.dataset import DenizReader
from neurosym.extraction_inputs import checked_index, checked_story, stimulus_text
from neurosym.io import object_hash, read_json, save_json
from neurosym.model_registry import read_lock
from neurosym.temporal import causal_bin_features, checked_alignment, fir_design


def main():
    config = read_json(ROOT / "configs/extraction.json")
    index = checked_index(ROOT / config["corpus"])
    lock = read_lock(config, ROOT / config["model_lock"])
    clock = read_json(ROOT / "configs/alignment.json")
    contract = read_json(ROOT / config["data_contract"])
    reader = DenizReader(ROOT / "data/raw/deniz", ROOT / config["data_contract"])
    tr = contract["tr_seconds"]
    counts = {"stories": 0, "words": 0, "units": 0, "response_rows": 0, "unresolved_words": 0, "unresolved_units": 0}
    details = []
    for entry in index["stories"]:
        story = checked_story(ROOT / config["corpus"], entry)
        aligned = checked_alignment(ROOT / config["alignment"], entry["id"], index["content_hash"])
        text = stimulus_text(story)
        if len(text) != len(story["text"]) or any(text[w["char_start"]:w["char_end"]] != w["text"] for w in story["words"]):
            raise ValueError("Model text masking changed actual word offsets.")
        for kind in ("words", "units"):
            for item, slot in zip(story[kind], aligned[kind]["bin_indices"], strict=True):
                stamp = item["offset_seconds"]
                if stamp is None:
                    if slot != -1:
                        raise ValueError("A missing timestamp was invented.")
                else:
                    center = (slot + 0.5 - clock["leading_silence_trs"]) * tr
                    if center + 1e-9 < stamp or center - tr >= stamp + 1e-9:
                        raise ValueError("Causal bin includes future text or delays an extra TR.")
            counts[kind] += len(story[kind])
            counts["unresolved_" + kind] += aligned[kind]["unresolved_events"]
        # Real word lengths exercise aggregation and conservation without fake vectors.
        lengths = np.array([[len(w["text"])] for w in story["words"]], dtype=np.float32)
        summed, observed = causal_bin_features(lengths, aligned["words"]["bin_indices"], aligned["raw_response_rows"], reduction="sum")
        mean, _ = causal_bin_features(lengths, aligned["words"]["bin_indices"], aligned["raw_response_rows"])
        timed = np.array(aligned["words"]["bin_indices"]) >= 0
        if not np.isclose(summed.sum(), lengths[timed].sum()) or not np.allclose(mean[:, 0] * observed, summed[:, 0]):
            raise ValueError("Real word feature aggregation lost events or misnormalized bins.")
        # The release documents five trailing silent response TRs absent from features.
        released = reader.features(entry["id"], ["numwords"], trim=False)["numwords"]
        onsets = np.array([w["onset_seconds"] for w in story["words"] if w["onset_seconds"] is not None])
        edges = np.arange(len(released) + 1) * tr
        shifted_counts = np.histogram(onsets + clock["leading_silence_trs"] * tr, edges)[0]
        unshifted_counts = np.histogram(onsets, edges)[0]
        shifted_error = float(np.mean(abs(shifted_counts - released[:, 0])))
        unshifted_error = float(np.mean(abs(unshifted_counts - released[:, 0])))
        if shifted_error >= unshifted_error:
            raise ValueError("The documented leading silence disagrees with released word-count timing.")
        full = np.pad(released, ((0, 5), (0, 0)))
        x, mask = fir_design(full, aligned["response_raw_indices"], clock["fir_delays_trs"])
        for column, delay in enumerate(clock["fir_delays_trs"]):
            if not np.array_equal(x[:, column], full[np.array(aligned["response_raw_indices"]) - delay, 0]):
                raise ValueError("Real released feature lag has an indexing error.")
        if not mask.all() or len(x) != reader.features(entry["id"], ["numwords"])["numwords"].shape[0]:
            raise ValueError("FIR rows disagree with the inspected trim.")
        _, known_words = fir_design(full, aligned["response_raw_indices"], clock["fir_delays_trs"], aligned["words"]["known_raw_rows"])
        _, known_units = fir_design(full, aligned["response_raw_indices"], clock["fir_delays_trs"], aligned["units"]["known_raw_rows"])
        counts["stories"] += 1
        counts["response_rows"] += len(x)
        details.append({"story_id": entry["id"], "response_rows": len(x),
                        "word_timing_valid_rows": int(known_words.sum()), "unit_timing_valid_rows": int(known_units.sum()),
                        "released_word_count_mae_with_leading_silence": shifted_error,
                        "released_word_count_mae_without_leading_silence": unshifted_error})
    folds = read_json(ROOT / config["alignment"] / "splits.json")
    if folds["corpus_hash"] != index["content_hash"]:
        raise ValueError("Split manifest uses a different corpus.")
    for outer in folds["outer_folds"]:
        for inner in outer["inner_folds"]:
            groups = [set(inner["train"]), set(inner["validation"]), set(outer["test"]), set(folds["heldout"])]
            if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
                raise ValueError("Story leakage in a nested evaluation split.")
            if set.union(*groups) != {s["id"] for s in index["stories"]}:
                raise ValueError("Nested split omits actual stories.")
    for path in list((ROOT / "neurosym").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
    report = {"status": "complete", "counts": counts, "stories": details,
              "model_lock_hash": lock["content_hash"], "models": len(lock["models"]),
              "weight_download_gib": sum(f["bytes"] for m in lock["models"] for f in m["files"]) / 2**30,
              "gpu_extraction_executed": False, "tokenizer_plans_executed": False,
              "checks": ["real text/offset identity", "no future event assignment", "missing-time preservation",
                         "word-feature conservation", "released-feature FIR alignment", "story-disjoint nested folds", "Python 3.11 syntax"]}
    save_json(ROOT / "artifacts/extraction-validation.json", report)
    print({k:v for k,v in report.items() if k != "stories"})


if __name__ == "__main__":
    main()
