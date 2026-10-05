"""Compile measured model states to the scan clock; read FIR designs on demand."""

import hashlib
import os
from pathlib import Path
import h5py
import numpy as np

from .extraction import exclusive_run
from .extraction_inputs import checked_index
from .io import immutable_json, object_hash, read_json, save_json
from .temporal import causal_bin_features, checked_alignment, fir_design


def align_model(root: Path, model_id: str) -> dict:
    config = read_json(root / "configs/extraction.json")
    index = checked_index(root / config["corpus"])
    source = root / config["output"] / model_id
    run = read_json(source / "run.json")
    complete = read_json(source / "complete.json")
    if run["corpus_hash"] != index["content_hash"] or complete["run_hash"] != object_hash(run):
        raise ValueError("Completed hidden states do not match this corpus/run.")
    if [s["story_id"] for s in complete["stories"]] != [s["id"] for s in index["stories"]]:
        raise ValueError("Extraction receipt does not cover the corpus.")
    alignment_root = root / config["alignment"]
    alignment_index = read_json(alignment_root / "index.json")
    destination = source / "aligned"
    identity = {"format_version": 1, "source_run_hash": object_hash(run),
                "alignment_hash": alignment_index["content_hash"],
                "code_hashes": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("model_features.py", "temporal.py")}}
    immutable_json(destination / "run.json", identity)
    records = []
    with exclusive_run(destination / "RUNNING.lock"):
        for entry in index["stories"]:
            alignment = checked_alignment(alignment_root, entry["id"], index["content_hash"])
            final = destination / (entry["id"] + ".h5")
            if final.exists():
                with h5py.File(final, "r") as file:
                    if file.attrs["run_hash"] != object_hash(identity) or not file.attrs["complete"]:
                        raise ValueError("Aligned feature output differs from this run.")
            else:
                partial = final.with_suffix(".partial.h5")
                if partial.is_symlink() or final.is_symlink():
                    raise ValueError("Feature output must not be a symlink.")
                with h5py.File(source / final.name, "r") as original, h5py.File(partial, "w") as file:
                    if not original.attrs["complete"] or original.attrs["run_hash"] != object_hash(run):
                        raise ValueError("Unfinished or mismatched actual hidden states.")
                    file.attrs.update(run_hash=object_hash(identity), complete=False, story_id=entry["id"],
                                      source_run_hash=object_hash(run), alignment_hash=alignment["content_hash"])
                    file.create_dataset("response_raw_indices", data=alignment["response_raw_indices"])
                    file.create_dataset("fir_delays_trs", data=alignment["fir_delays_trs"])
                    n_rows = alignment["raw_response_rows"]
                    for kind in ("words", "units"):
                        states = original[kind]
                        layer_count, events, hidden = states.shape
                        bins = alignment[kind]["bin_indices"]
                        if events != len(bins):
                            raise ValueError("Model endpoint rows do not match timing records.")
                        group = file.create_group(kind)
                        values = group.create_dataset("features", shape=(layer_count, n_rows, hidden), dtype="float32",
                                                      chunks=(1, min(64, n_rows), hidden), compression="lzf",
                                                      shuffle=True, fletcher32=True)
                        for layer in range(layer_count):
                            aggregated, counts = causal_bin_features(states[layer], bins, n_rows)
                            values[layer] = aggregated
                        group.create_dataset("observed_event_counts", data=counts)
                        group.create_dataset("known_raw_rows", data=alignment[kind]["known_raw_rows"])
                    file.attrs["complete"] = True
                    file.flush()
                os.replace(partial, final)
            records.append({"story_id": entry["id"], "path": final.name, "bytes": final.stat().st_size})
            print("ALIGNED", model_id, entry["id"], flush=True)
        report = {"run_hash": object_hash(identity), "stories": records}
        save_json(destination / "complete.json", report)
    return report


class FrozenFeatureReader:
    """Return rows matching DenizReader.response(..., trim=True), plus an explicit mask."""
    def __init__(self, root: Path, model_id: str):
        self.root = Path(root)
        self.config = read_json(self.root / "configs/extraction.json")
        self.source = self.root / self.config["output"] / model_id
        self.aligned_run = read_json(self.source / "aligned/run.json")
        complete = read_json(self.source / "aligned/complete.json")
        if complete["run_hash"] != object_hash(self.aligned_run):
            raise ValueError("Aligned model features are incomplete.")

    def design(self, story_id: str, layer: int, *, kind: str = "words", delays: list[int] | None = None):
        if kind not in ("words", "units"):
            raise ValueError("Unknown representation kind.")
        with h5py.File(self.source / "aligned" / (story_id + ".h5"), "r") as file:
            if file.attrs["run_hash"] != object_hash(self.aligned_run) or not file.attrs["complete"]:
                raise ValueError("Aligned story provenance mismatch.")
            if not 0 <= layer < file[kind]["features"].shape[0]:
                raise ValueError("Layer is outside the extracted layer range.")
            values = file[kind]["features"][layer]
            rows = file["response_raw_indices"][()].tolist()
            known = file[kind]["known_raw_rows"][()].tolist()
            selected = file["fir_delays_trs"][()].tolist() if delays is None else delays
        return fir_design(values, rows, selected, known)

    def events(self, story_id: str, layer: int, *, kind: str = "words") -> np.ndarray:
        if kind not in ("words", "units"):
            raise ValueError("Unknown representation kind.")
        with h5py.File(self.source / (story_id + ".h5"), "r") as file:
            if not file.attrs["complete"] or file.attrs["run_hash"] != self.aligned_run["source_run_hash"]:
                raise ValueError("Event-state provenance mismatch.")
            if not 0 <= layer < file[kind].shape[0]:
                raise ValueError("Layer is outside the extracted layer range.")
            return file[kind][layer]
