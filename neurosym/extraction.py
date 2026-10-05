"""Frozen, full-story causal hidden states with independently verified BPE prefixes."""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from .extraction_inputs import checked_index, checked_story
from .io import immutable_json, object_hash, read_json, save_json
from .model_registry import read_lock, select_model, snapshot_path


def runtime_metadata_json(metadata: dict) -> str:
    """Preserve loading diagnostics; Transformers can return sets of parameter names."""
    def encode_set(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(f"Unsupported runtime metadata type: {type(value).__name__}")

    return json.dumps(metadata, default=encode_set, sort_keys=True, allow_nan=False)


@contextmanager
def exclusive_run(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID"),
                       "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError as error:
        raise RuntimeError("Active or interrupted run lock; inspect the recorded process/job before removing: " + str(path)) from error
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare_run_record(destination: Path, identity: dict) -> None:
    """With the run lock held, archive only failed starts that wrote no feature data."""
    record = destination / "run.json"
    if record.is_symlink():
        raise ValueError("Extraction run record must not be a symlink.")
    if record.exists() and read_json(record) != identity:
        partials = []
        for path in destination.iterdir():
            if path.is_symlink():
                raise ValueError("Extraction output must not be a symlink: " + str(path))
            if path.name in {"run.json", "RUNNING.lock"}:
                continue
            if path.name == "failed-starts" and path.is_dir():
                continue
            if not path.is_file() or not path.name.endswith(".partial.h5"):
                raise ValueError("Existing extraction differs from this run; preserve its outputs: " + str(path))
            with h5py.File(path, "r") as file:
                if len(file) or len(file.attrs):
                    raise ValueError("Changed extraction code cannot reuse a nonempty partial story: " + str(path))
            partials.append(path)
        # The observed metadata-serialization failure leaves only run.json and an
        # empty HDF5 file. Keep both for provenance; never discard computed states.
        archive = destination / "failed-starts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True, exist_ok=False)
        for path in [*partials, record]:
            os.replace(path, archive / path.name)
        print("ARCHIVED FAILED START", archive, flush=True)
    immutable_json(record, identity)


def check_model_files(root: Path, config: dict, lock: dict, model: dict) -> Path:
    snapshot = snapshot_path(root, config, model)
    receipt = read_json(root / "artifacts/model-verification" / (model["id"] + ".json"))
    if receipt["lock_hash"] != lock["content_hash"] or receipt["revision"] != model["revision"]:
        raise ValueError("Weights have not been verified against the current lock.")
    for name, recorded in receipt["file_stats"].items():
        stat = (snapshot / name).stat()
        if stat.st_size != recorded["size"] or stat.st_mtime_ns != recorded["mtime_ns"]:
            raise ValueError("Checkpoint file changed since full hash verification: " + name)
    if set(receipt["file_stats"]) != {f["name"] for f in model["files"]}:
        raise ValueError("Incomplete checkpoint verification receipt.")
    return snapshot


def load_backbone(snapshot: Path, model: dict, config: dict):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Extraction requires an allocated CUDA GPU with bfloat16 support.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = Qwen3_5ForConditionalGeneration if model["loader"] == "qwen3_5" else AutoModelForCausalLM
    owner, loading = loader.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                            dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                            attn_implementation=config["attention_implementation"],
                                            output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Checkpoint did not load all required parameters: " + str(loading))
    backbone = owner.model.language_model if model["loader"] == "qwen3_5" else owner.model
    backbone.eval().requires_grad_(False)
    if any(parameter.device.type != "cuda" for parameter in backbone.parameters()):
        raise RuntimeError("Unexpected CPU/disk offload.")
    metadata = {"device": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                "parameter_dtypes": dict(Counter(str(p.dtype) for p in backbone.parameters())),
                "loading_info": loading, "attention": config["attention_implementation"],
                "generation": False, "chat_template": False,
                "optional_external_kernels": False}
    return backbone, metadata


def evaluate_sequence(backbone, ids: list[int], endpoints: list[dict], file, model: dict, chunk_size: int,
                      written: dict[str, np.ndarray]):
    import torch
    cache = None
    layer_count = model["num_hidden_layers"] + 1
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        selected = [e for e in endpoints if start <= e["token_index"] < end]
        with torch.inference_mode():
            output = backbone(input_ids=torch.tensor([ids[start:end]], dtype=torch.long, device="cuda:0"),
                              position_ids=torch.arange(start, end, device="cuda:0").unsqueeze(0),
                              past_key_values=cache, use_cache=True, output_hidden_states=bool(selected), return_dict=True)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Backbone failed to preserve the full story cache.")
        if selected:
            states = output.hidden_states
            if states is None or len(states) != layer_count:
                raise RuntimeError("Backbone hidden-state convention changed; expected embedding plus every layer.")
            # Make the final-layer normalization convention explicit across
            # architectures and Transformers output-capture implementations.
            states = (*states[:-1], output.last_hidden_state)
            for kind in ("words", "units"):
                chosen = sorted((e for e in selected if e["kind"] == kind), key=lambda e: e["row"])
                if not chosen:
                    continue
                rows = [e["row"] for e in chosen]
                positions = torch.tensor([e["token_index"] - start for e in chosen], device="cuda:0")
                for layer, hidden in enumerate(states):
                    values = hidden[0].index_select(0, positions).float().cpu().numpy()
                    if values.shape != (len(rows), model["hidden_size"]) or not np.isfinite(values).all():
                        raise RuntimeError("Unexpected shape or nonfinite actual hidden states.")
                    file[kind][layer, rows, :] = values
                written[kind][rows] = True
        del output


def extract_model(root: Path, model_id: str) -> None:
    config = read_json(root / "configs/extraction.json")
    required = {"format_version": 1, "compute_dtype": "bfloat16", "storage_dtype": "float32",
                "context_policy": "full_story", "layers": "all", "attention_implementation": "sdpa",
                "representation": "last_token_of_independently_tokenized_prefix",
                "text_policy": "raw_transcript_with_brace_event_tags_masked"}
    if any(config.get(k) != v for k, v in required.items()) or config["chunk_tokens"] < 1:
        raise ValueError("Unsupported extraction policy; no silent fallback.")
    lock = read_lock(config, root / config["model_lock"])
    model = select_model(lock, model_id)
    corpus = root / config["corpus"]
    index = checked_index(corpus)
    snapshot = check_model_files(root, config, lock, model)
    plans_root = root / config["tokenization"] / model_id
    plans_index = read_json(plans_root / "index.json")
    if (plans_index["content_hash"] != object_hash({k:v for k,v in plans_index.items() if k != "content_hash"})
            or plans_index["corpus_hash"] != index["content_hash"] or plans_index["model_lock_hash"] != lock["content_hash"]):
        raise ValueError("Tokenizer plans do not match the actual corpus and model lock.")
    if [e["story_id"] for e in plans_index["stories"]] != [e["id"] for e in index["stories"]]:
        raise ValueError("Tokenizer plans omit or reorder stories.")
    identity = {"format_version": 1, "config": config, "model_id": model_id, "model_lock_hash": lock["content_hash"],
                "corpus_hash": index["content_hash"], "tokenization_hash": plans_index["content_hash"],
                "code_hashes": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("extraction.py", "tokenization.py", "extraction_inputs.py", "model_registry.py")},
                "packages": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "accelerate", "tokenizers", "numpy", "h5py", "safetensors", "huggingface-hub")}}
    destination = root / config["output"] / model_id
    destination.mkdir(parents=True, exist_ok=True)
    run_hash = object_hash(identity)
    with exclusive_run(destination / "RUNNING.lock"):
        prepare_run_record(destination, identity)
        backbone = None
        records = []
        for entry in index["stories"]:
            story = checked_story(corpus, entry)
            plan = read_json(plans_root / (entry["id"] + ".json"))
            expected_plan = next(e["plan_hash"] for e in plans_index["stories"] if e["story_id"] == entry["id"])
            if (plan["story_hash"] != entry["content_hash"] or plan["model_revision"] != model["revision"]
                    or plan["content_hash"] != expected_plan
                    or object_hash({k:v for k,v in plan.items() if k != "content_hash"}) != expected_plan):
                raise ValueError("Tokenizer plan provenance changed.")
            final = destination / (entry["id"] + ".h5")
            if final.exists():
                with h5py.File(final, "r") as file:
                    if file.attrs["run_hash"] != run_hash or file.attrs["plan_hash"] != expected_plan or not file.attrs["complete"]:
                        raise ValueError("Existing extraction differs from this run: " + str(final))
                print("CACHED", model_id, entry["id"], flush=True)
            else:
                if backbone is None:
                    backbone, runtime = load_backbone(snapshot, model, config)
                    runtime_json = runtime_metadata_json(runtime)
                print("EXTRACT", model_id, entry["id"], len(plan["input_ids"]), "tokens;",
                      len(plan["independent_prefixes"]), "independent prefix forwards", flush=True)
                started = time.monotonic()
                # A partial story is recomputed to rebuild its exact recurrent/KV
                # cache; completed stories are immutable and reused.
                partial = destination / (entry["id"] + ".partial.h5")
                if partial.is_symlink() or final.is_symlink():
                    raise ValueError("Feature output must not be a symlink.")
                with h5py.File(partial, "w") as file:
                    file.attrs.update(run_hash=run_hash, plan_hash=expected_plan, story_hash=entry["content_hash"],
                                      model_revision=model["revision"], complete=False, runtime_json=runtime_json)
                    file.attrs["layer_convention"] = "0=embedding; 1..L=Transformers hidden_states, final entry after final normalization"
                    written = {}
                    for kind in ("words", "units"):
                        n = len(story[kind])
                        file.create_dataset(kind, shape=(model["num_hidden_layers"] + 1, n, model["hidden_size"]),
                                            dtype="float32", chunks=(1, min(64, n), model["hidden_size"]),
                                            compression="lzf", shuffle=True, fletcher32=True, fillvalue=np.nan)
                        written[kind] = np.zeros(n, dtype=bool)
                    canonical = [e for e in plan["endpoints"] if e["canonical_prefix"]]
                    evaluate_sequence(backbone, plan["input_ids"], canonical, file, model, config["chunk_tokens"], written)
                    for end, ids in plan["independent_prefixes"].items():
                        selected = [e for e in plan["endpoints"] if not e["canonical_prefix"] and e["char_end"] == int(end)]
                        evaluate_sequence(backbone, ids, selected, file, model, config["chunk_tokens"], written)
                    if not all(mask.all() for mask in written.values()):
                        raise RuntimeError("Some real extraction endpoints were not written.")
                    file.attrs["wall_seconds"] = time.monotonic() - started
                    file.attrs["complete"] = True
                    file.flush()
                os.replace(partial, final)
                print("SAVED", final, flush=True)
            records.append({"story_id": entry["id"], "file": final.name, "bytes": final.stat().st_size})
        save_json(destination / "complete.json", {"run_hash": run_hash, "stories": records,
                   "completed_utc": datetime.now(timezone.utc).isoformat()})
