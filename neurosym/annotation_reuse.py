"""Reuse only original drafts whose complete generation inputs still match.

Original files are copied byte-for-byte with explicit import provenance. Repaired
graphs are never adopted, and a changed history makes later drafts ineligible.
"""
import hashlib
import shutil

from .annotation import check_events, utcnow
from .annotation_jobs import request_files
from .annotation_source import Annotation, response_schema
from .io import object_hash, read_json, save_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_original_draft(archive, *, base, config, corpus_hash, story, unit, pass_name):
    if (archive / "RUNNING.lock").exists():
        raise RuntimeError("The source annotation run must be stopped before its drafts can be reused.")
    run_path = archive / "run.json"
    if not run_path.is_file():
        raise RuntimeError("Configured draft source is missing its run provenance: " + str(archive))
    run = read_json(run_path)
    if run["corpus_hash"] != corpus_hash or any(run["configuration"][k] != config[k]
                                               for k in ("model", "reasoning_effort")):
        raise RuntimeError("Draft source uses a different corpus, model or reasoning setting.")
    prompt_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
    identity = {"run_hash": object_hash(run), "pass": pass_name, "story_hash": story["content_hash"],
                "unit": unit, "prompt_sha256": prompt_hash}
    for path in request_files(archive / "attempts" / pass_name / unit["id"] / "source"):
        metadata = read_json(path)
        if (metadata.get("request_kind") != "source_annotation" or
                metadata.get("request_sha256") != prompt_hash or
                metadata.get("protocol_context", {}).get("unit_input_hash") != object_hash(identity)):
            continue
        files = {"metadata": path, "run": run_path,
            **{name: path.with_name(path.stem + suffix) for name, suffix in
               (("response", ".response.json"), ("schema", ".schema.json"),
                ("prompt", ".prompt.txt"), ("events", ".events.jsonl"))}}
        if not all(p.is_file() for p in files.values()):
            continue
        if files["prompt"].read_text(encoding="utf-8") != base or read_json(files["schema"]) != response_schema():
            continue
        try:
            runtime = check_events(files["events"])
            Annotation.model_validate(read_json(files["response"]))
        except (RuntimeError, ValueError):
            continue
        return {"files": files, "metadata": metadata, "identity": identity, "runtime": runtime}
    return None


def import_original_draft(job, directory, base, identity, story, unit, pass_name, protocol):
    source = job.config.get("reuse_original_drafts_from")
    if not source or request_files(directory):
        return False
    root = job.workspace.parent.parent
    archive = (root / source).resolve()
    if archive == job.output.resolve():
        raise ValueError("Draft reuse source cannot be the destination run.")
    found = find_original_draft(archive, base=base, config=job.config,
        corpus_hash=job.index["content_hash"], story=story, unit=unit, pass_name=pass_name)
    if found is None:
        return False
    stem = "request-0000"
    destinations = {"metadata": directory / (stem + ".origin.json"),
                    "run": directory / (stem + ".origin-run.json"),
                    **{name: directory / (stem + suffix) for name, suffix in
                       (("response", ".response.json"), ("schema", ".schema.json"),
                        ("prompt", ".prompt.txt"), ("events", ".events.jsonl"))}}
    hashes = {name: digest(path) for name, path in found["files"].items()}
    for name, path in found["files"].items():
        shutil.copyfile(path, destinations[name])
        if digest(destinations[name]) != hashes[name]:
            raise RuntimeError("An original draft changed during import.")
    previous = found["metadata"]
    metadata = {"request_kind": "source_import", "imported_utc": utcnow(),
        "input_hash": object_hash(identity), "cli": previous["cli"],
        "implementation_hash": job.implementation_hash, "request_sha256": previous["request_sha256"],
        "response_schema_hash": object_hash(response_schema()),
        "protocol_context": {"protocol": protocol, "unit_input_hash": object_hash(identity), "attempt": 1,
            "origin_identity": found["identity"], "origin_archive": str(archive), "new_model_request": False,
            "origin_files": {name: {"path": p.name, "sha256": hashes[name]}
                             for name, p in destinations.items()}}}
    save_json(directory / (stem + ".json"), metadata)
    job.progress("reusing_original_draft", source_run=str(archive))
    print(f"REUSE ORIGINAL DRAFT {unit['id']}: source/history/prompt/schema unchanged; no model request.", flush=True)
    return True


def decode_import(metadata_path, metadata):
    context, directory = metadata["protocol_context"], metadata_path.parent
    files = {}
    for name, spec in context["origin_files"].items():
        path = (directory / spec["path"]).resolve()
        if path.parent != directory.resolve() or digest(path) != spec["sha256"]:
            raise RuntimeError("Imported draft provenance changed.")
        files[name] = path
    if set(files) != {"metadata", "run", "response", "schema", "prompt", "events"}:
        raise RuntimeError("Incomplete imported draft provenance.")
    origin, run = read_json(files["metadata"]), read_json(files["run"])
    identity = context["origin_identity"]
    if (origin.get("request_kind") != "source_annotation" or identity["run_hash"] != object_hash(run) or
            origin["protocol_context"]["unit_input_hash"] != object_hash(identity) or
            origin["request_sha256"] != metadata["request_sha256"] or
            identity["prompt_sha256"] != metadata["request_sha256"] or
            origin["response_schema_hash"] != object_hash(read_json(files["schema"])) or
            hashlib.sha256(files["prompt"].read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                != metadata["request_sha256"] or read_json(files["schema"]) != response_schema()):
        raise RuntimeError("Imported draft generation inputs changed.")
    check_events(files["events"])
    return Annotation.model_validate(read_json(files["response"])).model_dump()
