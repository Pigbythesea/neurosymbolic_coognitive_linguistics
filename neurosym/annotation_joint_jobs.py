"""Single joint generation with checkpointed, mechanical-only correction requests."""
import hashlib
import json

from .annotation import check_events, utcnow
from .annotation_jobs import StoryJob, DeferredUnit, request_files
from .annotation_joint import PROTOCOL, prompt_base, response_schema, compile_annotation
from .io import object_hash, read_json, save_json


class JointStoryJob(StoryJob):
    def unit(self, story, unit, history, pass_name):
        base = prompt_base(self.instructions, story, unit, history)
        identity = {"run_hash": object_hash(self.identity), "pass": pass_name, "story_hash": story["content_hash"],
                    "unit": unit, "prompt_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest()}
        input_hash = object_hash(identity)
        destination = self.output / pass_name / story["story_id"] / (unit["id"] + ".json")
        if destination.exists():
            saved = read_json(destination)
            if (saved["input_hash"] != input_hash or saved["graph_hash"] != object_hash(saved["graph"])
                    or saved.get("protocol_context", {}).get("protocol") != PROTOCOL):
                raise ValueError("Cached joint annotation inputs or graph changed: " + str(destination))
            raw = read_json(self.output / saved["raw_response"])
            if object_hash(raw) != saved["protocol_context"]["response_hash"]:
                raise ValueError("Cached raw annotation changed: " + str(destination))
            graph, _ = compile_annotation(raw, story, unit, history)
            if object_hash(graph.model_dump()) != saved["graph_hash"]:
                raise ValueError("Cached annotation no longer compiles identically: " + str(destination))
            history.append(graph, unit)
            return "CACHED"
        directory = self.output / "attempts" / pass_name / unit["id"] / "joint"
        directory.mkdir(parents=True, exist_ok=True)
        schema, feedback, candidate = response_schema(), "", None
        completed = []
        for path in request_files(directory):
            metadata = read_json(path)
            if metadata.get("protocol_context", {}).get("unit_input_hash") != input_hash:
                continue
            raw = path.with_name(path.stem + ".response.json")
            log = path.with_name(path.stem + ".events.jsonl")
            if not log.is_file():
                continue
            try:
                runtime = check_events(log)
            except (RuntimeError, ValueError):
                continue
            if object_hash(read_json(path.with_name(path.stem + ".schema.json"))) != object_hash(schema):
                raise ValueError("Checkpointed joint schema changed.")
            completed.append((metadata, raw, {**runtime, "wall_seconds": None}))
        # Budget counts completed responses across restarts/sweeps. Malformed
        # JSON is a mechanical failure, not a reason to abort the whole corpus.
        limit = self.config["annotation_attempts"]
        for attempt in range(max(limit, len(completed))):
            recovered = attempt < len(completed)
            if recovered:
                metadata, raw, runtime = completed[attempt]
            else:
                prompt = base
                if feedback:
                    prompt += ("\nMECHANICAL CORRECTION ONLY: fix the reported compiler errors. Preserve other semantic decisions. "
                               "Return the entire corrected annotation, not a patch or review.\nERRORS:\n" + feedback)
                    if candidate is not None:
                        prompt += "\nPREVIOUS RESPONSE:\n" + json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
                metadata, raw, runtime = self.request(directory, prompt, input_hash, None,
                    response_schema=schema, request_kind="joint_annotation",
                    context={"protocol": PROTOCOL, "unit_input_hash": input_hash, "attempt": attempt + 1})
            try:
                candidate = read_json(raw)
                graph, anchors = compile_annotation(candidate, story, unit, history)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                feedback = str(error)
                save_json(raw.with_name(raw.stem + ".validation.json"),
                          {"valid": False, "kind": "mechanical", "error": feedback, "recorded_utc": utcnow()})
                print(f"FORMAT/REFERENCE REPAIR {unit['id']}: {feedback[:240]}", flush=True)
                continue
            metadata = {**metadata, "protocol_context": {**metadata["protocol_context"],
                "response_hash": object_hash(candidate), "source_anchors": anchors,
                "validation": "mechanical_only", "semantic_review": "not_performed",
                "antecedent_policy": "latest_preceding_mention_of_annotated_identity"}}
            self.store(destination, identity, graph, [], metadata, raw, runtime, recovered=recovered)
            history.append(graph, unit)
            return "RECOVERED" if recovered else "SAVED"
        raise DeferredUnit(f"{len(completed) if len(completed) > limit else limit} completed responses exhausted the mechanical repair budget. "
                           "Other stories continue. Inspect this unit before authorizing additional requests. " + feedback)
