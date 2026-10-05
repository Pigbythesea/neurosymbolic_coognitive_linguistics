"""Checkpointed source-first annotation with typed, field-specific corrections."""
import hashlib

from .annotation import check_events, utcnow
from .annotation_jobs import StoryJob, DeferredUnit, request_files
from .annotation_source import (Annotation, CompilationIssues, prompt_base,
    response_schema, compile_annotation, apply_corrections as apply_legacy_corrections)
from .annotation_completion import PROTOCOL, correction_request, apply_corrections
from .annotation_reuse import import_original_draft, decode_import
from .io import object_hash, read_json, save_json


def decode_request(output, metadata_path, seen=None):
    """Replay the actual saved draft/correction chain, never a synthesized response."""
    seen = set() if seen is None else seen
    metadata_path = metadata_path.resolve()
    if not metadata_path.is_relative_to(output.resolve()) or metadata_path in seen:
        raise RuntimeError("Invalid annotation correction provenance path.")
    seen.add(metadata_path)
    metadata = read_json(metadata_path)
    raw = metadata_path.with_name(metadata_path.stem + ".response.json")
    value = read_json(raw)
    context = metadata["protocol_context"]
    if context["protocol"] not in (PROTOCOL, "joint-source-v4"):
        raise RuntimeError("Saved request uses a different annotation protocol.")
    schema = read_json(metadata_path.with_name(metadata_path.stem + ".schema.json"))
    if object_hash(schema) != metadata["response_schema_hash"]:
        raise RuntimeError("Saved request schema changed.")
    if metadata["request_kind"] == "source_import":
        return decode_import(metadata_path, metadata)
    if metadata["request_kind"] == "source_annotation":
        if schema != response_schema():
            raise RuntimeError("Saved annotation schema differs from the running implementation.")
        return Annotation.model_validate(value).model_dump()
    if metadata["request_kind"] != "source_correction":
        raise RuntimeError("Unknown source annotation request type.")
    parent_path = output / context["parent_request"]
    parent_metadata = read_json(parent_path)
    if parent_metadata["protocol_context"]["unit_input_hash"] != context["unit_input_hash"]:
        raise RuntimeError("Correction belongs to a different unit input.")
    parent = decode_request(output, parent_path, seen)
    if object_hash(parent) != context["parent_candidate_hash"]:
        raise RuntimeError("Correction parent annotation changed.")
    tasks = context["correction_tasks"]
    apply = apply_corrections if tasks and tasks[0].get("repair_version") == 2 else apply_legacy_corrections
    return apply(parent, value, tasks)


def load_saved_candidate(output, saved):
    context = saved["protocol_context"]
    metadata_path = output / context["request_metadata"]
    if object_hash(read_json(metadata_path)) != context["request_metadata_hash"]:
        raise RuntimeError("Saved annotation request metadata changed.")
    raw = read_json(output / saved["raw_response"])
    if object_hash(raw) != context["response_hash"]:
        raise RuntimeError("Saved annotation response changed.")
    candidate = decode_request(output, metadata_path)
    if object_hash(candidate) != context["candidate_hash"]:
        raise RuntimeError("Saved annotation correction chain changed.")
    return candidate


class SourceStoryJob(StoryJob):
    def unit(self, story, unit, history, pass_name):
        base = prompt_base(self.instructions, story, unit, history)
        identity = {"run_hash": object_hash(self.identity), "pass": pass_name,
            "story_hash": story["content_hash"], "unit": unit,
            "prompt_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest()}
        input_hash = object_hash(identity)
        destination = self.output / pass_name / story["story_id"] / (unit["id"] + ".json")
        if destination.exists():
            saved = read_json(destination)
            if saved["input_hash"] != input_hash or saved["graph_hash"] != object_hash(saved["graph"]):
                raise RuntimeError("Cached annotation inputs or graph changed: " + str(destination))
            candidate = load_saved_candidate(self.output, saved)
            graph, _ = compile_annotation(candidate, story, unit, history)
            if object_hash(graph.model_dump()) != saved["graph_hash"]:
                raise RuntimeError("Cached source compilation changed: " + str(destination))
            history.append(graph, unit)
            return "CACHED"
        directory = self.output / "attempts" / pass_name / unit["id"] / "source"
        directory.mkdir(parents=True, exist_ok=True)
        import_original_draft(self, directory, base, identity, story, unit, pass_name, PROTOCOL)
        completed = []
        for path in request_files(directory):
            metadata = read_json(path)
            if metadata.get("protocol_context", {}).get("unit_input_hash") != input_hash:
                continue
            try:
                runtime = check_events(path.with_name(path.stem + ".events.jsonl"))
            except (RuntimeError, ValueError, FileNotFoundError):
                continue
            completed.append((path, metadata, {**runtime, "wall_seconds": None}))
        candidate, parent_path, issues, last_error = None, None, [], ""
        limit = self.config["annotation_attempts"]
        for attempt in range(max(limit, len(completed))):
            recovered = attempt < len(completed)
            if recovered:
                metadata_path, metadata, runtime = completed[attempt]
                raw = metadata_path.with_name(metadata_path.stem + ".response.json")
            else:
                context = {"protocol": PROTOCOL, "unit_input_hash": input_hash, "attempt": attempt + 1}
                if candidate is None:
                    prompt, schema, kind = base, response_schema(), "source_annotation"
                    if last_error:
                        prompt += "\nThe previous response was not a valid annotation object. Return the supplied JSON schema.\n" + last_error
                else:
                    if not issues:
                        raise DeferredUnit("The draft needs inspection; no field-specific correction is available. " + last_error)
                    prompt, schema, tasks = correction_request(base, candidate, issues)
                    if last_error:
                        prompt += "\nPrevious correction error: " + last_error
                    kind = "source_correction"
                    context.update(parent_request=parent_path.relative_to(self.output).as_posix(),
                        parent_candidate_hash=object_hash(candidate), correction_tasks=tasks)
                metadata, raw, runtime = self.request(directory, prompt, input_hash, None,
                    response_schema=schema, request_kind=kind, context=context)
                metadata_path = raw.with_name(raw.name.removesuffix(".response.json") + ".json")
            try:
                proposed = decode_request(self.output, metadata_path)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                last_error = str(error)
                save_json(raw.with_name(raw.stem + ".validation.json"),
                    {"valid": False, "kind": "response_format", "error": last_error, "recorded_utc": utcnow()})
                print(f"FORMAT {unit['id']}: {last_error[:220]}", flush=True)
                continue
            candidate, parent_path = proposed, metadata_path
            try:
                graph, provenance = compile_annotation(candidate, story, unit, history)
            except CompilationIssues as error:
                issues, last_error = error.issues, ""
                save_json(raw.with_name(raw.stem + ".validation.json"),
                    {"valid": False, "kind": "source_or_reference", "issues": issues, "recorded_utc": utcnow()})
                print(f"CORRECT {unit['id']}: {len(issues)} field(s); {str(error).splitlines()[0][:220]}", flush=True)
                continue
            except ValueError as error:
                last_error, issues = str(error), []
                save_json(raw.with_name(raw.stem + ".validation.json"),
                    {"valid": False, "kind": "structural", "error": last_error, "recorded_utc": utcnow()})
                raise DeferredUnit("Unmapped structural error; preserving draft for inspection. " + last_error) from error
            context = {**metadata["protocol_context"], **provenance,
                "response_hash": object_hash(read_json(raw)), "candidate_hash": object_hash(candidate),
                "request_metadata": metadata_path.relative_to(self.output).as_posix(),
                "request_metadata_hash": object_hash(metadata),
                "validation": "mechanical_only", "semantic_review": "not_performed",
                "antecedent_policy": "latest_preceding_mention_of_annotated_identity"}
            self.store(destination, identity, graph, [], {**metadata, "protocol_context": context},
                       raw, runtime, recovered=recovered)
            history.append(graph, unit)
            return "RECOVERED" if recovered else "SAVED"
        detail = last_error or ("\n".join(i["message"] for i in issues))
        raise DeferredUnit(f"{max(limit, len(completed))} completed responses used; draft and corrections retained. " + detail)
