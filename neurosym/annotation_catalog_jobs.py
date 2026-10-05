"""Checkpointed two-stage annotation; no mutation of accepted earlier graphs."""

import hashlib
import json

from .annotation import make_prompt, check_events, utcnow
from .annotation_jobs import StoryJob, DeferredUnit, request_files
from .annotation_catalog import (PROTOCOL, BINDING_TRANSPORT, Catalog, grounding_schema, ground_prompt,
                                 binding_schema, bind_prompt, exact_keys)
from .graphs import GraphDelta
from .annotation_grounding_checks import validate_grounding_spans
from .io import object_hash, read_json, save_json


class CatalogStoryJob(StoryJob):
    protocol = PROTOCOL

    def stage(self, directory, stage, prompt, schema, unit_hash, **context):
        identity = {"unit_input_hash": unit_hash, "protocol": self.protocol, "stage": stage,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "schema_hash": object_hash(schema)}
        stage_hash = object_hash(identity)
        # Checkpoint successful transport, including responses with a semantic
        # error. Replaying them reconstructs corrective feedback without paying
        # for the same request again after interruption.
        for path in reversed(request_files(directory)):
            metadata = read_json(path)
            if metadata.get("input_hash") != stage_hash:
                continue
            raw = path.with_name(path.stem + ".response.json")
            log = path.with_name(path.stem + ".events.jsonl")
            if not raw.is_file() or not log.is_file():
                continue
            try:
                runtime = check_events(log)
                response = read_json(raw)
            except (RuntimeError, ValueError):
                continue
            if object_hash(read_json(path.with_name(path.stem + ".schema.json"))) != identity["schema_hash"]:
                raise ValueError("Saved catalog request schema changed: " + str(path))
            self.progress("cached_" + stage)
            return response, metadata, raw, {**runtime, "wall_seconds": None}
        self.progress(stage)
        metadata, raw, runtime = self.request(directory, prompt, stage_hash, None,
            response_schema=schema, request_kind=stage,
            context={**identity, **context})
        return read_json(raw), metadata, raw, runtime

    def unit(self, story, unit, history, pass_name):
        destination = self.output / pass_name / story["story_id"] / (unit["id"] + ".json")
        base = make_prompt(self.instructions, story, unit, history)
        input_identity = {"run_hash": object_hash(self.identity), "pass": pass_name,
                          "story_hash": story["content_hash"], "unit": unit,
                          "prompt_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest()}
        unit_hash = object_hash(input_identity)
        if destination.exists():
            saved = read_json(destination)
            if saved["input_hash"] != unit_hash or saved["graph_hash"] != object_hash(saved["graph"]):
                raise ValueError("Completed annotation or its input changed: " + str(destination))
            history.append(GraphDelta.model_validate(saved["graph"]), unit)
            return "CACHED"
        directory = self.output / "attempts" / pass_name / unit["id"] / "catalog"
        directory.mkdir(parents=True, exist_ok=True)
        feedback, failure = "", ""
        grounding_requests, binding_requests = [], []
        # Two grounded versions, each with at most two link responses. This is
        # a semantic correction budget, not a loop of unconstrained JSON patches.
        for grounding_round in range(2):
            print(f"GROUND {unit['id']} version={grounding_round + 1}", flush=True)
            draft, ground_meta, ground_raw, ground_runtime = self.stage(
                directory, "ground", ground_prompt(base, history, feedback),
                grounding_schema(history, unit), unit_hash)
            grounding_requests.append({"raw_response": ground_raw.relative_to(self.output).as_posix(),
                                       "response_hash": object_hash(draft), "runtime": ground_runtime})
            try:
                catalog = Catalog(draft, history, unit)
                validate_grounding_spans(catalog, story)
            except ValueError as error:
                failure = str(error)
                feedback = failure + "\nPREVIOUS GROUNDING:\n" + json.dumps(draft)
                save_json(ground_raw.with_name(ground_raw.stem + ".validation.json"),
                          {"valid": False, "error": failure})
                print(f"GROUND CORRECTION {unit['id']}: {failure.splitlines()[0]}", flush=True)
                continue
            link_feedback = ""
            for binding_round in range(2):
                print(f"BIND {unit['id']} version={binding_round + 1}", flush=True)
                response, metadata, raw, runtime = self.stage(
                    directory, "bind", bind_prompt(base, catalog, link_feedback), binding_schema(catalog),
                    unit_hash, grounding_response=ground_raw.relative_to(self.output).as_posix(),
                    grounding_response_hash=object_hash(draft), binding_transport=BINDING_TRANSPORT)
                binding_requests.append({"raw_response": raw.relative_to(self.output).as_posix(),
                                         "response_hash": object_hash(response), "runtime": runtime})
                try:
                    exact_keys(response, ["result"], "binding response")
                    result = response["result"]
                    if isinstance(result, dict) and set(result) == {"correction_required"}:
                        explanation = result["correction_required"]
                        if not isinstance(explanation, str) or not explanation.strip():
                            raise ValueError("Grounding correction must explain the text-supported change.")
                        failure = explanation
                        feedback = explanation + "\nPREVIOUS GROUNDING:\n" + json.dumps(draft)
                        print(f"REGROUND {unit['id']}: {explanation[:200]}", flush=True)
                        break
                    exact_keys(result, ["links"], "binding result")
                    graph = catalog.compile(result["links"])
                except (ValueError, KeyError, TypeError) as error:
                    failure = str(error)
                    # Keep the invalid decisions alongside the error so the next
                    # response can correct a concrete candidate, not guess what
                    # a previous request generated.
                    link_feedback = failure + "\nPREVIOUS RESPONSE:\n" + json.dumps(response)
                    save_json(raw.with_name(raw.stem + ".validation.json"), {"valid": False, "error": failure})
                    print(f"LINK CORRECTION {unit['id']}: {failure.splitlines()[0]}", flush=True)
                    continue
                metadata = {**metadata, "protocol_context": {
                    **metadata["protocol_context"], "grounding_requests": grounding_requests,
                    "binding_requests": binding_requests,
                    "compiler": PROTOCOL, "semantic_audit_status": "unreviewed"}}
                self.store(destination, input_identity, graph, catalog.derivations, metadata, raw, runtime)
                history.append(graph, unit)
                return "SAVED"
            else:
                # Do not redo a valid grounding to repair a scope/temporal cycle.
                break
        raise DeferredUnit(failure or "No valid semantic binding within the correction budget.")
