"""Ground, bind, and review before committing a unit to a fresh story history."""

import hashlib
import json

from .annotation import utcnow
from .annotation_catalog import exact_keys
from .annotation_catalog_jobs import CatalogStoryJob
from .annotation_jobs import DeferredUnit
from .annotation_anchored import (PROTOCOL, SourceText, prompt_base, ground_schema, compile_grounding,
    grounding_prompt, bind_schema, compile_bindings, binding_prompt, review_schema,
    semantic_review_prompt, validate_review)
from .graphs import GraphDelta
from .io import object_hash, read_json, save_json


class AnchoredStoryJob(CatalogStoryJob):
    protocol = PROTOCOL

    def unit(self, story, unit, history, pass_name):
        destination = self.output / pass_name / story["story_id"] / (unit["id"] + ".json")
        source = SourceText(story, unit)
        base = prompt_base(self.instructions, story, unit, history)
        identity = {"run_hash": object_hash(self.identity), "pass": pass_name, "story_hash": story["content_hash"],
                    "unit": unit, "prompt_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest()}
        unit_hash = object_hash(identity)
        if destination.exists():
            saved = read_json(destination)
            context = saved.get("protocol_context", {})
            if (saved["input_hash"] != unit_hash or saved["graph_hash"] != object_hash(saved["graph"])
                    or context.get("protocol") != PROTOCOL
                    or context.get("reviewed_graph_hash") != saved["graph_hash"]
                    or context.get("semantic_check", {}).get("status") != "passed"):
                raise ValueError("Cached unit lacks matching source-anchored inputs and semantic review: " + str(destination))
            graph = GraphDelta.model_validate(saved["graph"])
            review_file = self.output / context["semantic_check"]["raw_response"]
            review = read_json(review_file)
            if object_hash(review) != context["semantic_check"]["response_hash"]:
                raise ValueError("Cached semantic review changed: " + str(review_file))
            failures, _ = validate_review(review, source, graph)
            if failures:
                raise ValueError("A rejected semantic review was saved as accepted: " + str(destination))
            history.append(graph, unit)
            return "CACHED"

        directory = self.output / "attempts" / pass_name / unit["id"] / "anchored"
        directory.mkdir(parents=True, exist_ok=True)
        trace, feedback, last_failure = [], "", ""

        def request(stage, prompt, schema, **context):
            response, metadata, raw, runtime = self.stage(directory, stage, prompt, schema, unit_hash, **context)
            trace.append({"stage": stage, "raw_response": raw.relative_to(self.output).as_posix(),
                          "response_hash": object_hash(response), "runtime": runtime})
            return response, metadata, raw, runtime

        def reject(raw, phase, message):
            save_json(raw.with_name(raw.stem + ".validation.json"),
                      {"valid": False, "phase": phase, "error": message, "recorded_utc": utcnow()})
            print(f"{phase.upper()} CORRECTION {unit['id']}: {message[:220]}", flush=True)

        for gi in range(self.config.get("grounding_attempts", 3)):
            print(f"GROUND {unit['id']} version={gi + 1} source=exact_quotes", flush=True)
            draft, _, ground_raw, _ = request("ground", grounding_prompt(base, history, feedback), ground_schema(history))
            try:
                catalog, ground_anchors = compile_grounding(draft, source, history)
            except (ValueError, KeyError, TypeError) as error:
                last_failure = str(error)
                feedback = last_failure + "\nPREVIOUS GROUNDING:\n" + json.dumps(draft, ensure_ascii=False)
                reject(ground_raw, "ground", last_failure)
                continue

            link_feedback, reground = "", False
            for bi in range(self.config.get("binding_attempts", 2)):
                print(f"BIND {unit['id']} version={bi + 1}", flush=True)
                response, _, raw, _ = request("bind", binding_prompt(base, catalog, source, link_feedback),
                    bind_schema(catalog), grounding_response_hash=object_hash(draft))
                try:
                    exact_keys(response, ["result"], "binding response")
                    result = response["result"]
                    if isinstance(result, dict) and set(result) == {"grounding_correction"}:
                        reason = result["grounding_correction"]
                        if not isinstance(reason, str) or not reason.strip():
                            raise ValueError("Grounding correction requires a concrete explanation.")
                        last_failure = reason
                        feedback = reason + "\nPREVIOUS GROUNDING:\n" + json.dumps(draft, ensure_ascii=False)
                        reground = True
                        reject(raw, "ground", reason)
                        break
                    exact_keys(result, ["links"], "binding result")
                    graph, link_anchors = compile_bindings(result["links"], source, catalog)
                except (ValueError, KeyError, TypeError) as error:
                    last_failure = str(error)
                    link_feedback = last_failure + "\nPREVIOUS LINKS:\n" + json.dumps(response, ensure_ascii=False)
                    reject(raw, "bind", last_failure)
                    continue

                review_feedback, reviewed = "", False
                for ri in range(2):
                    print(f"REVIEW {unit['id']} version={ri + 1}", flush=True)
                    review, metadata, review_raw, runtime = request("review",
                        semantic_review_prompt(base, source, graph, review_feedback), review_schema(graph),
                        grounding_response_hash=object_hash(draft), binding_response_hash=object_hash(response),
                        reviewed_graph_hash=object_hash(graph.model_dump()))
                    try:
                        failures, review_anchors = validate_review(review, source, graph)
                    except (ValueError, KeyError, TypeError) as error:
                        last_failure = str(error)
                        review_feedback = last_failure + "\nPREVIOUS REVIEW:\n" + json.dumps(review, ensure_ascii=False)
                        reject(review_raw, "review", last_failure)
                        continue
                    reviewed = True
                    break
                if not reviewed:
                    raise DeferredUnit("Semantic review format/evidence could not be validated: " + last_failure)
                if failures:
                    last_failure = json.dumps(failures, ensure_ascii=False)
                    reject(review_raw, "semantic", last_failure)
                    if any(item["family"] == "grounding" for item in failures):
                        feedback = last_failure + "\nPREVIOUS GROUNDING:\n" + json.dumps(draft, ensure_ascii=False)
                        reground = True
                        break
                    link_feedback = last_failure + "\nPREVIOUS LINKS:\n" + json.dumps(response, ensure_ascii=False)
                    continue

                check = {"status": "passed", "kind": "same_model_semantic_review", "human_review": "pending",
                         "raw_response": review_raw.relative_to(self.output).as_posix(), "response_hash": object_hash(review)}
                metadata = {**metadata, "protocol_context": {**metadata["protocol_context"],
                    "protocol": PROTOCOL, "stages": trace, "source_anchors": ground_anchors + link_anchors,
                    "review_evidence": review_anchors, "semantic_check": check}}
                self.store(destination, identity, graph, [], metadata, review_raw, runtime)
                history.append(graph, unit)
                return "SAVED"
            if not reground:
                break
        raise DeferredUnit(last_failure or "Source grounding or semantic review remained unresolved.")
