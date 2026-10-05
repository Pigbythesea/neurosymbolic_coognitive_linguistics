"""Resumable story jobs with targeted repairs, retry limits, and persistent progress."""

import hashlib
import json
from pathlib import Path
import re
import time
from pydantic import ValidationError

from .annotation import codex_command, check_events, invoke_codex, make_prompt, utcnow, verified_story
from .annotation_repair import apply_graph_edits, normalize_graph, repair_prompt
from .graphs import GraphDelta, GraphHistory
from .io import object_hash, read_json, save_json


def validate_policy(policy: dict) -> None:
    if policy.get("format_version") != 1:
        raise ValueError("Unsupported annotation runtime policy.")
    for name in ("max_patch_requests_per_visit", "story_sweeps", "heartbeat_seconds"):
        if type(policy.get(name)) is not int or policy[name] < 1:
            raise ValueError("Runtime policy needs a positive integer: " + name)
    delays = policy.get("transient_retry_delays_seconds")
    if not isinstance(delays, list) or any(type(delay) is not int or not 0 < delay <= 60 for delay in delays):
        raise ValueError("Retry delays must be a list of positive seconds up to 60.")


def request_files(directory: Path) -> list[Path]:
    return sorted((path for path in directory.glob("request-*.json")
                   if path.stem.removeprefix("request-").isdigit()),
                  key=lambda path: int(path.stem.removeprefix("request-")))


def decode_response(metadata: dict, raw: Path) -> dict:
    if not raw.is_file():
        raise ValueError("Codex completed without a structured response file.")
    response = read_json(raw)
    if metadata.get("request_kind") == "repair":
        base = metadata["repair_base_graph"]
        if object_hash(base) != metadata["repair_base_hash"]:
            raise ValueError("Stored repair input changed.")
        response = apply_graph_edits(base, response, legacy=metadata.get("repair_protocol", 1) == 1)
    if not isinstance(response, dict):
        raise ValueError("Annotation output is not a JSON object.")
    return response


def validate_candidate(candidate: dict, history: GraphHistory, unit: dict):
    graph = GraphDelta.model_validate(candidate)
    graph, repairs = normalize_graph(graph, history, unit)
    history.validate(graph, unit)
    return graph, repairs


def error_count(error: ValueError) -> int:
    return error.error_count() if isinstance(error, ValidationError) else len(str(error).splitlines())


def transient_error(message: str) -> bool:
    lower = message.casefold()
    # Subscription exhaustion and authentication cannot be repaired by repeatedly
    # spending requests. Temporary rate limits/network failures can be retried.
    if any(term in lower for term in ("usage limit", "usage_limit", "quota", "credits", "unauthorized",
                                     "authentication", "sign in", "log in", "401", "403")):
        return False
    return (any(term in lower for term in ("timed out", "timeout", "connection reset", "connection aborted",
                                          "connection error", "error sending request", "temporarily unavailable",
                                          "stream disconnected", "unexpected eof", "rate limit", "rate_limit"))
            or re.search(r"\b(?:429|500|502|503|504)\b|exceeded \d+ seconds", lower) is not None)


class DeferredUnit(Exception):
    pass


class StoryJob:
    def __init__(self, *, output, corpus, config, index, identity, implementation_hash, instructions,
                 codex, cli, workspace, graph_schema_path, repair_schema_path, policy, progress):
        self.output, self.corpus, self.config, self.index = output, corpus, config, index
        self.identity, self.implementation_hash, self.instructions = identity, implementation_hash, instructions
        self.codex, self.cli, self.workspace = codex, cli, workspace
        self.graph_schema_path, self.repair_schema_path = graph_schema_path, repair_schema_path
        self.policy, self.progress = policy, progress

    def request(self, directory, prompt, input_hash, candidate, *, response_schema=None,
                request_kind=None, context=None):
        # This limit was returned by the installed CLI's turn/start interface.
        # Check locally before starting another process or spending a request.
        request_chars = len(prompt) + (len(json.dumps(response_schema)) if response_schema is not None else 0)
        if request_chars > 1_000_000:
            raise RuntimeError(f"Local request-size check: {request_chars:,} prompt/schema characters approach "
                               "the CLI limit of 1,048,576. No model request was started. "
                               "Preserve history and compact the transport before retrying.")
        delays = self.policy["transient_retry_delays_seconds"]
        for retry in range(len(delays) + 1):
            files = request_files(directory)
            serial = int(files[-1].stem.removeprefix("request-")) + 1 if files else 0
            stem = f"request-{serial:04d}"
            raw = directory / (stem + ".response.json")
            log = directory / (stem + ".events.jsonl")
            repair = candidate is not None
            schema = self.repair_schema_path if repair else self.graph_schema_path
            if response_schema is not None:
                schema = directory / (stem + ".schema.json")
                save_json(schema, response_schema)
            command = codex_command(self.codex, self.config, self.workspace, schema, raw)
            metadata = {"started_utc": utcnow(), "command": command, "input_hash": input_hash,
                        "cli": self.cli, "implementation_hash": self.implementation_hash,
                        "request_kind": request_kind or ("repair" if repair else "annotation"),
                        "request_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
            if context:
                metadata["protocol_context"] = context
            if response_schema is not None:
                metadata["response_schema_hash"] = object_hash(response_schema)
            if repair:
                metadata.update(repair_base_graph=candidate, repair_base_hash=object_hash(candidate), repair_protocol=2)
            (directory / (stem + ".prompt.txt")).write_text(prompt, encoding="utf-8")
            save_json(directory / (stem + ".json"), metadata)
            self.progress("request", request=str(log), retry=retry)
            try:
                runtime = invoke_codex(command, prompt, log, self.config["timeout_seconds"],
                                       heartbeat=lambda elapsed: self.progress("waiting_for_model", elapsed_seconds=round(elapsed)),
                                       heartbeat_seconds=self.policy["heartbeat_seconds"])
                save_json(directory / (stem + ".runtime.json"), runtime)
                return metadata, raw, runtime
            except RuntimeError as error:
                save_json(directory / (stem + ".runtime-error.json"), {"error": str(error), "recorded_utc": utcnow()})
                if not transient_error(str(error)) or retry == len(delays):
                    raise
                delay = delays[retry]
                self.progress("retry_wait", retry_in_seconds=delay, error=str(error))
                print(f"TEMPORARY SERVICE ERROR: retrying in {delay}s; completed units retained.", flush=True)
                time.sleep(delay)

    def store(self, destination, input_identity, graph, repairs, metadata, raw, runtime, *, recovered=False):
        saved = {"input_hash": object_hash(input_identity), "input_identity": input_identity,
                 "cli": metadata["cli"], "completed_utc": None if recovered else utcnow(), "runtime": runtime,
                 "available_at_token": input_identity["unit"]["end_token"],
                 "available_at_seconds": input_identity["unit"]["offset_seconds"],
                 "graph": graph.model_dump(), "graph_hash": object_hash(graph.model_dump()),
                 "structural_repairs": repairs, "implementation_hash": self.implementation_hash,
                 "raw_response": raw.relative_to(self.output).as_posix(), "semantic_audit_status": "unreviewed"}
        if recovered:
            saved["recovered_utc"] = utcnow()
        if metadata.get("request_kind") == "repair":
            saved["model_edits"] = read_json(raw)["edits"]
        if metadata.get("protocol_context"):
            saved["protocol_context"] = metadata["protocol_context"]
        save_json(destination, saved)

    def unit(self, story, unit, history, pass_name):
        destination = self.output / pass_name / story["story_id"] / (unit["id"] + ".json")
        prompt = make_prompt(self.instructions, story, unit, history)
        input_identity = {"run_hash": object_hash(self.identity), "pass": pass_name,
                          "story_hash": story["content_hash"], "unit": unit,
                          "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        input_hash = object_hash(input_identity)
        if destination.exists():
            saved = read_json(destination)
            if saved["input_hash"] != input_hash or saved["graph_hash"] != object_hash(saved["graph"]):
                raise ValueError("Completed annotation or its input changed: " + str(destination))
            history.append(GraphDelta.model_validate(saved["graph"]), unit)
            return "CACHED"
        directory = self.output / "attempts" / pass_name / unit["id"]
        directory.mkdir(parents=True, exist_ok=True)
        candidate, error, best_error_count = None, "", float("inf")
        # Reuse a completed response or continue repairing it. A restart never
        # discards the work already spent on this unit and starts from scratch.
        recorded = [(path, read_json(path)) for path in request_files(directory)]
        # Prefer a complete original response that compiles correctly; this avoids
        # unnecessary semantic changes from later repairs of derived bookkeeping.
        recorded = ([item for item in reversed(recorded) if item[1].get("request_kind") != "repair"]
                    + [item for item in reversed(recorded) if item[1].get("request_kind") == "repair"])
        for metadata_path, metadata in recorded:
            if metadata["input_hash"] != input_hash:
                continue
            raw = metadata_path.with_name(metadata_path.stem + ".response.json")
            log = metadata_path.with_name(metadata_path.stem + ".events.jsonl")
            if not raw.is_file() or not log.is_file():
                continue
            try:
                runtime = check_events(log)
            except RuntimeError:
                continue
            try:
                loaded = decode_response(metadata, raw)
            except (ValueError, KeyError, TypeError):
                continue
            try:
                graph, repairs = validate_candidate(loaded, history, unit)
            except ValueError as failure:
                count = error_count(failure)
                if count < best_error_count:
                    candidate, error, best_error_count = loaded, str(failure), count
                continue
            self.store(destination, input_identity, graph, repairs, metadata, raw,
                       {**runtime, "wall_seconds": None}, recovered=True)
            history.append(graph, unit)
            return "RECOVERED"
        protocol_error, protocol_failures, unchanged = "", 0, 0
        for attempt in range(self.policy["max_patch_requests_per_visit"] + (1 if candidate is None else 0)):
            request_prompt = repair_prompt(prompt, candidate, error, history, protocol_error) if candidate is not None else prompt
            metadata, raw, runtime = self.request(directory, request_prompt, input_hash, candidate)
            self.progress("validating")
            try:
                proposed = decode_response(metadata, raw)
            except (ValueError, KeyError, TypeError) as failure:
                protocol_error = str(failure)
                protocol_failures += 1
                save_json(raw.with_name(raw.name.replace(".response.json", ".validation.json")),
                          {"valid": False, "kind": "repair_protocol", "error": protocol_error, "graph_error": error})
                print(f"FORMAT ERROR {unit['id']}: {protocol_error.splitlines()[0]}", flush=True)
                if protocol_failures >= 2:
                    raise RuntimeError("Two malformed repair responses; stopping to avoid wasting more requests. "
                                       "The original graph error is preserved: " + error + "; format error: " + protocol_error)
                continue
            protocol_error, protocol_failures = "", 0
            try:
                graph, repairs = validate_candidate(proposed, history, unit)
            except ValueError as failure:
                count = error_count(failure)
                changed = candidate is None or object_hash(candidate) != object_hash(proposed)
                improved = changed and count <= best_error_count
                if improved:
                    candidate, error, best_error_count = proposed, str(failure), count
                unchanged = 0 if improved else unchanged + 1
                save_json(raw.with_name(raw.name.replace(".response.json", ".validation.json")),
                          {"valid": False, "kind": "graph", "error": str(failure),
                           "retained_graph_error": error, "retained_proposed_candidate": improved})
                print(f"REPAIR {unit['id']}: {str(failure).splitlines()[0]}", flush=True)
                if unchanged >= 2:
                    break
                continue
            self.store(destination, input_identity, graph, repairs, metadata, raw, runtime)
            history.append(graph, unit)
            return "SAVED"
        raise DeferredUnit(error or "No valid annotation after this visit's repair budget.")


def run_jobs(*, output, corpus, config, index, identity, implementation_hash, instructions,
             codex, cli, workspace, graph_schema_path, repair_schema_path, policy, passes,
             job_type=StoryJob):
    validate_policy(policy)
    states = {name: {entry["id"]: {"status": "pending",
                     "completed_units": len(list((output / name / entry["id"]).glob(entry["id"] + "_u*.json"))),
                     "total_units": entry["units"]}
                     for entry in index["stories"]} for name in passes}
    failures, current = {}, {}
    status = "running"
    started = utcnow()

    def progress(phase, **details):
        current.update(phase=phase, **details)
        report = {"format_version": 1, "run_hash": object_hash(identity), "implementation_hash": implementation_hash,
                  "status": status, "started_utc": started, "updated_utc": utcnow(), "current": dict(current),
                  "completed_units": sum(item["completed_units"] for branch in states.values() for item in branch.values()),
                  "total_units": sum(item["total_units"] for branch in states.values() for item in branch.values()),
                  "stories": states, "unresolved": list(failures.values())}
        save_json(output / "progress.json", report)
        if phase == "waiting_for_model":
            print(f"WAITING {current.get('unit_id', '')}: {details.get('elapsed_seconds', 0)}s; "
                  f"saved {report['completed_units']}/{report['total_units']}", flush=True)
        return report

    job = job_type(output=output, corpus=corpus, config=config, index=index, identity=identity,
                   implementation_hash=implementation_hash, instructions=instructions, codex=codex, cli=cli,
                   workspace=workspace, graph_schema_path=graph_schema_path, repair_schema_path=repair_schema_path,
                   policy=policy, progress=progress)
    progress("starting")
    try:
        for sweep in range(1, policy["story_sweeps"] + 1):
            for pass_name in passes:
                for entry in index["stories"]:
                    state = states[pass_name][entry["id"]]
                    if state["status"] == "complete":
                        continue
                    story = verified_story(corpus, entry)
                    history = GraphHistory()
                    state.update(status="running")
                    for unit in story["units"]:
                        current.clear()
                        current.update(pass_name=pass_name, story_id=story["story_id"], unit_id=unit["id"], sweep=sweep)
                        progress("annotating")
                        print(f"ANNOTATE {pass_name}/{unit['id']} sweep={sweep} "
                              f"tokens={unit['start_token']}:{unit['end_token']}", flush=True)
                        key = pass_name + "/" + story["story_id"]
                        try:
                            result = job.unit(story, unit, history, pass_name)
                        except DeferredUnit as failure:
                            failure_record = {"pass": pass_name, "story_id": story["story_id"], "unit_id": unit["id"],
                                              "error": str(failure), "sweep": sweep, "status": "deferred",
                                              "remaining_story_units": len(story["units"]) - state["completed_units"],
                                              "attempts": str(output / "attempts" / pass_name / unit["id"])}
                            failures[key] = failure_record
                            state.update(status="deferred", blocked_at=unit["id"])
                            save_json(output / "deferred" / pass_name / (unit["id"] + ".json"), failure_record)
                            progress("story_deferred")
                            print(f"DEFERRED {pass_name}/{unit['id']}; continuing independent stories.", flush=True)
                            break
                        if result != "CACHED":
                            state["completed_units"] += 1
                        if key in failures and failures[key]["unit_id"] == unit["id"]:
                            resolved = failures.pop(key)
                            save_json(output / "deferred" / pass_name / (unit["id"] + ".json"),
                                      {**resolved, "status": "resolved", "resolved_utc": utcnow()})
                        report = progress("saved")
                        print(f"[{report['completed_units']}/{report['total_units']}] {result} {pass_name}/{unit['id']}", flush=True)
                    else:
                        state.update(status="complete")
                        state.pop("blocked_at", None)
                if all(item["status"] == "complete" for item in states[pass_name].values()):
                    save_json(output / f"complete.{pass_name}.json", {"completed_utc": utcnow(),
                              "run_hash": object_hash(identity), "units": sum(entry["units"] for entry in index["stories"])})
            if not failures:
                break
        status = "needs_review" if failures else "complete"
        report = progress("finished")
        save_json(output / "failures.json", {"run_hash": object_hash(identity), "updated_utc": utcnow(),
                                            "unresolved": list(failures.values())})
        print(f"ANNOTATION {status.upper()}: {report['completed_units']}/{report['total_units']} units; "
              f"{len(failures)} deferred stories. Report: {output / 'progress.json'}", flush=True)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "stopped"
        progress("stopped", error=str(error))
        save_json(output / "failures.json", {"run_hash": object_hash(identity), "updated_utc": utcnow(),
                                            "unresolved": list(failures.values()), "stop_reason": str(error)})
        raise
