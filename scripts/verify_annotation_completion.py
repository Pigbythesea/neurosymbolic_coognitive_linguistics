"""Regression checks using the actual v4 drafts and their recorded failures.

Hand-authored corrections below exercise the repair interface on real text. They
are never written to the annotation corpus and are not model-accuracy evidence.
No model calls or fabricated CLI responses are used.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import implementation_identity
from neurosym.annotation_completion import correction_request, apply_corrections, PROTOCOL
from neurosym.annotation_reuse import find_original_draft, import_original_draft
from neurosym.annotation_source import (CompilationIssues, SourceResolver, compile_annotation,
                                        prompt_base, get_path, set_path)
from neurosym.annotation_source_jobs import SourceStoryJob, decode_request, load_saved_candidate
from neurosym.graphs import GraphHistory, GraphDelta, graph_schema
from neurosym.io import read_json, object_hash, save_json


def empty_additions(candidate):
    return {field: [] for field in candidate}


def completion(candidate, tasks):
    return {task["field"]: (empty_additions(candidate) if task["mode"] == "additions" else
                            deepcopy(get_path(candidate, task["path"]))) for task in tasks}


def target_field(tasks, path):
    return next(task["field"] for task in tasks if task["path"] == path)


def main():
    archive = ROOT / "data/annotations/deniz-luna-v4"
    if (archive / "RUNNING.lock").exists():
        raise ValueError("Stop v4 before verifying immutable archived records.")
    original_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob("*") if p.is_file()}
    config = read_json(ROOT / "configs/annotation.json")
    index = read_json(ROOT / config["corpus"] / "index.json")
    instructions = (ROOT / config["prompt"]).read_text(encoding="utf-8")
    counts = Counter()
    for entry in index["stories"]:
        story = read_json(ROOT / config["corpus"] / entry["path"])
        history = GraphHistory()
        for unit in story["units"]:
            directory = archive / "attempts/primary" / unit["id"] / "source"
            raw_path = directory / "request-0000.response.json"
            if not raw_path.is_file():
                break
            candidate = read_json(raw_path)
            base = prompt_base(instructions, story, unit, history)
            found = find_original_draft(archive, base=base, config=config, corpus_hash=index["content_hash"],
                                        story=story, unit=unit, pass_name="primary")
            assert found is not None
            counts["exact_input_original_drafts_reusable"] += 1
            # A different input must never inherit an old draft.
            assert find_original_draft(archive, base=base + " ", config=config,
                corpus_hash=index["content_hash"], story=story, unit=unit, pass_name="primary") is None
            counts["changed_inputs_rejected"] += 1
            try:
                compile_annotation(candidate, story, unit, history)
            except CompilationIssues as error:
                issues = error.issues
            else:
                issues = []
            uid = unit["id"]
            if uid in ("story_01_u0005", "story_01_u0006", "story_01_u0008", "story_01_u0014",
                       "story_02_u0001", "story_02_u0003"):
                assert issues
                _, schema, tasks = correction_request(base, candidate, issues)
                assert schema["additionalProperties"] is False
                assert set(schema["required"]) == set(schema["properties"])
                if uid == "story_01_u0005":
                    assert all(t["mode"] == "location" for t in tasks)
                    # Actual recorded model correction: s65 heads, s71 tails.
                    response = read_json(directory / "request-0001.response.json")
                    fixed = apply_corrections(candidate, response, tasks)
                    graph, _ = compile_annotation(fixed, story, unit, history)
                    saved = read_json(archive / "primary/story_01" / (uid + ".json"))
                    assert graph.model_dump() == saved["graph"]
                    counts["actual_location_correction_unchanged"] += 1
                else:
                    assert tasks[-1]["mode"] == "additions"
                    response = completion(candidate, tasks)
                    if uid in ("story_01_u0008", "story_02_u0003"):
                        value = "all the time" if uid == "story_01_u0008" else "avatar"
                        literal = {"key": "new_value", "kind": "time" if uid == "story_01_u0008" else "text",
                                   "value": value, "unit": None, "evidence": value}
                        response["additions"]["literals"].append(literal)
                        response[target_field(tasks, ["events", 0])]["arguments"][1]["target"] = "new_value"
                        fixed = apply_corrections(candidate, response, tasks)
                        graph, _ = compile_annotation(fixed, story, unit, history)
                        argument = graph.events[0].arguments[1]
                        assert argument.target_kind == "literal"
                        assert next(v.value for v in graph.literals if v.id == argument.target_id) == value
                        counts["missing_value_completed_without_unrelated_retargeting"] += 1
                    elif uid == "story_01_u0006":
                        response[target_field(tasks, ["relations", 1])] = None
                        fixed = apply_corrections(candidate, response, tasks)
                        graph, _ = compile_annotation(fixed, story, unit, history)
                        assert len(graph.relations) == 1
                        counts["unsupported_record_deletion"] += 1
                    elif uid == "story_01_u0014":
                        ref = response[target_field(tasks, ["referents", 0])]
                        ref["mentions"] = [m for m in ref["mentions"] if m["key"] != "m2"]
                        source = SourceResolver(story, unit)
                        locations = source.matches("I", current=True)
                        for key, span in zip(("m0", "m1"), locations, strict=True):
                            next(m for m in ref["mentions"] if m["key"] == key)["span"] = {"quote": "I", "source_id": f"s{span.start}"}
                        fixed = apply_corrections(candidate, response, tasks)
                        graph, _ = compile_annotation(fixed, story, unit, history)
                        assert len(graph.mentions) == sum(len(r["mentions"]) for r in candidate["referents"]) - 1
                        counts["unsupported_mention_removed_without_substitute"] += 1
                    else:
                        # Reuse only the real location choices, not v4's forced c0 substitutions.
                        meta = read_json(directory / "request-0001.json")
                        answers = read_json(directory / "request-0001.response.json")
                        for old in meta["protocol_context"]["correction_tasks"]:
                            if old.get("choices"):
                                root_path = old["path"][:2]
                                record = response[target_field(tasks, root_path)]
                                set_path(record, old["path"][2:], {"quote": old["quote"], "source_id": answers[old["field"]]})
                        response["additions"]["contexts"].append({"key": "c1", "kind": "desired", "holder": "n0",
                            "attribution_event": "e2", "parents": [], "evidence": "he wanted JT's input"})
                        fixed = apply_corrections(candidate, response, tasks)
                        graph, _ = compile_annotation(fixed, story, unit, history)
                        assert graph.events[3].context_ids == [graph.contexts[1].id]
                        assert graph.events[4].context_ids == [graph.contexts[1].id]
                        counts["missing_context_added_with_links_preserved"] += 1
                    # Unrequested keys are rejected rather than copied into a graph.
                    try:
                        apply_corrections(candidate, {**response, "unrequested": []}, tasks)
                    except ValueError:
                        counts["unrequested_edits_rejected"] += 1
                    else:
                        raise AssertionError("Unrequested edit accepted")
                    edited = {tuple(t["path"]) for t in tasks if t["mode"] == "record"}
                    for field, records in candidate.items():
                        for i, record in enumerate(records):
                            if (field, i) not in edited:
                                assert record in fixed[field]
                    counts["unaffected_records_preserved"] += 1
            # Check import and decoding with an actual original CLI trace; no new CLI invocation.
            if uid == "story_01_u0000":
                implementation = implementation_identity(ROOT, config, index, instructions, graph_schema())
                output = ROOT / "artifacts/annotation-completion-verification/imports" / object_hash(implementation)
                dest = output / "attempts/primary" / uid / "source"
                dest.mkdir(parents=True, exist_ok=True)
                identity = {"run_hash": object_hash(implementation), "pass": "primary", "story_hash": story["content_hash"],
                            "unit": unit, "prompt_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest()}
                origin = read_json(directory / "request-0000.json")
                job = SourceStoryJob(output=output, corpus=ROOT / config["corpus"], config=config, index=index,
                    identity=implementation, implementation_hash=object_hash(implementation), instructions=instructions,
                    codex=None, cli=origin["cli"], workspace=ROOT / "artifacts/annotation-context",
                    graph_schema_path=ROOT / config["schema"], repair_schema_path=ROOT / "schemas/graph-repair.schema.json",
                    policy=read_json(ROOT / "configs/annotation-runtime.json"), progress=lambda *args, **kwargs: None)
                import_original_draft(job, dest, base, identity, story, unit, "primary", PROTOCOL)
                assert decode_request(output, dest / "request-0000.json") == candidate
                counts["actual_original_trace_import_replayed"] += 1
            saved_path = archive / "primary" / entry["id"] / (uid + ".json")
            if not saved_path.is_file():
                break
            saved = read_json(saved_path)
            loaded = load_saved_candidate(archive, saved)
            graph, _ = compile_annotation(loaded, story, unit, history)
            assert object_hash(graph.model_dump()) == saved["graph_hash"]
            history.append(GraphDelta.model_validate(saved["graph"]), unit)
            counts["archived_v4_graphs_replayed"] += 1
    assert counts["missing_value_completed_without_unrelated_retargeting"] == 2
    assert counts["missing_context_added_with_links_preserved"] == 1
    assert counts["unsupported_mention_removed_without_substitute"] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in original_hashes.items())
    report = {"status": "passed", "counts": dict(counts), "v4_archive_files_unchanged": len(original_hashes),
        "model_calls": 0, "scope": "Actual v4 failures and source text; authored corrections check transport only. "
        "No fresh model accuracy is claimed and no correction fixture enters production annotations."}
    save_json(ROOT / "artifacts/annotation-completion-verification/report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
