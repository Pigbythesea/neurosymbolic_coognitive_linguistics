"""Verify v4 against actual corpus text, saved failures and accepted v3 responses.

No model calls, mock CLI runs, invented stimuli or scientific fits. The archive
is compatibility evidence, not semantic gold. Prompt demonstrations are authored
examples, not fresh model output or an independent accuracy evaluation.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import verified_story, implementation_identity
from neurosym.annotation_joint import prior_catalog, JointAnnotation
from neurosym.annotation_source import (SourceResolver, Annotation, compile_annotation, prompt_base,
    response_schema, CompilationIssues, correction_request, apply_corrections)
from neurosym.graphs import GraphDelta, GraphHistory, graph_schema
from neurosym.io import read_json, save_json, object_hash
from neurosym.semantic_records import compile_unit
from neurosym.semantic_queries import compile_queries
from neurosym.temporal import checked_alignment


def project(raw, history):
    """Lossless interface projection; valid old occurrence choices stay explicit."""
    value = deepcopy(raw)
    _, lookup = prior_catalog(history)
    for ref in value["referents"]:
        old = ref.pop("identity")
        if isinstance(old, str):
            entity = history.entities[lookup[old][1]]
            ref.update(description={k: getattr(entity, k) for k in ("label", "concept", "kind")}, existing=old)
        else:
            ref.update(description=old, existing=None)
    return value


def anchor_paths(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict) and set(v) == {"quote", "occurrence"}:
                yield [k], v
            else:
                for path, q in anchor_paths(v):
                    yield [k, *path], q
    elif isinstance(value, list):
        for i, v in enumerate(value):
            for path, q in anchor_paths(v):
                yield [i, *path], q


def main():
    config = read_json(ROOT / "configs/annotation.json")
    corpus = ROOT / config["corpus"]
    index = read_json(corpus / "index.json")
    instructions = (ROOT / config["prompt"]).read_text(encoding="utf-8")
    archive = ROOT / "data/annotations/deniz-luna-v3"
    paths = list((archive / "attempts").glob("primary/*/joint/*.response.json"))
    paths += list((archive / "primary").glob("*/*.json"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    counts, failures = Counter(), []
    schema = response_schema()
    for definition in [schema, *schema["$defs"].values()]:
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(definition["properties"])
    examples = [json.loads(s) for s in re.findall(r"EXAMPLE_JSON\n(.*?)\nEND_EXAMPLE", instructions, re.S)]
    largest = {"characters": 0}
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        history = GraphHistory()
        for example in [e for e in examples if e["story_id"] == entry["id"]]:
            unit = story["units"][0]
            source = SourceResolver(story, unit)
            assert source.normalized(example["current_passage"]) == " ".join(source.tokens)
            graph, provenance = compile_annotation(example["output"], story, unit, GraphHistory())
            if entry["id"] == "story_02":
                assert graph.entity_updates[0].value == "Sarah"
                counts["typed_literal_values_resolved"] += 1
            assert all(a["span"]["end"] <= unit["end_token"] for a in provenance["source_anchors"])
            counts["real_source_demonstrations"] += 1
        alignment = checked_alignment(ROOT / "data/processed/alignment", entry["id"], index["content_hash"])
        for ui, unit in enumerate(story["units"]):
            source = SourceResolver(story, unit)
            for wi in range(unit["start_token"], unit["end_token"]):
                word = source.words[wi]["text"]
                located, issue = source.locate(word, ["word"], current=True)
                if issue:
                    assert any(c["source_id"] == f"s{wi}" for c in issue["choices"])
                    located, issue = source.locate({"quote": word, "source_id": f"s{wi}"}, ["word"], current=True)
                    counts["word_locations_disambiguated"] += 1
                assert issue is None and located == {"start": wi, "end": wi + 1}
                counts["real_word_roundtrips"] += 1
            passage = story["text"][unit["char_start"]:unit["prefix_char_end"]]
            located, issue = source.locate(passage, ["passage"], current=True)
            assert issue is None and located == {"start": unit["start_token"], "end": unit["end_token"]}
            counts["real_passage_roundtrips"] += 1
            saved_path = archive / "primary" / entry["id"] / (unit["id"] + ".json")
            saved = read_json(saved_path) if saved_path.exists() else None
            for path in sorted((archive / "attempts/primary" / unit["id"] / "joint").glob("*.response.json")):
                try:
                    raw = JointAnnotation.model_validate(read_json(path)).model_dump()
                except ValueError:
                    counts["archived_malformed_responses"] += 1
                    continue
                draft = project(raw, history)
                from neurosym.annotation_source import set_path
                for p, old in list(anchor_paths(draft)):
                    current = p[-1] == "trigger" or (p[0] == "referents" and p[-1] == "span")
                    matches = source.matches(old["quote"], current=current)
                    # Reproduce an actual valid archived choice, even if ambiguous.
                    if 0 < old["occurrence"] <= len(matches):
                        q = {"quote": old["quote"], "source_id": f"s{matches[old['occurrence'] - 1].start}"}
                    else:
                        q = old["quote"]
                        if len(matches) == 1:
                            counts["invalid_counts_now_uniquely_resolved"] += 1
                    set_path(draft, p, q)
                try:
                    graph, provenance = compile_annotation(draft, story, unit, history)
                except CompilationIssues as error:
                    counts["responses_with_explicit_remaining_issues"] += 1
                    failures.append({"response": path.relative_to(ROOT).as_posix(), "issues": error.issues})
                    # A real failed response must produce a valid targeted schema.
                    _, correction_schema, tasks = correction_request(instructions, draft, error.issues)
                    assert set(correction_schema["required"]) == {t["field"] for t in tasks}
                    counts["real_failure_correction_schemas"] += 1
                else:
                    counts["archived_responses_compiling"] += 1
                    if saved and path.relative_to(archive).as_posix() == saved["raw_response"]:
                        assert graph.model_dump() == saved["graph"]
                        counts["accepted_graphs_unchanged"] += 1
                        # Use an actual repeated quote and its archived selected
                        # location to exercise field-only correction application.
                        for p, old in anchor_paths(raw):
                            current = p[-1] == "trigger" or (p[0] == "referents" and p[-1] == "span")
                            _, issue = source.locate(old["quote"], p, current=current)
                            if issue and issue["choices"] and old["occurrence"] <= len(source.matches(old["quote"], current=current)):
                                _, cs, tasks = correction_request(instructions, draft, [issue])
                                chosen = source.matches(old["quote"], current=current)[old["occurrence"] - 1]
                                patched = apply_corrections(draft, {tasks[0]["field"]: f"s{chosen.start}"}, tasks)
                                assert patched == Annotation.model_validate(draft).model_dump()
                                try:
                                    apply_corrections(draft, {"unexpected_field": "rewrite"}, tasks)
                                except ValueError:
                                    counts["unrequested_corrections_rejected"] += 1
                                else:
                                    raise AssertionError("Unrequested field accepted")
                                counts["archived_source_choices_replayed"] += 1
                                break
            if saved:
                graph = GraphDelta.model_validate(saved["graph"])
                size = len(prompt_base(instructions, story, unit, history)) + len(json.dumps(schema))
                if size > largest["characters"]:
                    largest = {"characters": size, "unit": unit["id"]}
                history.append(graph, unit)
                record, *_ = compile_unit(story, unit, graph, history, saved, ui, alignment)
                queries, *_ = compile_queries(story, unit, graph, history, record, 1729)
                counts["downstream_queries_from_actual_graphs"] += len(queries)
    assert counts["accepted_graphs_unchanged"] == len(list((archive / "primary").glob("*/*.json")))
    assert counts["invalid_counts_now_uniquely_resolved"] > 0
    assert counts["archived_source_choices_replayed"] > 0
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    # Exercise the new reference index on long, actually observed histories.
    # These archived graphs never become current-run annotation inputs.
    legacy_archive = ROOT / "data/annotations/deniz-luna-v1/primary"
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        history = GraphHistory()
        available = []
        for unit in story["units"]:
            path = legacy_archive / entry["id"] / (unit["id"] + ".json")
            if not path.is_file():
                break
            available.append((unit, path))
        for position, (unit, path) in enumerate(available):
            # Render the largest observed prefix before appending its annotation.
            graph = GraphDelta.model_validate(read_json(path)["graph"])
            if position == len(available) - 1:
                size = len(prompt_base(instructions, story, unit, history)) + len(json.dumps(schema))
                if size > largest["characters"]:
                    largest = {"characters": size, "unit": unit["id"], "history": "archived_v1"}
            history.append(graph, unit)
            counts["long_history_records_checked"] += 1
    assert largest["characters"] < 1_000_000
    for folder in ("neurosym", "scripts"):
        for path in (ROOT / folder).glob("*.py"):
            ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
    report = {"status": "passed", "counts": dict(counts), "largest_saved_prefix_request": largest,
        "remaining_actual_failures": failures, "archive_files_unchanged": len(before),
        "model_calls": 0, "fresh_annotation_accuracy_measured": False,
        "limitations": "Actual archived labels and authored demonstrations verify transport only. Fresh v4 CLI generation and correction replay require real-run coverage.",
        "implementation": implementation_identity(ROOT, config, index, instructions, graph_schema())}
    save_json(ROOT / "artifacts/source-annotation-verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("remaining_actual_failures", "implementation")}, indent=2))


if __name__ == "__main__":
    main()
