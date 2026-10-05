"""Check exact grounding and rejection gates using the real corpus and captured labels.

No inference and no generated annotations. Replayed old labels are diagnostic,
never adopted by the new run. The semantic rejection regression below is a
source-inspected assessment of the actual first saved unit, not an accuracy score.
"""

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import implementation_identity, initialize_run, verified_story, utcnow
from neurosym.annotation_anchored import (SourceText, prompt_base, ground_schema, compile_grounding,
    bind_schema, compile_bindings, review_schema, review_items, validate_review, semantic_review_prompt,
    grounding_prompt, binding_prompt)
from neurosym.annotation_anchored_jobs import AnchoredStoryJob
from neurosym.annotation_jobs import DeferredUnit
from neurosym.graphs import GraphDelta, GraphHistory, graph_schema
from neurosym.io import object_hash, read_json, save_json
from verify_annotation_catalog import replay, check_schema


def quoted_draft(draft, source):
    value = deepcopy(draft)
    for referent in value["referents"]:
        for mention in referent["mentions"]:
            mention["span"] = source.anchor_for_span(mention["span"], current=True)
    for event in value["events"]:
        for field in ("trigger", "evidence"):
            event[field] = source.anchor_for_span(event[field], current=field == "trigger")
    for item in value["uncertainties"]:
        item["span"] = source.anchor_for_span(item["span"], current=False)
    return value


def quoted_links(links, source):
    value = deepcopy(links)
    for field, span_field in (("relations", "evidence"), ("entity_updates", "evidence"), ("uncertainties", "span")):
        for item in value[field]:
            item[span_field] = source.anchor_for_span(item[span_field], current=False)
    return value


def first_unit_review(graph):
    """Source-inspected judgements for the actual archived opening sentence.

    The inhabit trigger points at 'we'; its evidence omits 'inhabit'. The universe
    is the location of inhabitation rather than an affected patient. These are
    rejection evidence, not an independent gold-standard annotation of the story.
    """
    prefix = "story_01_u0000_"
    evidence = {
        "coverage": ("we inhabit", "needs_correction", "The inhabit record is not anchored to its predicate."),
        "n1": ("some scientists", "supported", "Explicit group description."),
        "n2": ("the universe that we inhabit", "supported", "The inhabited universe is explicitly described."),
        "n3": ("we inhabit", "supported", "We denotes the group of inhabitants."),
        "n4": ("the only universe there is", "supported", "A predicative description occurs within the negative proposition."),
        "m1": ("some scientists", "supported", "The span quotes the introducing noun phrase."),
        "m2": ("the universe", "supported", "The span quotes the referring noun phrase."),
        "m3": ("we", "supported", "The span quotes a pronoun."),
        "m4": ("the only universe there is", "supported", "The span quotes the predicative description."),
        "e1": ("scientists who say", "supported", "The source explicitly presents the saying event."),
        "e2": ("is not the only universe there is", "supported", "The copular proposition is negative and reported."),
        "e3": ("we inhabit", "needs_correction", "The stored trigger [9,10) quotes we, not inhabit; evidence ends before the predicate."),
    }
    binding_evidence = {
        "m1": "some scientists", "m2": "the universe", "m3": "we", "m4": "the only universe there is",
        "e1": "scientists who say", "e2": "say that the universe", "e3": "the universe that we inhabit",
        "e1/argument/0": "scientists who say", "e1/argument/1": "say that the universe that we inhabit is not the only universe there is",
        "e2/argument/0": "the universe that we inhabit is not", "e2/argument/1": "is not the only universe there is",
        "e3/argument/0": "we inhabit", "e3/argument/1": "the universe that we inhabit",
    }
    result = {"grounding": {}, "bindings": {}}
    for short, (quote, verdict, reason) in evidence.items():
        result["grounding"][short if short == "coverage" else prefix + short] = {
            "verdict": verdict, "evidence": {"quote": quote, "occurrence": 1}, "reason": reason}
    for short, quote in binding_evidence.items():
        result["bindings"][prefix + short] = {"verdict": "supported",
            "evidence": {"quote": quote, "occurrence": 1}, "reason": "No additional defect identified in this source inspection."}
    result["bindings"][prefix + "e3/argument/1"].update(verdict="needs_correction",
        reason="The universe is the location of inhabitation, not an affected patient; review this role.")
    assert {k: set(v) for k, v in result.items()} == {k: set(v) for k, v in review_items(graph).items()}
    return result


def main():
    config = read_json(ROOT / "configs/annotation.json")
    legacy = read_json(ROOT / "configs/annotation-legacy-luna-v1.json")
    old_output = ROOT / legacy["output"]
    new_output = ROOT / config["output"]
    assert config["protocol"] == "source-anchored-v2"
    assert config["model"] == legacy["model"] and config["reasoning_effort"] == "none"
    assert old_output != new_output and not (old_output / "RUNNING.lock").exists()
    if new_output.exists() and any(new_output.iterdir()):
        raise ValueError("This pre-run check requires the new output to be empty; do not run it over a started v2 run.")
    corpus = ROOT / config["corpus"]
    index = read_json(corpus / "index.json")
    instructions = (ROOT / config["prompt"]).read_text(encoding="utf-8")
    counts, unchanged, sample = Counter(), {}, None
    max_prompt = Counter()
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        history = GraphHistory()
        for unit in story["units"]:
            source = SourceText(story, unit)
            for i in range(unit["start_token"], unit["end_token"]):
                span = {"start": i, "end": i + 1}
                for current in (True, False):
                    anchor = source.anchor_for_span(span, current=current)
                    assert source.resolve(anchor, current=current, path="real-word-replay", records=[]) == span
                counts["real_words_resolved"] += 1
            assert len(source.tokens) == unit["end_token"]
            counts["real_units_checked"] += 1
            path = old_output / "primary" / entry["id"] / (unit["id"] + ".json")
            if not path.is_file():
                continue
            unchanged[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = read_json(path)
            graph = GraphDelta.model_validate(saved["graph"])
            draft, old_catalog, response = replay(graph, history, unit)
            anchored = quoted_draft(draft, source)
            schema = ground_schema(history)
            check_schema(anchored, schema, schema["$defs"])
            try:
                catalog, anchors = compile_grounding(anchored, source, history)
            except ValueError as error:
                assert any(term in str(error) for term in (
                    "Grounding/source mismatch", "same mention span", "Duplicate predicate/trigger")), str(error)
                counts["archived_groundings_rejected_by_new_source_checks"] += 1
                history.append(graph, unit)
                continue
            links = quoted_links(response["result"]["links"], source)
            schema = bind_schema(catalog)
            check_schema({"result": {"links": links}}, schema, schema["$defs"])
            compiled, _ = compile_bindings(links, source, catalog)
            assert compiled.model_dump() == old_catalog.compile(response["result"]["links"]).model_dump()
            base = prompt_base(instructions, story, unit, history)
            assert source.input(history)["prefix_text"] == story["text"][:unit["prefix_char_end"]]
            for phase, prompt in (("ground", grounding_prompt(base, history)),
                                  ("bind", binding_prompt(base, catalog, source)),
                                  ("review", semantic_review_prompt(base, source, compiled))):
                max_prompt[phase] = max(max_prompt[phase], len(prompt))
                assert len(prompt) < 1_048_576
            if sample is None:
                sample = (story, unit, anchored, links, compiled, source)
            history.append(graph, unit)
            counts["archived_graphs_preserved_modulo_allocated_ids"] += 1
    assert counts["real_units_checked"] == 1217 and counts["real_words_resolved"] == 23655
    assert sample is not None
    story, unit, draft, links, graph, source = sample
    assert unit["id"] == "story_01_u0000"
    review = first_unit_review(graph)
    schema = review_schema(graph)
    check_schema(review, schema, schema["$defs"])
    failures, review_evidence = validate_review(review, source, graph)
    assert {f["family"] for f in failures} == {"grounding", "bindings"}
    assert any(f["claim"].endswith("e3") for f in failures)

    # Replay source-inspected negative evidence through the real job boundary.
    # No successful review or replacement annotation is fabricated.
    replay_root = ROOT / "artifacts/source-anchored-regression"
    captured = {"ground": draft, "bind": {"result": {"links": links}}, "review": review}
    for stage, response in captured.items():
        save_json(replay_root / (stage + ".json"), response)
    save_json(replay_root / "provenance.json", {"source": "Actual archived story_01_u0000 graph projected to exact source quotes",
        "review_origin": "Source-inspected assistant regression assessment; not a model CLI response or independent gold",
        "accepted_new_annotations": 0, "model_calls": 0})

    class ReplayRejectedJob(AnchoredStoryJob):
        def stage(self, directory, stage, prompt, schema, unit_hash, **context):
            if stage == "ground" and hasattr(self, "ground_seen"):
                raise DeferredUnit("Captured evidence ends here; no replacement annotation or model call is permitted in this check.")
            if stage == "ground":
                self.ground_seen = True
            return captured[stage], {"protocol_context": {}, "cli": {"origin": "real-source replay"}}, replay_root / (stage + ".json"), {"replayed": True}

    fresh_history = GraphHistory()
    job = ReplayRejectedJob(output=replay_root, corpus=corpus, config=config, index=index, identity={},
        implementation_hash="offline-regression", instructions=instructions, codex=None, cli=None,
        workspace=None, graph_schema_path=None, repair_schema_path=None, policy={}, progress=lambda *a, **k: None)
    try:
        job.unit(story, unit, fresh_history, "primary")
    except DeferredUnit:
        pass
    else:
        raise AssertionError("Rejected semantic assessment unexpectedly produced an accepted unit.")
    assert not fresh_history.entities and not fresh_history.events
    assert not (replay_root / "primary").exists()
    identity = implementation_identity(ROOT, config, index, instructions, graph_schema())
    try:
        initialize_run(old_output, identity)
    except ValueError as error:
        assert "cannot be resumed" in str(error)
    else:
        raise AssertionError("The old annotation history was adopted by the new protocol.")
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in unchanged.items())
    report = {"recorded_utc": utcnow(), "protocol": config["protocol"], "counts": dict(counts),
        "largest_replayed_prompt_characters": dict(max_prompt), "legacy_files_byte_unchanged": len(unchanged),
        "legacy_file_sha256": unchanged, "rejected_review_prevents_commit_and_history_update": True,
        "old_run_cannot_be_adopted": True, "model_calls": 0, "live_protocol_measured": False,
        "new_output": config["output"], "implementation_hash": object_hash(identity)}
    save_json(ROOT / "artifacts/source-anchored-validation.json", report)
    print({k: v for k, v in report.items() if k != "legacy_file_sha256"})


if __name__ == "__main__":
    main()
