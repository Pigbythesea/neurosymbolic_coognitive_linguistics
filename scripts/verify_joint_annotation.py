"""Offline verification on real corpus words and archived annotations; no model calls.

Archived labels are transport/compatibility evidence, not a reference for semantic
accuracy. Projections exist only in memory and never enter the annotation dataset.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import verified_story
from neurosym.annotation_anchored import SourceText
from neurosym.annotation_joint import compile_annotation, prior_catalog, prompt_base, response_schema
from neurosym.graphs import GraphDelta, GraphHistory
from neurosym.io import object_hash, read_json, save_json
from neurosym.semantic_records import compile_unit
from neurosym.semantic_queries import compile_queries, evaluate_query
from neurosym.temporal import checked_alignment


def project(graph, history, source):
    """Re-express observed archive fields in the new request transport only."""
    _, catalog = prior_catalog(history)
    names = {identity: alias for alias, (_, identity) in catalog.items()}
    for i, entity in enumerate(graph.entities):
        names[entity.id] = f"n{i}"
    for i, mention in enumerate(graph.mentions):
        names[mention.id] = f"m{i}"
    for i, event in enumerate(graph.events):
        names[event.id] = f"e{i}"
    refs = {}
    new = {e.id: e for e in graph.entities}
    for mention in graph.mentions:
        entity = new.get(mention.entity_id)
        # Local referent key is independent of its prior identity alias.
        key = names[entity.id] if entity else "n_old_" + names[mention.entity_id]
        ref = refs.setdefault(mention.entity_id, {"key": key,
            "identity": {k: getattr(entity, k) for k in ("label", "concept", "kind")} if entity else names[mention.entity_id],
            "mentions": []})
        ref["mentions"].append({"key": names[mention.id],
            "span": source.anchor_for_span(mention.span.model_dump(), current=True), "form": mention.form,
            "reference_status": "explicit" if mention.reference_status == "first_mention" else mention.reference_status})
    contexts, events = [], []
    for event in graph.events:
        ids = []
        if event.status not in (None, "asserted"):
            key = "c_" + names[event.id]
            contexts.append({"key": key, "kind": event.status, "holder": None,
                "attribution_event": names.get(event.scope_parent_id), "parents": [],
                "evidence": source.anchor_for_span(event.evidence.model_dump(), current=False)})
            ids.append(key)
        elif event.scope_parent_id:
            # Do not invent a context kind that the old format did not specify.
            raise ValueError("legacy_untyped_scope")
        events.append({"key": names[event.id], "predicate": event.predicate, "sense_id": event.sense_id,
            "trigger": source.anchor_for_span(event.trigger.model_dump(), current=True),
            "evidence": source.anchor_for_span(event.evidence.model_dump(), current=False),
            "arguments": [{"role": a.role, "frame_role": a.frame_role,
                           "target": names[a.mention_id or a.target_id]} for a in event.arguments],
            "polarity": event.polarity, "tense": event.tense, "contexts": ids})
    return {"referents": list(refs.values()), "literals": [], "contexts": contexts, "events": events,
        "relations": [{"type": r.type, "source": names[r.source_event], "target": names[r.target_event],
                       "evidence": source.anchor_for_span(r.evidence.model_dump(), current=False),
                       "status": r.status, "contexts": []} for r in graph.relations],
        "identity_links": [], "entity_updates": [{"entity": names[u.entity_id], "attribute": u.attribute,
            "value": u.value, "operation": u.operation, "evidence": source.anchor_for_span(u.evidence.model_dump(), current=False),
            "status": u.status, "contexts": []} for u in graph.entity_updates],
        "uncertainties": [{"span": source.anchor_for_span(u.span.model_dump(), current=False),
                           "category": u.category, "explanation": u.explanation} for u in graph.uncertainties]}


def main():
    config = read_json(ROOT / "configs/annotation.json")
    corpus = ROOT / config["corpus"]
    index = read_json(corpus / "index.json")
    instructions = (ROOT / "prompts/graph_annotation_joint.txt").read_text(encoding="utf-8")
    counts, skips, largest = Counter(), Counter(), {"characters": 0}
    schema = response_schema()
    for definition in [schema, *schema["$defs"].values()]:
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            assert set(definition["properties"]) == set(definition["required"])
    archive = ROOT / "data/annotations/deniz-luna-v1/primary"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.glob("*/*.json")}
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        alignment = checked_alignment(ROOT / "data/processed/alignment", entry["id"], index["content_hash"])
        history = GraphHistory()
        for i, unit in enumerate(story["units"]):
            source = SourceText(story, unit)
            for wi in range(unit["start_token"], unit["end_token"]):
                anchor = source.anchor_for_span({"start": wi, "end": wi + 1}, current=True)
                assert source.resolve(anchor, current=True, path="word", records=[]) == {"start": wi, "end": wi + 1}
                counts["real_word_roundtrips"] += 1
            counts["corpus_units"] += 1
            path = archive / entry["id"] / (unit["id"] + ".json")
            if not path.exists():
                continue
            saved = read_json(path)
            graph = GraphDelta.model_validate(saved["graph"])
            size = len(prompt_base(instructions, story, unit, history)) + len(json.dumps(schema))
            if size > largest["characters"]:
                largest = {"characters": size, "unit_id": unit["id"]}
            try:
                projected = project(graph, history, source)
            except ValueError as error:
                if str(error) != "legacy_untyped_scope":
                    raise
                skips[str(error)] += 1
            else:
                compiled, anchors = compile_annotation(projected, story, unit, history)
                assert [e.predicate for e in compiled.events] == [e.predicate for e in graph.events]
                assert sum(len(e.arguments) for e in compiled.events) == sum(len(e.arguments) for e in graph.events)
                assert len(compiled.mentions) == len(graph.mentions)
                assert all(a["span"]["end"] <= unit["end_token"] for a in anchors)
                counts["archived_joint_compilations"] += 1
                counts["compiled_contexts"] += len(compiled.contexts)
            history.append(graph, unit)
            counts["archived_prefix_graphs"] += 1
            # Actual archived semantic queries exercise downstream contracts.
            # No scientific readout is fitted by this verification.
            if i < 4:
                record, features, definitions, occurrences, reviews = compile_unit(story, unit, graph, history, saved, i, alignment)
                queries, answers, private, _ = compile_queries(story, unit, graph, history, record, 1729)
                for q, a, p in zip(queries, answers, private, strict=True):
                    result = evaluate_query(q["input"]["ast"], history, graph, story, p)
                    assert a["acceptable_indices"] == (sorted(p["candidate_ids"].index(v) for v in result) if result else [])
                    counts["query:" + q["input"]["ast"]["op"]] += 1
    assert largest["characters"] < 1_000_000
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    modules = [*ROOT.glob("neurosym/*.py"), *ROOT.glob("scripts/*.py")]
    for p in modules:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p), feature_version=(3, 11))
    report = {"status": "passed", "counts": dict(counts), "projection_exclusions": dict(skips),
        "largest_archived_prefix_request": largest, "constant_response_schema_characters": len(json.dumps(schema)),
        "archive_files_unchanged": len(before), "model_calls": 0, "scientific_fits": 0,
        "new_annotation_accuracy_measured": False,
        "limits": "Legacy v3 archive replay verifies transport and compatibility, not annotation accuracy. Use verify_source_annotation.py for the current v4 protocol.",
        "code_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}}
    save_json(ROOT / "artifacts/joint-annotation-verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "code_hashes"}, indent=2))


if __name__ == "__main__":
    main()
