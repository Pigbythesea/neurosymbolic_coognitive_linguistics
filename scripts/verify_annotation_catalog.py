"""Replay accepted REAL annotations through the catalog compiler, without inference.

No generated or replayed graph is written to the annotation dataset. This checks
representation preservation, input provenance, and schema/reference integrity; it
does not measure how well a model follows the new two-stage instructions.
"""

import argparse
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import implementation_identity, initialize_run, make_prompt, utcnow, verified_story
from neurosym.annotation_catalog import Catalog, binding_schema, grounding_schema, PROTOCOL, BINDING_TRANSPORT
from neurosym.annotation_catalog_jobs import CatalogStoryJob
from neurosym.graphs import GraphDelta, GraphHistory, graph_schema
from neurosym.io import immutable_json, object_hash, read_json, save_json


def check_schema(value, schema, definitions):
    """Check the emitted schema subset on real replay responses (no dependency)."""
    if "$ref" in schema:
        return check_schema(value, definitions[schema["$ref"].split("/")[-1]], definitions)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                check_schema(value, option, definitions)
                return
            except AssertionError:
                pass
        raise AssertionError("No schema variant accepts: " + repr(value)[:200])
    kind = schema.get("type")
    if kind == "object":
        assert isinstance(value, dict) and schema["additionalProperties"] is False
        assert set(value) == set(schema["required"]) == set(schema["properties"])
        for key, item in value.items():
            check_schema(item, schema["properties"][key], definitions)
    elif kind == "array":
        assert isinstance(value, list)
        assert schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf"))
        for item in value:
            check_schema(item, schema["items"], definitions)
    elif kind == "string":
        assert isinstance(value, str) and len(value) >= schema.get("minLength", 0)
        if "pattern" in schema:
            assert re.fullmatch(schema["pattern"], value)
    elif kind in ("integer", "number"):
        assert type(value) in ((int,) if kind == "integer" else (int, float))
        assert schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf"))
    elif kind == "null":
        assert value is None
    elif kind == "boolean":
        assert type(value) is bool
    else:
        raise AssertionError("Unchecked schema keyword/type: " + repr(schema))
    if "enum" in schema:
        assert value in schema["enum"]


def replay(graph, history, unit):
    groups = []
    new = {v.id: v for v in graph.entities}
    for mention in graph.mentions:
        if mention.entity_id not in groups:
            groups.append(mention.entity_id)
    original_mentions, referents = [], []
    for entity in groups:
        mentions = [m for m in graph.mentions if m.entity_id == entity]
        if entity in new:
            mentions.sort(key=lambda m: m.span != new[entity].first_mention)
        original_mentions.extend(mentions)
        referents.append({"identity": ({"type": "new", "entity": new[entity].model_dump(
            exclude={"id", "first_mention"})} if entity in new else
            {"type": "existing", "entity": list(history.entities).index(entity)}),
            "mentions": [m.model_dump(include={"span", "form", "confidence"}) for m in mentions]})
    draft = {"referents": referents,
             "events": [e.model_dump(exclude={"id", "arguments", "scope_parent_id"}) for e in graph.events],
             "uncertainties": [u.model_dump() for u in graph.uncertainties]}
    ground_schema = grounding_schema(history, unit)
    check_schema(draft, ground_schema, ground_schema["$defs"])
    catalog = Catalog(draft, history, unit)
    for mention in catalog.mentions:
        assert all(catalog.mention_map[key].entity_id == mention.entity_id
                   for key in catalog.anchors[mention.id])
    assert not ({v.id for v in catalog.entities} & set(history.entities))
    assert all(any(m.entity_id == e.id and m.span == e.first_mention for m in catalog.mentions)
               for e in catalog.entities)
    renamed = {old.id: current.id for old, current in zip(original_mentions, catalog.mentions)}
    renamed.update({old.entity_id: current.entity_id for old, current in zip(original_mentions, catalog.mentions)})
    renamed.update({old.id: current.id for old, current in zip(graph.events, catalog.events)})
    renamed.update({old.id: f"{unit['id']}_r{i + 1}" for i, old in enumerate(graph.relations)})
    rename = lambda key: renamed.get(key, key)
    links = {"references": {}, "events": {}, "relations": [], "entity_updates": [], "uncertainties": []}
    for mention in graph.mentions:
        key = rename(mention.id)
        links["references"][key] = {"status": mention.reference_status,
            "anchor": catalog.anchors[key].index(rename(mention.antecedent_id)) if mention.antecedent_id else None}
    for event in graph.events:
        key = rename(event.id)
        arguments = []
        for arg in event.arguments:
            target = {"kind": "mention" if arg.mention_id else arg.target_kind,
                      "id": rename(arg.mention_id or arg.target_id)}
            arguments.append({"role": arg.role, "frame_role": arg.frame_role,
                              "target": catalog.targets.index(target)})
        links["events"][key] = {"arguments": arguments,
            "scope_parent": catalog.parents[key].index(rename(event.scope_parent_id)) if event.scope_parent_id else None}
    for relation in graph.relations:
        value = relation.model_dump(exclude={"id"})
        for key in ("source_event", "target_event"):
            value[key] = list(catalog.event_map).index(rename(value[key]))
        links["relations"].append(value)
    for update in graph.entity_updates:
        value = update.model_dump(exclude={"entity_id"})
        value["entity"] = list(catalog.entity_map).index(rename(update.entity_id))
        links["entity_updates"].append(value)
    response = {"result": {"links": links}}
    schema = binding_schema(catalog)
    check_schema(response, schema, schema["$defs"])
    compiled = catalog.compile(links)

    def translate(value):
        if isinstance(value, str):
            return rename(value)
        if isinstance(value, dict):
            return {k: translate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [translate(v) for v in value]
        return value

    expected, actual = translate(graph.model_dump()), compiled.model_dump()
    for field in ("entities", "mentions", "events", "relations"):
        expected[field].sort(key=lambda v: v["id"])
        actual[field].sort(key=lambda v: v["id"])
    assert actual == expected, "Semantic roundtrip changed: " + unit["id"]
    return draft, catalog, response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-revision", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/annotation-legacy-luna-v1.json")
    args = parser.parse_args()
    config = read_json(args.config)
    output, corpus = ROOT / config["output"], ROOT / config["corpus"]
    if (output / "RUNNING.lock").exists():
        raise RuntimeError("Stop annotation before checking/installing the protocol revision.")
    base, index = read_json(output / "run.json"), read_json(corpus / "index.json")
    instructions = (ROOT / config["prompt"]).read_text(encoding="utf-8")
    count, hashes = 0, {}

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Cache verification must never call a model.")

    job = CatalogStoryJob(output=output, corpus=corpus, config=config, index=index, identity=base,
        implementation_hash="offline-verification", instructions=instructions,
        codex=None, cli=None, workspace=None, graph_schema_path=None, repair_schema_path=None,
        policy={}, progress=unexpected_request)
    job.request = unexpected_request
    for pass_name in config["passes"]:
        for entry in index["stories"]:
            history, resumed_history = GraphHistory(), GraphHistory()
            story = verified_story(corpus, entry)
            for unit in story["units"]:
                path = output / pass_name / entry["id"] / (unit["id"] + ".json")
                if not path.is_file():
                    break
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                saved = read_json(path)
                prompt = make_prompt(instructions, story, unit, history)
                expected = {"run_hash": object_hash(base), "pass": pass_name, "story_hash": entry["content_hash"],
                            "unit": unit, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
                assert saved["input_hash"] == object_hash(expected)
                assert saved["graph_hash"] == object_hash(saved["graph"])
                graph = GraphDelta.model_validate(saved["graph"])
                assert job.unit(story, unit, resumed_history, pass_name) == "CACHED"
                draft, catalog, response = replay(graph, history, unit)
                history.append(graph, unit)
                count += 1
    assert count, "No real saved annotations available for verification."
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    for path in (ROOT / "neurosym").glob("annotation*.py"):
        ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
    identity = implementation_identity(ROOT, config, index, instructions, graph_schema())
    report = {"protocol": PROTOCOL, "binding_transport": BINDING_TRANSPORT,
              "recorded_utc": utcnow(), "real_graph_roundtrips": count,
              "saved_units_byte_unchanged": len(hashes), "saved_file_sha256": hashes,
              "input_provenance_revalidated": True, "schema_and_compiler_checks": "passed",
              "resumed_saved_units_without_requests": count,
              "model_inference_run": False, "live_protocol_validated": False,
              "implementation_hash": object_hash(identity)}
    save_json(ROOT / "artifacts/annotation-catalog-validation.json", report)
    if args.record_revision:
        receipt = {"base_run_hash": object_hash(base), "identity": identity,
                   "recorded_utc": utcnow(), "reason":
                       "Use shared mention/event catalogs, exclude self by schema, check prompt size locally, "
                       "and reject clear source-word/pronoun mismatches in NEW groundings. Preserve all accepted "
                       "legacy units, their original hashes, graph schema, corpus, and semantic task instructions. "
                       "New records carry protocol/stage/schema provenance; audit reports protocol counts. "
                       "Compiler compatibility does not validate semantic labels; the separate span audit flags "
                       "saved units for review without rewriting them.",
                   "validation": report}
        immutable_json(output / "runner-revisions" / (object_hash(identity) + ".json"), receipt)
        assert initialize_run(output, identity) == base
    print(f"PASS: {count} real graphs roundtrip without semantic changes; {len(hashes)} saved files unchanged.")
    print("No model calls made. Live two-stage reliability has not yet been measured.")


if __name__ == "__main__":
    main()
