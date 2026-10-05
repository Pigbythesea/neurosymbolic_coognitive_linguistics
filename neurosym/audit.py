"""Validate actual annotations and report agreement without treating either run as gold."""

from collections import Counter
from pathlib import Path

from .annotation import annotation_passes, verified_story
from .graphs import GraphDelta, GraphHistory
from .io import object_hash, read_json, save_json
from .semantic_records import context_profile


def span_key(span) -> tuple:
    return span.start, span.end


def semantic_atoms(graph: GraphDelta, history: GraphHistory) -> dict[str, set]:
    def entity(identity):
        item = history.entities[identity]
        return item.kind, item.concept, span_key(item.first_mention)

    def event(identity):
        item = history.events[identity]
        return item.predicate, span_key(item.trigger), item.polarity, item.status

    def argument(item):
        if item.target_kind == "literal":
            literal = history.literals[item.target_id]
            target = (literal.kind, literal.value, literal.unit)
        else:
            target = entity(item.target_id) if item.target_kind == "entity" else event(item.target_id)
        return item.role, item.target_kind, target

    return {
        "concepts": {(span_key(item.span), *entity(item.entity_id)[:2]) for item in graph.mentions},
        "events": {event(item.id) for item in graph.events},
        "roles": {(event(item.id), argument(arg)) for item in graph.events for arg in item.arguments},
        "reference": {(span_key(item.span), span_key(history.entities[item.entity_id].first_mention), item.reference_status)
                      for item in graph.mentions},
        "discourse": {(item.type, event(item.source_event), event(item.target_event), item.status) for item in graph.relations},
        "scope": {(event(item.id), tuple(history.event_statuses(item)),
                   object_hash(context_profile(item, history)))
                  for item in graph.events},
        "identity": {(entity(link.left_entity), entity(link.right_entity), link.relation, link.status,
                      tuple(sorted(c.kind for c in history.context_chain(link.context_ids)))) for link in graph.identity_links},
        "literals": {(v.kind, v.value, v.unit, span_key(v.evidence)) for v in graph.literals},
    }


def annotate_audit(root: Path, config_path: Path) -> dict:
    config = read_json(config_path)
    corpus = root / config["corpus"]
    output = root / config["output"]
    index = read_json(corpus / "index.json")
    run = read_json(output / "run.json")
    if run["corpus_hash"] != index["content_hash"] or run["configuration"] != config:
        raise ValueError("Annotation run does not match this corpus/configuration.")
    required = annotation_passes(config)
    passes = []
    for name in config["passes"]:
        receipt_path = output / f"complete.{name}.json"
        if not receipt_path.is_file():
            if name in required:
                raise ValueError("Required annotation pass is incomplete: " + name)
            continue
        receipt = read_json(receipt_path)
        if receipt["run_hash"] != object_hash(run) or receipt["units"] != sum(item["units"] for item in index["stories"]):
            raise ValueError("Annotation completion receipt does not match this run: " + name)
        passes.append(name)
    counts = {name: Counter() for name in passes}
    phenomena = {name: Counter() for name in passes}
    protocols = {name: Counter() for name in passes}
    semantic_checks = {name: Counter() for name in passes}
    pair = passes[:2]
    agreement = ({name: Counter() for name in ("concepts", "events", "roles", "reference", "discourse", "scope", "identity", "literals")}
                 if len(pair) == 2 else {})
    review_units = []
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        histories = {name: GraphHistory() for name in passes}
        for unit in story["units"]:
            versions, atoms, automated_reviews = {}, {}, {}
            for pass_name in passes:
                path = output / pass_name / story["story_id"] / (unit["id"] + ".json")
                if not path.is_file():
                    raise ValueError("Full-corpus audit requires all real annotations. Missing: " + str(path))
                saved = read_json(path)
                protocols[pass_name][saved.get("protocol_context", {}).get("protocol", "legacy-graph-v1")] += 1
                if (saved["graph_hash"] != object_hash(saved["graph"]) or
                        saved["input_identity"]["run_hash"] != object_hash(run) or
                        saved["input_identity"]["story_hash"] != entry["content_hash"]):
                    raise ValueError("Annotation provenance changed: " + str(path))
                graph = GraphDelta.model_validate(saved["graph"])
                if config.get("protocol") in ("joint-source-v4", "joint-source-v5"):
                    from .annotation_source import compile_annotation
                    from .annotation_source_jobs import load_saved_candidate
                    candidate = load_saved_candidate(output, saved)
                    compiled, _ = compile_annotation(candidate, story, unit, histories[pass_name])
                    if object_hash(compiled.model_dump()) != saved["graph_hash"]:
                        raise ValueError("Source annotation compilation changed: " + str(path))
                    semantic_checks[pass_name]["mechanical_only_no_model_review"] += 1
                if config.get("protocol") == "joint-source-v3":
                    from .annotation_joint import compile_annotation
                    context = saved.get("protocol_context", {})
                    raw = read_json(output / saved["raw_response"])
                    if context.get("protocol") != config["protocol"] or object_hash(raw) != context.get("response_hash"):
                        raise ValueError("Joint annotation provenance changed: " + str(path))
                    compiled, _ = compile_annotation(raw, story, unit, histories[pass_name])
                    if object_hash(compiled.model_dump()) != saved["graph_hash"]:
                        raise ValueError("Joint source compilation changed: " + str(path))
                    semantic_checks[pass_name]["mechanical_only_no_model_review"] += 1
                if config.get("protocol") == "source-anchored-v2":
                    from .annotation_anchored import SourceText, validate_review
                    context = saved.get("protocol_context", {})
                    check = context.get("semantic_check", {})
                    if (context.get("protocol") != config["protocol"] or check.get("status") != "passed"
                            or context.get("reviewed_graph_hash") != saved["graph_hash"]):
                        raise ValueError("Missing required semantic review: " + str(path))
                    review = read_json(output / check["raw_response"])
                    if object_hash(review) != check["response_hash"]:
                        raise ValueError("Semantic review provenance changed: " + str(path))
                    failures, _ = validate_review(review, SourceText(story, unit), graph)
                    if failures:
                        raise ValueError("Rejected semantic review in completed corpus: " + str(path))
                    semantic_checks[pass_name]["same_model_review_passed"] += 1
                    automated_reviews[pass_name] = review
                histories[pass_name].append(graph, unit)
                versions[pass_name] = graph.model_dump()
                atoms[pass_name] = semantic_atoms(graph, histories[pass_name])
                counts[pass_name].update({name: len(getattr(graph, name))
                                          for name in ("entities", "mentions", "events", "relations", "entity_updates", "uncertainties", "literals", "contexts", "identity_links")})
                counts[pass_name]["units"] += 1
                for item in graph.events:
                    phenomena[pass_name]["polarity:" + item.polarity] += 1
                    for status in histories[pass_name].event_statuses(item):
                        phenomena[pass_name]["factuality:" + status] += 1
                    for argument in item.arguments:
                        phenomena[pass_name]["role:" + argument.role] += 1
                for item in graph.relations:
                    phenomena[pass_name]["discourse:" + item.type] += 1
            differences = {}
            if len(pair) == 2:
                for family in agreement:
                    first, second = atoms[pair[0]][family], atoms[pair[1]][family]
                    agreement[family].update(first=len(first), second=len(second), shared=len(first & second))
                    differences[family] = {pair[0] + "_only": sorted(first - second, key=repr),
                                           pair[1] + "_only": sorted(second - first, key=repr)}
            review_units.append({"unit": unit, "story_split": entry["split"],
                                 "current_text": story["text"][unit["char_start"]:unit["prefix_char_end"]],
                                 "prefix_text": story["text"][:unit["prefix_char_end"]],
                                 "annotations": versions, "differences": differences,
                                 "automated_semantic_reviews": automated_reviews,
                                 "human_decision": None, "human_corrections": None})
    scores = {}
    for family, count in agreement.items():
        denominator = count["first"] + count["second"]
        scores[family] = {**dict(count), "symmetric_f1": 2 * count["shared"] / denominator if denominator else None}
    report = {"format_version": 1, "run_hash": object_hash(run), "corpus_hash": index["content_hash"],
              "audited_passes": passes, "incomplete_optional_passes": [name for name in config["passes"] if name not in passes],
              "counts": {k: dict(v) for k, v in counts.items()},
              "phenomena": {k: dict(v) for k, v in phenomena.items()}, "same_model_agreement": scores or None,
              "annotation_protocol_counts": {k: dict(v) for k, v in protocols.items()},
              "automated_semantic_checks": {k: dict(v) for k, v in semantic_checks.items()},
              "automated_review_interpretation": "Same-model checks are fallible and do not establish independent semantic accuracy.",
              "interpretation": ("Same-model repeatability is not label accuracy or independent human validation."
                                 if len(pair) == 2 else "Single annotation pass; repeatability was not measured. Human semantic review is pending."),
              "human_review": "pending", "units_for_review": len(review_units)}
    destination = root / "artifacts/annotation-audit" / output.name
    save_json(destination / "report.json", report)
    save_json(destination / "review.json", {"run_hash": object_hash(run), "units": review_units})
    save_json(destination / "source-only.json", {"run_hash": object_hash(run),
        "purpose": "Independent source-first annotation, including omissions; no model labels are included.",
        "units": [{k: u[k] for k in ("unit", "story_split", "current_text", "prefix_text")} for u in review_units]})
    print("ANNOTATION AUDIT:", destination / "report.json")
    print("Human review material:", destination / "review.json")
    print("Audited passes: " + ", ".join(passes) + ". Human semantic review remains pending.")
    return report
