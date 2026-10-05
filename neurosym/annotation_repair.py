"""Typed graph edits and logged derivation of redundant graph bookkeeping."""

from collections import Counter
import copy
import json

from .graphs import GraphDelta, GraphHistory, Span, graph_schema


def repair_schema() -> dict:
    definitions = graph_schema()["$defs"]
    records = [{"$ref": "#/$defs/" + name} for name in
               ("Span", "Entity", "Mention", "Argument", "Event", "Relation", "EntityUpdate", "Uncertainty")]
    scalars = [{"type": kind} for kind in ("string", "number", "boolean", "null")]
    value = {"anyOf": [*scalars, *records, {"type": "array", "items": {"anyOf": [*scalars, *records]}}]}
    return {"type": "object", "$defs": definitions, "properties": {
        "edits": {"type": "array", "minItems": 1, "items": {
            "type": "object", "properties": {"op": {"type": "string", "enum": ["add", "replace", "remove"]},
                                              "path": {"type": "string"}, "value": value},
            "required": ["op", "path", "value"], "additionalProperties": False}}},
        "required": ["edits"], "additionalProperties": False}


def apply_graph_edits(graph: dict, response: dict, *, legacy: bool = False) -> dict:
    """Apply explicit model edits; the ordinary schema/history validators still decide acceptance."""
    if not isinstance(response, dict) or set(response) != {"edits"} or not isinstance(response["edits"], list):
        raise ValueError("Repair output must contain an edits array.")
    if not response["edits"]:
        raise ValueError("Repair returned no edits for an invalid graph.")
    result = copy.deepcopy(graph)
    fields = {"entities", "mentions", "events", "relations", "entity_updates", "uncertainties"}
    for edit in response["edits"]:
        value_key = "value_json" if legacy else "value"
        if not isinstance(edit, dict) or set(edit) != {"op", "path", value_key}:
            raise ValueError("Each edit requires op, path, and " + value_key + ". Use native JSON values for value.")
        operation, path = edit["op"], edit["path"]
        if operation not in ("add", "replace", "remove") or not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("Invalid repair operation/path.")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]
        if parts[0] not in fields:
            raise ValueError("Repairs cannot change schema, story, or unit identity.")
        value = copy.deepcopy(edit[value_key])
        if operation == "remove" and value is not None:
            raise ValueError("A remove edit must use a null value.")
        parent = result
        for part in parts[:-1]:
            if isinstance(parent, list):
                if not part.isdigit() or not 0 <= int(part) < len(parent):
                    raise ValueError("Repair array path is out of range: " + path)
                parent = parent[int(part)]
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                raise ValueError("Repair path does not exist: " + path)
        key = parts[-1]
        if legacy and operation != "remove" and isinstance(value, str):
            # Compatibility for the actual old replies: scalar strings were
            # often returned unquoted while spans were encoded JSON objects.
            current = (parent.get(key) if isinstance(parent, dict) else
                       parent[int(key)] if isinstance(parent, list) and key.isdigit() and int(key) < len(parent) else None)
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
            value = value if isinstance(current, str) and not isinstance(decoded, str) else decoded
        if isinstance(parent, list):
            if operation == "add" and key == "-":
                position = len(parent)
            elif key.isdigit():
                position = int(key)
            else:
                raise ValueError("Repair array index must be nonnegative: " + path)
            limit = len(parent) if operation == "add" else len(parent) - 1
            if not 0 <= position <= limit:
                raise ValueError("Repair array index is out of range: " + path)
            if operation == "add":
                parent.insert(position, value)
            elif operation == "replace":
                parent[position] = value
            else:
                parent.pop(position)
        elif isinstance(parent, dict):
            if operation != "add" and key not in parent:
                raise ValueError("Repair field does not exist: " + path)
            if operation == "remove":
                del parent[key]
            else:
                parent[key] = value
        else:
            raise ValueError("Repair parent is not a collection: " + path)
    return result


def repair_prompt(base_prompt: str, candidate: dict, error: str, history: GraphHistory, protocol_error: str = "") -> str:
    # The original task is quoted context. Only the repair protocol governs the
    # response format; no new source text or semantic decisions are supplied.
    rows = candidate.get("mentions", [])
    rows = rows if isinstance(rows, list) else []
    mentions = {**{key: value.model_dump() for key, value in history.mentions.items()},
                **{item["id"]: item for item in rows if isinstance(item, dict) and isinstance(item.get("id"), str)}}
    facts = []
    for item in rows:
        if (isinstance(item, dict) and isinstance(item.get("antecedent_id"), str)
                and item["antecedent_id"] in mentions):
            anchor = mentions[item["antecedent_id"]]
            facts.append({"mention_id": item.get("id"), "entity_id": item.get("entity_id"),
                          "antecedent_id": item["antecedent_id"], "antecedent_entity_id": anchor.get("entity_id")})
    return (
        "Repair an invalid semantic graph using only the quoted annotation task and its story prefix. "
        "Return ONLY edits matching the supplied repair schema, not a regenerated graph. "
        "Do not use tools. Each edit has op=add/replace/remove and an RFC6901-style path such as /entities/0/first_mention. "
        "value is a NATIVE JSON value: a string, number, boolean, null, object, or array. "
        "Never encode JSON inside a string. For remove, value is null. Edits apply sequentially; "
        "remove higher array indices first. Preserve all unaffected records and meaning. "
        "Fix inconsistent IDs in every affected mention/argument together. "
        "Entity first_mention and event evidence containment are compiled from the existing mentions/triggers. "
        "Do not change semantic choices just to adjust that bookkeeping. Reuse existing entities when justified; "
        "remove unused duplicate declarations. Do not invent mentions, erase meaningful assertions, or move "
        "evidence merely to pass validation. Resolve ambiguities from the supplied text or retain uncertainty. "
        "The quoted task's graph-output instruction describes the desired GRAPH, but your output here is only EDITS.\n"
        + json.dumps({"quoted_annotation_task": base_prompt, "invalid_graph": candidate,
                      "validation_errors": error, "previous_repair_format_error": protocol_error,
                      "reference_integrity_facts": facts},
                     ensure_ascii=False, separators=(",", ":"))
    )


def normalize_graph(graph: GraphDelta, history: GraphHistory, unit: dict | None = None) -> tuple[GraphDelta, list[dict]]:
    """Derive bookkeeping fields without changing mentions, triggers, or semantic links.

    A first-mention mismatch is resolved from the entity's existing current-unit
    mentions. Evidence containment uses the smallest span covering the model's
    existing evidence and trigger. Neither operation invents mention/trigger spans
    or resolves a referent. All changes are recorded before normal validation.
    """
    repairs = []
    graph = graph.model_copy(deep=True)
    if unit is not None:
        for entity in graph.entities:
            mentions = [m for m in graph.mentions if m.entity_id == entity.id
                        and unit["start_token"] <= m.span.start < m.span.end <= unit["end_token"]]
            if mentions and not any(m.span == entity.first_mention for m in mentions):
                first = min(mentions, key=lambda m: (m.span.start, m.span.end, m.id))
                repairs.append({"operation": "derive_first_mention_from_mentions", "entity_id": entity.id,
                                "before": entity.first_mention.model_dump(), "after": first.span.model_dump(),
                                "source_mention_id": first.id})
                entity.first_mention = first.span.model_copy()
        for event in graph.events:
            if (unit["start_token"] <= event.trigger.start < event.trigger.end <= unit["end_token"]
                    and event.evidence.end <= unit["end_token"]
                    and not event.evidence.start <= event.trigger.start < event.trigger.end <= event.evidence.end):
                combined = Span(start=min(event.evidence.start, event.trigger.start),
                                end=max(event.evidence.end, event.trigger.end))
                repairs.append({"operation": "include_trigger_in_evidence", "event_id": event.id,
                                "before": event.evidence.model_dump(), "after": combined.model_dump(),
                                "trigger": event.trigger.model_dump()})
                event.evidence = combined
    referenced = {item.entity_id for item in graph.mentions}
    referenced.update(argument.target_id for event in graph.events for argument in event.arguments
                      if argument.target_kind == "entity")
    referenced.update(item.entity_id for item in graph.entity_updates)
    id_counts = Counter(item.id for field in ("entities", "mentions", "events", "relations")
                        for item in getattr(graph, field))
    retained = []
    for entity in graph.entities:
        anchors = []
        if (entity.id not in referenced and entity.id not in history.entities and id_counts[entity.id] == 1
                and entity.id.startswith(graph.unit_id + "_n")):
            for mention in graph.mentions:
                target = history.entities.get(mention.entity_id)
                antecedent = history.mentions.get(mention.antecedent_id)
                if (mention.span == entity.first_mention and mention.reference_status in ("explicit", "inferred")
                        and target is not None and antecedent is not None
                        and antecedent.entity_id == target.id
                        and target.concept == entity.concept and target.kind == entity.kind):
                    anchors.append(mention)
        targets = {mention.entity_id for mention in anchors}
        if len(targets) == 1:
            repairs.append({"operation": "remove_unreferenced_duplicate_entity", "entity": entity.model_dump(),
                            "existing_entity_id": next(iter(targets)),
                            "supporting_mention_ids": [mention.id for mention in anchors]})
        else:
            retained.append(entity)
    if not repairs:
        return graph, []
    return graph.model_copy(update={"entities": retained}), repairs
