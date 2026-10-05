"""Location selection and semantic-record completion are distinct repair tasks."""
from copy import deepcopy
import json

from .annotation_source import Annotation, get_path, set_path, response_schema, schema_at

PROTOCOL = "joint-source-v5"
REPAIR_VERSION = 2


def definitions(field, record):
    keys = {record["key"]} if "key" in record else set()
    if field == "referents":
        keys.update(m["key"] for m in record["mentions"])
    return keys


def references(field, record):
    """Inspect reference fields only, never descriptions or literal text values."""
    result = set(record.get("contexts", []))
    if field == "referents":
        result.add(record["existing"])
    elif field == "events":
        result.update(a["target"] for a in record["arguments"])
    elif field == "contexts":
        result.update(record["parents"])
        result.update((record["holder"], record["attribution_event"]))
    elif field == "relations":
        result.update((record["source"], record["target"]))
    elif field == "identity_links":
        result.update((record["left"], record["right"]))
    elif field == "entity_updates":
        result.add(record["entity"])
        if isinstance(record["value"], dict):
            result.add(record["value"]["literal"])
    return result - {None}


def affected_records(candidate, issues):
    """Allow the current-unit reference component to be edited atomically.

    This is a dependency calculation, not a semantic decision. Historical
    records are never editable. Literal words never create graph dependencies.
    """
    nodes = {(field, i): record for field, records in candidate.items() for i, record in enumerate(records)}
    owners = {}
    for path, record in nodes.items():
        for key in definitions(path[0], record):
            owners.setdefault(key, set()).add(path)
    neighbours = {path: set() for path in nodes}
    for path, record in nodes.items():
        keys = references(path[0], record) | definitions(path[0], record)
        for key in keys:
            for other in owners.get(key, ()):
                neighbours[path].add(other)
                neighbours[other].add(path)
    selected = {tuple(issue["path"][:2]) for issue in issues}
    if not selected or not selected <= set(nodes):
        raise ValueError("A record-completion issue must identify a current annotation record.")
    queue = list(selected)
    while queue:
        path = queue.pop()
        for other in neighbours[path] - selected:
            selected.add(other)
            queue.append(other)
    return sorted(selected)


def correction_request(base, candidate, issues):
    location_only = bool(issues) and all(issue.get("choices") for issue in issues)
    tasks, properties = [], {}
    if location_only:
        seen = set()
        for issue in issues:
            path = tuple(issue["path"])
            if path in seen:
                continue
            seen.add(path)
            field = f"fix_{len(tasks)}"
            tasks.append({**issue, "field": field, "mode": "location", "repair_version": REPAIR_VERSION})
            properties[field] = {"type": "string", "enum": [c["source_id"] for c in issue["choices"]]}
        instructions = ("Select the intended source location for each requested field. "
                        "Only these source locations will change; the semantic records stay unchanged.")
    else:
        for path in affected_records(candidate, issues):
            field = f"record_{len(tasks)}"
            tasks.append({"path": list(path), "field": field, "mode": "record", "repair_version": REPAIR_VERSION})
            properties[field] = {"anyOf": [schema_at(list(path)), {"type": "null"}]}
        tasks.append({"path": [], "field": "additions", "mode": "additions", "repair_version": REPAIR_VERSION})
        annotation_schema = response_schema()
        properties["additions"] = {k: v for k, v in annotation_schema.items() if k != "$defs"}
        instructions = (
            "Complete the affected semantic records using the source text. Return each requested record "
            "unchanged or corrected, or null to delete an unsupported record. You can remove an unsupported "
            "mention or argument inside a replacement record. Add missing referents, literals, events or "
            "contexts in additions, and connect them using their local keys. Empty addition arrays mean no additions. "
            "An unrecognized target can be meaningful text whose node was omitted: preserve its meaning by "
            "creating the appropriate source-supported record. Existing references are options, not compulsory answers. "
            "Do not substitute an unrelated reference to satisfy validation. Update affected links together when "
            "removing or renaming records. Preserve supported content; fix the listed defects without a general critique. "
            "For ambiguous quotations, use the supplied source_id or a surrounding source quote in the replacement. "
            "Only the listed current records and additions can change; earlier story records cannot be rewritten.")
    schema = {"type": "object", "properties": properties, "required": list(properties),
              "additionalProperties": False, "$defs": response_schema()["$defs"]}
    payload = {"annotation": candidate, "reported_defects": issues, "editable_fields": tasks}
    prompt = (base + ("\n\nSOURCE LOCATION SELECTION\n" if location_only else "\n\nSEMANTIC RECORD COMPLETION\n")
              + instructions + "\n" + json.dumps(payload, ensure_ascii=False))
    return prompt, schema, tasks


def apply_corrections(candidate, response, tasks):
    if not isinstance(response, dict) or set(response) != {t["field"] for t in tasks}:
        raise ValueError("Correction must contain exactly the requested fields.")
    if not tasks or any(t.get("repair_version") != REPAIR_VERSION for t in tasks):
        raise ValueError("Unsupported record-completion protocol.")
    result = deepcopy(candidate)
    if all(t["mode"] == "location" for t in tasks):
        for task in tasks:
            value = response[task["field"]]
            if value not in [c["source_id"] for c in task["choices"]]:
                raise ValueError("Correction selected an unavailable source location.")
            set_path(result, task["path"], {"quote": task["quote"], "source_id": value})
    else:
        if any(t["mode"] not in ("record", "additions") for t in tasks):
            raise ValueError("Record completion cannot mix arbitrary field edits.")
        addition_tasks = [t for t in tasks if t["mode"] == "additions"]
        if len(addition_tasks) != 1:
            raise ValueError("Record completion needs exactly one additions object.")
        replacements = {}
        for task in tasks:
            if task["mode"] == "record":
                path = tuple(task["path"])
                if len(path) != 2 or path in replacements:
                    raise ValueError("Duplicate or invalid replacement target.")
                get_path(candidate, path)  # All positions refer to the original draft.
                replacements[path] = response[task["field"]]
        additions = Annotation.model_validate(response[addition_tasks[0]["field"]]).model_dump()
        for field, records in candidate.items():
            replaced = [deepcopy(replacements.get((field, i), record)) for i, record in enumerate(records)]
            result[field] = [record for record in replaced if record is not None] + additions[field]
    # Apply atomically; a malformed completion never damages the retained draft.
    return Annotation.model_validate(result).model_dump()
