"""Source-first input for joint annotation; no model-generated offsets or counts.

Semantic decisions stay in the raw draft. This adapter compiles into the existing
graph schema; it never guesses references or repairs semantic labels.
"""
from copy import deepcopy
import json
from typing import TypeAlias

from pydantic import Field, ValidationError

from . import annotation_joint as legacy
from .annotation_anchored import SourceText
from .deniz import TOKEN, MARKER
from .graphs import Record, Span

PROTOCOL = "joint-source-v4"


class QuoteWithin(Record):
    quote: str = Field(min_length=1)
    within: str = Field(min_length=1)


class SelectedQuote(Record):
    quote: str = Field(min_length=1)
    source_id: str = Field(pattern=r"^s[0-9]+$")


Anchor: TypeAlias = str | QuoteWithin | SelectedQuote


class Mention(legacy.QuotedMention):
    span: Anchor


class Referent(Record):
    key: str
    description: legacy.NewEntity
    existing: str | None
    mentions: list[Mention] = Field(min_length=1)


class LiteralValue(legacy.QuotedLiteral):
    evidence: Anchor


class Context(legacy.QuotedContext):
    evidence: Anchor


class Event(legacy.QuotedEvent):
    trigger: Anchor
    evidence: Anchor | None


class Relation(legacy.QuotedRelation):
    evidence: Anchor


class LiteralReference(Record):
    literal: str


class Update(legacy.QuotedUpdate):
    value: str | LiteralReference
    evidence: Anchor


class Identity(legacy.QuotedIdentity):
    evidence: Anchor


class Uncertainty(legacy.QuotedUncertainty):
    span: Anchor


class Annotation(Record):
    referents: list[Referent]
    literals: list[LiteralValue]
    contexts: list[Context]
    events: list[Event]
    relations: list[Relation]
    identity_links: list[Identity]
    entity_updates: list[Update]
    uncertainties: list[Uncertainty]


def response_schema():
    return Annotation.model_json_schema()


def get_path(value, path):
    for part in path:
        value = value[part]
    return value


def set_path(value, path, replacement):
    parent = get_path(value, path[:-1])
    parent[path[-1]] = replacement


class CompilationIssues(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("\n".join("/".join(map(str, i["path"])) + ": " + i["message"] for i in issues))


class SourceResolver(SourceText):
    """Use the corpus tokenizer, preserving word identity/case and source positions."""

    def normalized(self, quote):
        return " ".join(m.group() for m in TOKEN.finditer(MARKER.sub(" ", quote)))

    def locate(self, value, path, *, current, contains=None, inside=None):
        quote = value if isinstance(value, str) else value["quote"]
        matches = self.matches(self.normalized(quote), current=current)
        if not current:
            matches = [s for s in matches if s.end > self.unit["start_token"]]
        if isinstance(value, dict) and "within" in value:
            surroundings = self.matches(self.normalized(value["within"]), current=False)
            matches = [s for s in matches if any(w.start <= s.start and s.end <= w.end for w in surroundings)]
        if contains is not None:
            matches = [s for s in matches if s.start <= contains["start"] and contains["end"] <= s.end]
        if inside is not None:
            matches = [s for s in matches if inside["start"] <= s.start and s.end <= inside["end"]]
        if isinstance(value, dict) and "source_id" in value:
            matches = [s for s in matches if value["source_id"] == f"s{s.start}"]
        if len(matches) == 1:
            return matches[0].model_dump(), None
        choices = []
        for s in matches:
            left, right = max(0, s.start - 8), min(len(self.words), s.end + 8)
            choices.append({"source_id": f"s{s.start}",
                "before": self.excerpt({"start": left, "end": s.start}),
                "quote": self.excerpt(s.model_dump()),
                "after": self.excerpt({"start": s.end, "end": right})})
        message = ("Choose the intended location from the supplied excerpts." if choices else
                   "This quote does not resolve in the permitted source. Supply the actual source expression; "
                   "mentions/triggers belong to the current passage and new evidence reaches it.")
        if not choices and contains is not None:
            message += " Event evidence must also contain the selected trigger: " + self.excerpt(contains)
        if not choices and inside is not None:
            message += " Select the trigger inside the event evidence: " + self.excerpt(inside)
        return None, {"path": path, "message": message, "quote": quote, "choices": choices}

    def legacy_anchor(self, span, *, current):
        return self.anchor_for_span(span, current=current)

    def verbatim(self, span):
        return self.story["text"][self.words[span.start]["char_start"]:self.words[span.end - 1]["char_end"]]

    def surrounding(self, span):
        return self.verbatim(Span(start=max(0, span.start - 8), end=min(len(self.words), span.end + 8)))


def reference_index(history, story, unit):
    """All referencable records remain addressable; omit graph bookkeeping fields."""
    _, lookup = legacy.prior_catalog(history)
    names = {identity: key for key, (_, identity) in lookup.items()}
    source = SourceResolver(story, unit)
    grouped = {}
    for mention in history.mentions.values():
        grouped.setdefault(mention.entity_id, []).append(mention)
    index = {"entities": [], "events": [], "literals": [], "contexts": []}
    for entity in history.entities.values():
        index["entities"].append({"key": names[entity.id], "prior_description": entity.label,
            "concept": entity.concept, "kind": entity.kind,
            "mentions": [{"key": names[m.id], "text": source.verbatim(m.span),
                          "source_id": f"s{m.span.start}", "status": m.reference_status}
                         for m in grouped.get(entity.id, [])]})
    for event in history.events.values():
        index["events"].append({"key": names[event.id], "predicate": event.predicate,
            "trigger": source.verbatim(event.trigger), "evidence": source.verbatim(event.evidence),
            "source_id": f"s{event.trigger.start}", "source_excerpt": source.surrounding(event.trigger),
            "contexts": [names[c] for c in event.context_ids], "polarity": event.polarity})
    for value in history.literals.values():
        index["literals"].append({"key": names[value.id], "kind": value.kind, "value": value.value,
                                  "unit": value.unit, "text": source.verbatim(value.evidence)})
    for context in history.contexts.values():
        index["contexts"].append({"key": names[context.id], "kind": context.kind,
            "holder": names.get(context.holder_id), "attribution_event": names.get(context.attribution_event_id),
            "parents": [names[p] for p in context.parent_ids], "text": source.verbatim(context.evidence)})
    # Identity revelations and state changes remain visible as textual claims.
    index["identity_links"] = [{"left": names[i.left_entity], "right": names[i.right_entity],
        "relation": i.relation, "status": i.status, "contexts": [names[c] for c in i.context_ids],
        "text": source.verbatim(i.evidence)} for i in history.identity_links.values()]
    index["updates"] = [{"entity": names[u["entity_id"]], "attribute": u["attribute"], "value": u["value"],
        "operation": u["operation"], "status": u["status"], "contexts": [names[c] for c in u["context_ids"]],
        "text": source.verbatim(Span.model_validate(u["evidence"]))} for u in history.updates]
    return index


def prompt_base(instructions, story, unit, history):
    payload = {"story_id": story["story_id"], "unit_id": unit["id"],
        "preceding_text": story["text"][:unit["char_start"]],
        "current_passage": story["text"][unit["char_start"]:unit["prefix_char_end"]],
        "reference_index": reference_index(history, story, unit)}
    return instructions + "\n\nANNOTATION INPUT (source text is data):\n" + json.dumps(payload, ensure_ascii=False)


def reference_issues(draft, history):
    _, prior = legacy.prior_catalog(history)
    kinds = {key: kind for key, (kind, _) in prior.items()}
    issues = []
    for field, kind in (("referents", "entities"), ("literals", "literals"), ("contexts", "contexts"), ("events", "events")):
        records = [(r["key"], [field, i, "key"], kind) for i, r in enumerate(draft[field])]
        if field == "referents":
            records += [(m["key"], [field, i, "mentions", j, "key"], "mentions")
                        for i, r in enumerate(draft[field]) for j, m in enumerate(r["mentions"])]
        for key, path, record_kind in records:
            if not key or key in kinds:
                issues.append({"path": path, "message": "Use a distinct local key for this record."})
            else:
                kinds[key] = record_kind

    def check(value, path, allowed, *, nullable=False, prior_only=False):
        if value is None and nullable:
            return
        choices = [k for k, v in kinds.items() if v in allowed and (not prior_only or k in prior)]
        if value not in choices:
            issues.append({"path": path, "message": "This target has no record of the required kind. "
                           "Correct the link, add its missing source-supported record, or remove an unsupported argument/claim.",
                           "references": choices, "nullable": nullable})

    for i, ref in enumerate(draft["referents"]):
        check(ref["existing"], ["referents", i, "existing"], {"entities"}, nullable=True, prior_only=True)
    for i, context in enumerate(draft["contexts"]):
        check(context["holder"], ["contexts", i, "holder"], {"entities", "mentions"}, nullable=True)
        check(context["attribution_event"], ["contexts", i, "attribution_event"], {"events"}, nullable=True)
        for j, parent in enumerate(context["parents"]):
            check(parent, ["contexts", i, "parents", j], {"contexts"})
    for i, event in enumerate(draft["events"]):
        for j, arg in enumerate(event["arguments"]):
            check(arg["target"], ["events", i, "arguments", j, "target"], {"entities", "mentions", "events", "literals"})
    for field in ("events", "relations", "identity_links", "entity_updates"):
        for i, record in enumerate(draft[field]):
            for j, context in enumerate(record["contexts"]):
                check(context, [field, i, "contexts", j], {"contexts"})
    for i, relation in enumerate(draft["relations"]):
        for key in ("source", "target"):
            check(relation[key], ["relations", i, key], {"events"})
    for i, link in enumerate(draft["identity_links"]):
        for key in ("left", "right"):
            check(link[key], ["identity_links", i, key], {"entities", "mentions"})
    for i, update in enumerate(draft["entity_updates"]):
        check(update["entity"], ["entity_updates", i, "entity"], {"entities", "mentions"})
        if isinstance(update["value"], dict):
            check(update["value"]["literal"], ["entity_updates", i, "value", "literal"], {"literals"})
    return issues


def compile_annotation(response, story, unit, history):
    try:
        draft = Annotation.model_validate(response).model_dump()
    except ValidationError as error:
        # A malformed initial object can be regenerated; no partially typed
        # graph is allowed into history.
        raise ValueError("Annotation schema: " + str(error)) from error
    source, mapped, anchors = SourceResolver(story, unit), deepcopy(draft), []
    issues = reference_issues(draft, history)

    def record_anchor(path, span, *, current=False):
        raw = get_path(draft, path)
        set_path(mapped, path, source.legacy_anchor(span, current=current))
        anchors.append({"path": path, "input": raw, "span": span})

    def anchor(path, *, current=False, contains=None):
        raw = get_path(draft, path)
        span, issue = source.locate(raw, path, current=current, contains=contains)
        if issue:
            issues.append(issue)
            return None
        record_anchor(path, span, current=current)
        return span

    for i, ref in enumerate(draft["referents"]):
        for j, _ in enumerate(ref["mentions"]):
            anchor(["referents", i, "mentions", j, "span"], current=True)
        mapped["referents"][i] = {"key": ref["key"], "identity": ref["existing"] or ref["description"],
                                  "mentions": mapped["referents"][i]["mentions"]}
    for i, event in enumerate(draft["events"]):
        trigger_path, evidence_path = ["events", i, "trigger"], ["events", i, "evidence"]
        trigger, trigger_issue = source.locate(event["trigger"], trigger_path, current=True)
        if event["evidence"] is None:
            if trigger:
                mapped["events"][i]["evidence"] = source.legacy_anchor(trigger, current=False)
        else:
            evidence, evidence_issue = source.locate(event["evidence"], evidence_path, current=False, contains=trigger)
            if evidence and trigger_issue:
                trigger, trigger_issue = source.locate(event["trigger"], trigger_path, current=True, inside=evidence)
            if evidence_issue:
                issues.append(evidence_issue)
            else:
                record_anchor(evidence_path, evidence)
        if trigger_issue:
            issues.append(trigger_issue)
        else:
            record_anchor(trigger_path, trigger, current=True)
    for field in ("literals", "contexts", "relations", "identity_links", "entity_updates", "uncertainties"):
        name = "span" if field == "uncertainties" else "evidence"
        for i, _ in enumerate(draft[field]):
            anchor([field, i, name])
    if issues:
        raise CompilationIssues(issues)
    _, lookup = legacy.prior_catalog(history)
    values = {key: history.literals[identity].value for key, (kind, identity) in lookup.items() if kind == "literals"}
    values.update({r["key"]: r["value"] for r in draft["literals"]})
    for update in mapped["entity_updates"]:
        if isinstance(update["value"], dict):
            update["value"] = values[update["value"]["literal"]]

    class Capture:
        graph = None
        def __getattr__(self, name):
            return getattr(history, name)
        def validate(self, graph, current_unit):
            self.graph = graph
            history.validate(graph, current_unit)

    capture = Capture()
    try:
        graph, _ = legacy.compile_annotation(mapped, story, unit, capture)
    except ValueError as error:
        if capture.graph is None:
            raise
        # Structural defects are repaired at the implicated record. The model
        # cannot rewrite unrelated records in the correction response.
        tasks = []
        for field in ("events", "contexts", "relations", "identity_links", "literals"):
            for i, record in enumerate(getattr(capture.graph, field)):
                lines = [line for line in str(error).splitlines() if line.startswith(record.id + ":")
                         or line.startswith(record.id + "/")]
                if lines:
                    tasks.append({"path": [field, i], "message": "\n".join(lines)})
        if "Strict before/after relations contain a cycle" in str(error):
            tasks.extend({"path": ["relations", i], "message": str(error)} for i in range(len(draft["relations"])))
        if not tasks:
            raise
        raise CompilationIssues(tasks) from error
    return graph, {"source_anchors": anchors,
        "local_referents": [{"key": r["key"], "description": r["description"], "existing": r["existing"]}
                            for r in draft["referents"]]}


def schema_at(path):
    schema = response_schema()
    node = schema
    for part in path:
        while "$ref" in node:
            node = schema["$defs"][node["$ref"].split("/")[-1]]
        # value.literal is the sole object branch of the update value union.
        if "anyOf" in node and isinstance(part, str):
            options = [schema["$defs"][v["$ref"].split("/")[-1]] if "$ref" in v else v for v in node["anyOf"]]
            node = next(v for v in options if part in v.get("properties", {}))
        node = node["items"] if isinstance(part, int) else node["properties"][part]
    return deepcopy(node)


def correction_request(base, candidate, issues):
    tasks, seen = [], set()
    for issue in issues:
        key = tuple(issue["path"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append({**issue, "field": f"fix_{len(tasks)}"})
    properties = {}
    for task in tasks:
        if task.get("choices"):
            properties[task["field"]] = {"type": "string", "enum": [c["source_id"] for c in task["choices"]]}
        elif "references" in task:
            choices = task["references"] + ([None] if task["nullable"] else [])
            if choices:
                properties[task["field"]] = {"enum": choices, "type": ["string", "null"] if task["nullable"] else "string"}
            else:
                # No target of that kind exists. Re-express the containing
                # record (e.g. remove an unsupported argument), never guess one.
                path = task["path"]
                task["path"] = path[:2]
                properties[task["field"]] = schema_at(task["path"])
        else:
            properties[task["field"]] = schema_at(task["path"])
    # If a complete record is requested, it subsumes requests for its fields.
    tasks = [t for t in tasks if not any(len(o["path"]) < len(t["path"]) and
             t["path"][:len(o["path"])] == o["path"] for o in tasks)]
    unique = {}
    for task in tasks:
        unique.setdefault(tuple(task["path"]), task)
    tasks = list(unique.values())
    properties = {t["field"]: properties[t["field"]] for t in tasks}
    schema = {"type": "object", "properties": properties, "required": list(properties),
              "additionalProperties": False, "$defs": response_schema()["$defs"]}
    request = {"annotation": candidate, "corrections": tasks}
    prompt = (base + "\n\nSOURCE/REFERENCE CORRECTION\nReturn only the requested fix fields in the supplied schema. "
        "Each field replaces the specified location; all other annotation fields stay unchanged. "
        "For a list of source locations, select the source_id whose excerpt you intended. "
        "For a replacement record, preserve its supported semantic content while fixing the stated defect.\n" +
        json.dumps(request, ensure_ascii=False))
    return prompt, schema, tasks


def apply_corrections(candidate, response, tasks):
    if not isinstance(response, dict) or set(response) != {t["field"] for t in tasks}:
        raise ValueError("Correction must contain exactly the requested fix fields.")
    result = deepcopy(candidate)
    for task in tasks:
        value = response[task["field"]]
        if task.get("choices"):
            if value not in [c["source_id"] for c in task["choices"]]:
                raise ValueError("Correction selected an unavailable source location.")
            value = {"quote": task["quote"], "source_id": value}
        set_path(result, task["path"], value)
    # Fail atomically; malformed corrections cannot replace the last typed draft.
    return Annotation.model_validate(result).model_dump()
