"""Semantic graph schema and cross-unit validation for prefix-only annotations."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Span(Record):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Token spans are nonempty and half-open: start < end.")
        return self


class Entity(Record):
    id: str
    label: str
    concept: str = Field(pattern=r"^[a-z0-9][a-z0-9_'-]*$")
    kind: Literal["person", "group", "animal", "object", "location", "time", "quantity", "abstract", "other", "unresolved"]
    first_mention: Span


class Mention(Record):
    id: str
    entity_id: str
    span: Span
    form: Literal["name", "pronoun", "description", "deictic", "other"]
    antecedent_id: str | None
    reference_status: Literal["explicit", "inferred", "unresolved", "first_mention"]
    confidence: float | None = Field(default=None, ge=0, le=1)


class Argument(Record):
    role: Literal["agent", "patient", "theme", "recipient", "experiencer", "stimulus", "beneficiary", "instrument",
                  "source", "goal", "location", "time", "manner", "cause", "content", "attribute", "other"]
    frame_role: str | None
    target_kind: Literal["entity", "event", "literal"]
    target_id: str
    mention_id: str | None


class Event(Record):
    id: str
    predicate: str = Field(pattern=r"^[a-z0-9][a-z0-9_'-]*$")
    sense_id: str | None
    trigger: Span
    evidence: Span
    arguments: list[Argument]
    polarity: Literal["positive", "negative", "unresolved"]
    # Legacy v1 fields are retained for reading archived annotations. New graphs
    # use explicit, nestable contexts rather than choosing one exclusive status.
    status: Literal["asserted", "hypothetical", "desired", "promised", "questioned", "reported", "uncertain"] | None = None
    tense: Literal["past", "present", "future", "unspecified"]
    scope_parent_id: str | None = None
    context_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Relation(Record):
    id: str
    type: Literal["before", "after", "overlap", "causes", "motivates", "enables", "condition", "contrast", "elaboration", "same_event"]
    source_event: str
    target_event: str
    evidence: Span
    status: Literal["explicit", "inferred", "unresolved"]
    context_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class EntityUpdate(Record):
    entity_id: str
    attribute: str
    value: str
    operation: Literal["assert", "retract"]
    evidence: Span
    status: Literal["explicit", "inferred", "unresolved"]
    context_ids: list[str] = Field(default_factory=list)


class LiteralValue(Record):
    id: str
    kind: Literal["name", "text", "number", "quantity", "date", "time", "boolean"]
    value: str
    unit: str | None
    evidence: Span


class Context(Record):
    id: str
    kind: Literal["reported", "believed", "hypothetical", "desired", "promised", "questioned", "uncertain", "negated"]
    holder_id: str | None
    attribution_event_id: str | None
    parent_ids: list[str]
    evidence: Span


class IdentityLink(Record):
    id: str
    left_entity: str
    right_entity: str
    relation: Literal["same", "different", "possible"]
    status: Literal["explicit", "inferred", "unresolved"]
    context_ids: list[str]
    evidence: Span


class Uncertainty(Record):
    span: Span
    category: Literal["reference", "role", "predicate", "scope", "discourse", "text"]
    explanation: str


class GraphDelta(Record):
    schema_version: Literal[1, 2]
    story_id: str
    unit_id: str
    entities: list[Entity]
    mentions: list[Mention]
    events: list[Event]
    relations: list[Relation]
    entity_updates: list[EntityUpdate]
    uncertainties: list[Uncertainty]
    literals: list[LiteralValue] = Field(default_factory=list)
    contexts: list[Context] = Field(default_factory=list)
    identity_links: list[IdentityLink] = Field(default_factory=list)


def graph_schema() -> dict:
    """Use enum for singleton literals for structured-output schema compatibility."""
    schema = GraphDelta.model_json_schema()

    def normalize(value):
        if isinstance(value, dict):
            if "const" in value:
                value["enum"] = [value.pop("const")]
            for item in value.values():
                normalize(item)
        elif isinstance(value, list):
            for item in value:
                normalize(item)

    normalize(schema)
    return schema


class GraphHistory:
    """Append-only history. Later information never rewrites earlier graph snapshots."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.mentions: dict[str, Mention] = {}
        self.events: dict[str, Event] = {}
        self.relations: dict[str, Relation] = {}
        self.literals: dict[str, LiteralValue] = {}
        self.contexts: dict[str, Context] = {}
        self.identity_links: dict[str, IdentityLink] = {}
        self.updates: list[dict] = []
        self.available_at: dict[str, int] = {}

    def validate(self, graph: GraphDelta, unit: dict) -> None:
        if graph.unit_id != unit["id"] or graph.story_id != unit["story_id"]:
            raise ValueError("Graph unit/story IDs differ from the supplied prefix.")
        errors = []

        def span(value: Span, location: str, *, current: bool = False, introduces: bool = False):
            if value.end > unit["end_token"] or (current and value.start < unit["start_token"]):
                errors.append(f"{location}: evidence is outside the permitted text.")
            if introduces and value.end <= unit["start_token"]:
                errors.append(f"{location}: a new assertion needs evidence in the current unit.")

        all_old = set().union(*(getattr(self, f) for f in
            ("entities", "mentions", "events", "relations", "literals", "contexts", "identity_links")))
        all_new = set()
        for field, prefix in (("entities", "n"), ("mentions", "m"), ("events", "e"), ("relations", "r"),
                              ("literals", "v"), ("contexts", "c"), ("identity_links", "i")):
            for item in getattr(graph, field):
                if not item.id.startswith(unit["id"] + "_" + prefix) or item.id in all_old or item.id in all_new:
                    errors.append(f"Invalid, reused, or duplicate {field} ID: {item.id}")
                all_new.add(item.id)
        entities = {**self.entities, **{item.id: item for item in graph.entities}}
        mentions = {**self.mentions, **{item.id: item for item in graph.mentions}}
        events = {**self.events, **{item.id: item for item in graph.events}}
        literals = {**self.literals, **{item.id: item for item in graph.literals}}
        contexts = {**self.contexts, **{item.id: item for item in graph.contexts}}

        def check_contexts(ids, location):
            if len(ids) != len(set(ids)) or any(c not in contexts for c in ids):
                errors.append(f"{location}: unknown or duplicate context reference.")

        for item in graph.literals:
            span(item.evidence, item.id, introduces=True)
        for item in graph.contexts:
            span(item.evidence, item.id, introduces=True)
            check_contexts(item.parent_ids, item.id)
            if item.holder_id is not None and item.holder_id not in entities:
                errors.append(f"{item.id}: unknown context holder.")
            if item.attribution_event_id is not None and item.attribution_event_id not in events:
                errors.append(f"{item.id}: unknown attribution event.")
        for item in graph.identity_links:
            span(item.evidence, item.id, introduces=True)
            check_contexts(item.context_ids, item.id)
            if item.left_entity not in entities or item.right_entity not in entities or item.left_entity == item.right_entity:
                errors.append(f"{item.id}: identity link needs two distinct known entity IDs.")
        for item in graph.entities:
            span(item.first_mention, item.id, current=True)
            if not any(m.entity_id == item.id and m.span == item.first_mention for m in graph.mentions):
                errors.append(f"{item.id}: first_mention needs a corresponding mention record.")
        for item in graph.mentions:
            span(item.span, item.id, current=True)
            if item.entity_id not in entities:
                errors.append(f"{item.id}: unknown entity {item.entity_id}")
            if item.antecedent_id is not None:
                antecedent = mentions.get(item.antecedent_id)
                if antecedent is None or antecedent.entity_id != item.entity_id or antecedent.id == item.id:
                    errors.append(f"{item.id}: referent anchor must be another available mention of the same entity.")
            if item.reference_status == "first_mention" and item.antecedent_id is not None:
                errors.append(f"{item.id}: first mentions cannot have antecedents.")
            if item.reference_status == "unresolved" and item.antecedent_id is not None:
                errors.append(f"{item.id}: an unresolved reference cannot assert a specific antecedent.")
        for item in graph.events:
            check_contexts(item.context_ids, item.id)
            if graph.schema_version == 2 and (item.status is not None or item.scope_parent_id is not None):
                errors.append(f"{item.id}: schema v2 uses contexts, not legacy status/scope_parent_id.")
            span(item.trigger, item.id + "/trigger", current=True)
            span(item.evidence, item.id + "/evidence", introduces=True)
            if not item.evidence.start <= item.trigger.start < item.trigger.end <= item.evidence.end:
                errors.append(f"{item.id}: event evidence must contain its trigger.")
            if item.scope_parent_id is not None and (item.scope_parent_id not in events or item.scope_parent_id == item.id):
                errors.append(f"{item.id}: unknown or self-referential scope parent.")
            for argument in item.arguments:
                targets = {"entity": entities, "event": events, "literal": literals}[argument.target_kind]
                if argument.target_id not in targets:
                    errors.append(f"{item.id}: unknown argument target {argument.target_id}")
                if argument.mention_id is not None:
                    mention = mentions.get(argument.mention_id)
                    if argument.target_kind != "entity" or mention is None or mention.entity_id != argument.target_id:
                        errors.append(f"{item.id}: argument mention does not ground its entity.")
        for item in graph.relations:
            check_contexts(item.context_ids, item.id)
            span(item.evidence, item.id, introduces=True)
            if item.source_event not in events or item.target_event not in events or item.source_event == item.target_event:
                errors.append(f"{item.id}: relation endpoints must be two known events.")
        for item in graph.entity_updates:
            check_contexts(item.context_ids, item.entity_id)
            span(item.evidence, item.entity_id + "/update", introduces=True)
            if item.entity_id not in entities:
                errors.append(f"Unknown updated entity: {item.entity_id}")
        for item in graph.uncertainties:
            span(item.span, "uncertainty", introduces=True)
        # Only strict temporal-order edges require a DAG. Coreference and other
        # semantic relations are allowed reentrancy and are not forced into trees.
        edges = {}
        for item in [*self.relations.values(), *graph.relations]:
            if item.status == "unresolved" or item.type not in ("before", "after"):
                continue
            source, target = item.source_event, item.target_event
            if item.type == "after":
                source, target = target, source
            # Temporal order is checked within its declared world/attribution.
            # Different hypothetical timelines need not share one ordering.
            world = tuple(sorted(item.context_ids))
            edges.setdefault((world, source), set()).add((world, target))
        visiting, visited = set(), set()

        def visit(node):
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            if not all(visit(target) for target in edges.get(node, ())):
                return False
            visiting.remove(node)
            visited.add(node)
            return True

        if not all(visit(node) for node in edges):
            errors.append("Strict before/after relations contain a cycle.")
        def context_cycle(node, trail):
            if node in trail:
                return True
            return any(context_cycle(parent, trail | {node}) for parent in contexts[node].parent_ids if parent in contexts)
        for item in graph.contexts:
            if context_cycle(item.id, set()):
                errors.append(f"{item.id}: context parent cycle.")
        for item in graph.events:
            seen = {item.id}
            parent = item.scope_parent_id
            while parent is not None and parent in events:
                if parent in seen:
                    errors.append(f"{item.id}: event scope contains a cycle.")
                    break
                seen.add(parent)
                parent = events[parent].scope_parent_id
        if errors:
            raise ValueError("\n".join(errors))

    def append(self, graph: GraphDelta, unit: dict) -> None:
        self.validate(graph, unit)
        for field in ("entities", "mentions", "events", "relations", "literals", "contexts", "identity_links"):
            registry = getattr(self, field)
            for item in getattr(graph, field):
                registry[item.id] = item
                self.available_at[item.id] = unit["end_token"]
        self.updates.extend({**item.model_dump(), "available_at_token": unit["end_token"]} for item in graph.entity_updates)

    def ledger(self) -> dict:
        return {"entities": [item.model_dump() for item in self.entities.values()],
                "mentions": [item.model_dump() for item in self.mentions.values()],
                "events": [item.model_dump() for item in self.events.values()],
                "relations": [item.model_dump() for item in self.relations.values()],
                "literals": [item.model_dump() for item in self.literals.values()],
                "contexts": [item.model_dump() for item in self.contexts.values()],
                "identity_links": [item.model_dump() for item in self.identity_links.values()],
                "entity_updates": self.updates}

    def context_chain(self, ids):
        found = set()
        def visit(identity):
            if identity not in found:
                found.add(identity)
                for parent in self.contexts[identity].parent_ids:
                    visit(parent)
        for identity in ids:
            visit(identity)
        return [self.contexts[c] for c in sorted(found)]

    def event_statuses(self, event):
        kinds = {c.kind for c in self.context_chain(event.context_ids)}
        if event.status is not None:
            kinds.add(event.status)
        return sorted(kinds or {"asserted"})

    def equivalent_entities(self, identity):
        """Only prefix-available, explicit, unscoped identity revelations unify.

        The stored nodes and previous snapshots are never merged or rewritten.
        Reported beliefs and uncertain identities remain separate propositions.
        """
        # A later explicit correction supersedes the same pair's earlier link,
        # while both records remain in the historical ledger.
        latest = {}
        for link in self.identity_links.values():
            if link.status == "explicit" and not link.context_ids:
                latest[frozenset((link.left_entity, link.right_entity))] = link.relation
        result = {identity}
        changed = True
        while changed:
            changed = False
            for pair, relation in latest.items():
                if relation != "same":
                    continue
                if result & pair and not pair <= result:
                    result.update(pair)
                    changed = True
        # Contradictory transitive links do not license extra answer identities.
        if any(relation == "different" and pair <= result for pair, relation in latest.items()):
            return {identity}
        return result
