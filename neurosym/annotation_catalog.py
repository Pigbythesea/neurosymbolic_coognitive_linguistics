"""Ground nodes first, then bind semantic links to an enumerated, immutable catalog.

The model never invents graph IDs or repeats an argument's entity/type/mention
identity. The compiled output is still GraphDelta v1 for downstream consumers.
"""

from copy import deepcopy
import json
from typing import Literal

from pydantic import Field

from .graphs import (Record, Span, Entity, Mention, Event, Argument, Relation,
                     EntityUpdate, Uncertainty, GraphDelta, GraphHistory, graph_schema)


PROTOCOL = "ground-and-bind-v1"
BINDING_TRANSPORT = "shared-catalog-v2"


class EntityDescription(Record):
    label: str
    concept: str = Field(pattern=r"^[a-z0-9][a-z0-9_'-]*$")
    kind: Entity.model_fields["kind"].annotation


class NewIdentity(Record):
    type: Literal["new"]
    entity: EntityDescription


class ExistingIdentity(Record):
    type: Literal["existing"]
    entity: int = Field(ge=0)


class GroundMention(Record):
    span: Span
    form: Mention.model_fields["form"].annotation
    confidence: float = Field(ge=0, le=1)


class Referent(Record):
    identity: NewIdentity | ExistingIdentity
    mentions: list[GroundMention] = Field(min_length=1)


class GroundEvent(Record):
    predicate: str = Field(pattern=r"^[a-z0-9][a-z0-9_'-]*$")
    sense_id: str | None
    trigger: Span
    evidence: Span
    polarity: Event.model_fields["polarity"].annotation
    status: Event.model_fields["status"].annotation
    tense: Event.model_fields["tense"].annotation
    confidence: float = Field(ge=0, le=1)


class Grounding(Record):
    referents: list[Referent]
    events: list[GroundEvent]
    uncertainties: list[Uncertainty]


def closed(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


def selection(count, *, nullable=False):
    if count == 0:
        if nullable:
            return {"type": "null"}
        raise ValueError("Cannot select from an empty catalog.")
    schema = {"type": "integer", "minimum": 0, "maximum": count - 1}
    return {"anyOf": [schema, {"type": "null"}]} if nullable else schema


def selection_except(count, excluded, *, nullable=False):
    """Select a shared catalog index without requiring a copied filtered list."""
    if not 0 <= excluded < count:
        raise ValueError("Excluded catalog index is not available.")
    choices = []
    if excluded:
        choices.append({"type": "integer", "minimum": 0, "maximum": excluded - 1})
    if excluded + 1 < count:
        choices.append({"type": "integer", "minimum": excluded + 1, "maximum": count - 1})
    if nullable:
        choices.append({"type": "null"})
    if not choices:
        raise ValueError("No catalog selections are available.")
    return choices[0] if len(choices) == 1 else {"anyOf": choices}


def normalize_schema(value):
    if isinstance(value, dict):
        if "const" in value:
            value["enum"] = [value.pop("const")]
        for child in value.values():
            normalize_schema(child)
    elif isinstance(value, list):
        for child in value:
            normalize_schema(child)
    return value


def span_bounds(schema, unit, *, current=False):
    schema["properties"]["start"].update(minimum=unit["start_token"] if current else 0,
                                          maximum=unit["end_token"] - 1)
    schema["properties"]["end"].update(minimum=unit["start_token"] + 1, maximum=unit["end_token"])


def grounding_schema(history, unit):
    schema = normalize_schema(Grounding.model_json_schema())
    span_bounds(schema["$defs"]["Span"], unit)
    current = deepcopy(schema["$defs"]["Span"])
    span_bounds(current, unit, current=True)
    schema["$defs"]["CurrentSpan"] = current
    schema["$defs"]["GroundMention"]["properties"]["span"] = {"$ref": "#/$defs/CurrentSpan"}
    schema["$defs"]["GroundEvent"]["properties"]["trigger"] = {"$ref": "#/$defs/CurrentSpan"}
    identity = schema["$defs"]["Referent"]["properties"]["identity"]
    if history.entities:
        schema["$defs"]["ExistingIdentity"]["properties"]["entity"] = selection(len(history.entities))
    else:
        identity.clear()
        identity["$ref"] = "#/$defs/NewIdentity"
        del schema["$defs"]["ExistingIdentity"]
    return schema


def choose(items, index, label):
    if type(index) is not int or not 0 <= index < len(items):
        raise ValueError(f"{label}: select an integer in 0..{len(items) - 1}.")
    return items[index]


def exact_keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label}: expected exactly the fields {list(names)}.")


class Catalog:
    def __init__(self, grounding, history: GraphHistory, unit):
        self.grounding = Grounding.model_validate(grounding)
        self.history, self.unit = history, unit
        self.entities, self.mentions, self.events, self.derivations = [], [], [], []
        prior_entities = list(history.entities)
        prefix = unit["id"]
        for referent in self.grounding.referents:
            if isinstance(referent.identity, NewIdentity):
                identity = f"{prefix}_n{len(self.entities) + 1}"
                # The first listed mention is the model's introducing occurrence;
                # its span has a single authoritative representation.
                self.entities.append(Entity(id=identity, **referent.identity.entity.model_dump(),
                                            first_mention=referent.mentions[0].span))
            else:
                identity = choose(prior_entities, referent.identity.entity, "existing entity")
            for mention in referent.mentions:
                self.mentions.append(Mention(id=f"{prefix}_m{len(self.mentions) + 1}", entity_id=identity,
                                             **mention.model_dump(), antecedent_id=None,
                                             reference_status="explicit"))
        for item in self.grounding.events:
            event = Event(id=f"{prefix}_e{len(self.events) + 1}", **item.model_dump(),
                          arguments=[], scope_parent_id=None)
            # Evidence includes the explicitly supplied trigger by definition.
            # Never infer a trigger or change its token coordinates.
            if (0 <= event.evidence.start < event.evidence.end <= unit["end_token"]
                    and unit["start_token"] <= event.trigger.start < event.trigger.end <= unit["end_token"]
                    and not event.evidence.start <= event.trigger.start < event.trigger.end <= event.evidence.end):
                before = event.evidence.model_dump()
                event.evidence = Span(start=min(event.evidence.start, event.trigger.start),
                                      end=max(event.evidence.end, event.trigger.end))
                self.derivations.append({"kind": "evidence_includes_declared_trigger", "id": event.id,
                                         "before": before, "after": event.evidence.model_dump()})
            self.events.append(event)
        self.entity_map = {**history.entities, **{v.id: v for v in self.entities}}
        self.mention_map = {**history.mentions, **{v.id: v for v in self.mentions}}
        self.event_map = {**history.events, **{v.id: v for v in self.events}}
        # Share each entity's complete mention pool and the complete event pool.
        # Self is excluded by the request schema and compiler, not by copying
        # every list with one element removed. Pool indices never shift per row.
        self.anchor_pools = {}
        for mention in self.mention_map.values():
            self.anchor_pools.setdefault(mention.entity_id, []).append(mention.id)
        self.anchors = {m.id: self.anchor_pools[m.entity_id] for m in self.mentions}
        self.event_ids = list(self.event_map)
        self.parents = {e.id: self.event_ids for e in self.events}
        self.targets = ([{"kind": "mention", "id": v} for v in self.mention_map]
                        + [{"kind": "entity", "id": v} for v in self.entity_map]
                        + [{"kind": "event", "id": v} for v in self.event_map])
        history.validate(self.empty_graph(), unit)

    def empty_graph(self):
        return GraphDelta(schema_version=1, story_id=self.unit["story_id"], unit_id=self.unit["id"],
                          entities=deepcopy(self.entities), mentions=deepcopy(self.mentions),
                          events=deepcopy(self.events), relations=[], entity_updates=[],
                          uncertainties=deepcopy(self.grounding.uncertainties))

    def reference_choices(self, mention):
        # Entity kind and discourse reference status are distinct: two mentions
        # can refer to the same as-yet unidentified person, for example.
        # Preserve the GraphDelta status vocabulary; this protocol changes
        # identity transport, not interpretation of already accepted labels.
        return ["explicit", "inferred", "unresolved", "first_mention"]

    def schema(self):
        defs = deepcopy(graph_schema()["$defs"])
        span_bounds(defs["Span"], self.unit)
        references = {}
        for mention in self.mentions:
            variants = []
            statuses = self.reference_choices(mention)
            resolvable = [v for v in statuses if v in ("explicit", "inferred")]
            if resolvable:
                variants.append(closed({"status": {"type": "string", "enum": resolvable},
                                        "anchor": selection_except(len(self.anchors[mention.id]),
                                            self.anchors[mention.id].index(mention.id), nullable=True)}))
            for status in ("first_mention", "unresolved"):
                if status in statuses:
                    variants.append(closed({"status": {"type": "string", "enum": [status]},
                                            "anchor": {"type": "null"}}))
            references[mention.id] = variants[0] if len(variants) == 1 else {"anyOf": variants}
        argument = deepcopy(defs["Argument"]["properties"])
        for key in ("target_kind", "target_id", "mention_id"):
            argument.pop(key)
        # Any event has at least itself in the target catalog; an empty event
        # list never uses this definition.
        argument["target"] = selection(len(self.targets)) if self.targets else {"type": "null"}
        defs["BoundArgument"] = closed(argument)
        events = {e.id: closed({"arguments": array({"$ref": "#/$defs/BoundArgument"}),
                                "scope_parent": selection_except(len(self.event_ids),
                                    self.event_ids.index(e.id), nullable=True)})
                  for e in self.events}
        relation = deepcopy(defs["Relation"]["properties"])
        relation.pop("id")
        for field in ("source_event", "target_event"):
            relation[field] = selection(len(self.event_map)) if self.event_map else {"type": "null"}
        update = deepcopy(defs["EntityUpdate"]["properties"])
        update.pop("entity_id")
        update["entity"] = selection(len(self.entity_map)) if self.entity_map else {"type": "null"}
        relations = array(closed(relation))
        updates = array(closed(update))
        if len(self.event_map) < 2:
            relations["maxItems"] = 0
        if not self.entity_map:
            updates["maxItems"] = 0
        result = closed({"references": closed(references), "events": closed(events),
                         "relations": relations, "entity_updates": updates,
                         "uncertainties": array({"$ref": "#/$defs/Uncertainty"})})
        # Only carry definitions reachable from this request's actual schema.
        needed = set()

        def walk(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    name = value["$ref"].split("/")[-1]
                    if name not in needed:
                        needed.add(name)
                        walk(defs[name])
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(result)
        result["$defs"] = {name: defs[name] for name in sorted(needed)}
        return result

    def payload(self):
        return {"compiled_nodes": {"entities": [v.model_dump() for v in self.entities],
                                   "mentions": [v.model_dump() for v in self.mentions],
                                   "events": [v.model_dump() for v in self.events]},
                "argument_targets": [{"index": i, **v} for i, v in enumerate(self.targets)],
                "reference_anchor_pools": {
                    entity: values for entity, values in self.anchor_pools.items()
                    if any(m.entity_id == entity for m in self.mentions)},
                "reference_anchors": {m.id: {"pool": m.entity_id,
                    "excluded_self_index": self.anchors[m.id].index(m.id)} for m in self.mentions},
                "scope_parents": {e.id: {"pool": "relation_events",
                    "excluded_self_index": self.event_ids.index(e.id)} for e in self.events},
                "relation_events": self.event_ids,
                "update_entities": [{"index": i, "entity_id": v} for i, v in enumerate(self.entity_map)]}

    def compile(self, response):
        exact_keys(response, ("references", "events", "relations", "entity_updates", "uncertainties"), "binding")
        exact_keys(response["references"], [m.id for m in self.mentions], "references")
        exact_keys(response["events"], [e.id for e in self.events], "events")
        graph = self.empty_graph()
        for mention in graph.mentions:
            decision = response["references"][mention.id]
            exact_keys(decision, ("status", "anchor"), mention.id)
            if decision["status"] not in self.reference_choices(mention):
                raise ValueError(mention.id + ": incompatible reference status.")
            mention.reference_status = decision["status"]
            mention.antecedent_id = (None if decision["anchor"] is None else
                                    choose(self.anchors[mention.id], decision["anchor"], "antecedent"))
        for event in graph.events:
            decision = response["events"][event.id]
            exact_keys(decision, ("arguments", "scope_parent"), event.id)
            event.scope_parent_id = (None if decision["scope_parent"] is None else
                                     choose(self.parents[event.id], decision["scope_parent"], "scope parent"))
            for argument in decision["arguments"]:
                exact_keys(argument, ("role", "frame_role", "target"), "argument")
                target = choose(self.targets, argument["target"], "argument target")
                mention = self.mention_map[target["id"]] if target["kind"] == "mention" else None
                event.arguments.append(Argument(role=argument["role"], frame_role=argument["frame_role"],
                    target_kind="event" if target["kind"] == "event" else "entity",
                    target_id=mention.entity_id if mention else target["id"],
                    mention_id=mention.id if mention else None))
        for i, relation in enumerate(response["relations"]):
            relation = dict(relation)
            for key in ("source_event", "target_event"):
                relation[key] = choose(list(self.event_map), relation[key], key)
            graph.relations.append(Relation(id=f"{self.unit['id']}_r{i + 1}", **relation))
        for update in response["entity_updates"]:
            update = dict(update)
            entity = choose(list(self.entity_map), update.pop("entity"), "updated entity")
            graph.entity_updates.append(EntityUpdate(entity_id=entity, **update))
        graph.uncertainties.extend(Uncertainty.model_validate(v) for v in response["uncertainties"])
        # Revalidate model assignments as well as the cross-record constraints.
        graph = GraphDelta.model_validate(graph.model_dump())
        self.history.validate(graph, self.unit)
        return graph


GROUND_INSTRUCTIONS = """
TRANSPORT PROTOCOL ground-and-bind-v1, stage GROUND. This replaces the output
format/ID bookkeeping instructions above, not the semantic annotation rules.
Return ONLY referents, events, uncertainties using the supplied schema.
Do not emit IDs, arguments, antecedents, relations, updates, or scope links yet.

Group all CURRENT mentions of the SAME entity in one referent row. Every row must
contain at least one actual mention. Different individuals require different rows.
For an already known referent use identity {type:existing, entity:INDEX} from
EXISTING ENTITIES below. Do not redeclare it as new. For a newly expressed entity,
use identity {type:new, entity:{label,concept,kind}}. List its introducing mention
first. Grouping is the authoritative coreference decision. If genuinely unresolved,
use a separate new entity of kind unresolved and record the ambiguity; do not bind
it to a guessed known individual. Keep generic kinds and particular instances distinct.

Include mentions needed by event arguments (including manner, location, time, etc.)
when supported by text, and all expressed predicates/states including embedded ones.
An argument can later select a declared mention, entity, or event, but cannot create
one in the binding step. Do not omit text-supported nodes in anticipation of links.
Do not create unspoken mentions for implicit participants. Later binding may use a
known entity directly when no mention grounds it. Spans use the original token indices.
The code will assign IDs and first_mention from these records. New mentions and
triggers must be within the CURRENT unit. Event evidence includes the trigger.
"""

BIND_INSTRUCTIONS = """
TRANSPORT PROTOCOL ground-and-bind-v1, stage BIND. This replaces the output
format/ID bookkeeping instructions above, not the semantic annotation rules.
The grounded nodes below are fixed. Return semantic links using the provided
catalog indices, never invent IDs or nodes. Every schema property naming a current
mention or event needs a decision, including empty arguments when appropriate.

For each argument choose ONE target index. Prefer a mention when text grounds the
argument; the compiler derives its entity and type. Use a direct entity only for
an implicit/unexpressed participant, or an event for propositional/event content.
For each reference choose status and an anchor index from its shared entity pool
in reference_anchor_pools, or null when no supported anchor. reference_anchors
names that mention's pool and excluded_self_index. Array positions are ZERO-BASED
indices. Do not select the excluded index; do not remove that entry or renumber
the pool. first_mention and unresolved require null. Anchors may be cataphoric
within the available prefix. Scope parents and relation endpoints BOTH use the
single relation_events array; a scope parent cannot select its own event index.
Entity updates use update_entities. These catalogs have distinct index spaces.

All substantive choices remain yours: roles, coreference anchors, factuality,
scope, chronology, causation, and updates must be supported by the supplied prefix.
Do not assert chronological order from narrative order. Do not invent missing
nodes or force ambiguous links. Record genuine uncertainty. Preserve all grounded
events and mentions; do not delete unsupported-looking nodes through a link edit.
If grounding itself is wrong, return the correction_required alternative with
specific text-grounded feedback; the grounding stage will be redone explicitly.
"""


def ground_prompt(base_prompt, history, feedback=""):
    catalog = [{"index": i, **v.model_dump()} for i, v in enumerate(history.entities.values())]
    return (base_prompt + "\n\n" + GROUND_INSTRUCTIONS + "\nEXISTING ENTITIES:\n"
            + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
            + ("\nCORRECT THIS GROUNDING ERROR:\n" + feedback if feedback else ""))


def binding_schema(catalog):
    # A semantic omission must go back to grounding, never be silently dropped
    # or represented as a fabricated entity just to pass validation.
    bound = catalog.schema()
    definitions = bound.pop("$defs")
    answer = closed({"links": bound})
    correction = closed({"correction_required": {"type": "string", "minLength": 1}})
    result = closed({"result": {"anyOf": [answer, correction]}})
    result["$defs"] = definitions
    return result


def bind_prompt(base_prompt, catalog, feedback=""):
    return (base_prompt + "\n\nBINDING TRANSPORT " + BINDING_TRANSPORT + "\n"
            + BIND_INSTRUCTIONS + "\nBINDING CATALOG:\n"
            + json.dumps(catalog.payload(), ensure_ascii=False, separators=(",", ":"))
            + ("\nCORRECT THESE LINK ERRORS:\n" + feedback if feedback else ""))
