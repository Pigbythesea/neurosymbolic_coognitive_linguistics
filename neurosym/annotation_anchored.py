"""Exact word quotations compiled to spans, followed by claim-level model review."""

from collections import Counter
from copy import deepcopy
import json

from pydantic import Field

from .annotation import compact_ledger
from .annotation_catalog import (Catalog, Grounding, closed, array, normalize_schema,
                                 selection, BIND_INSTRUCTIONS, BINDING_TRANSPORT)
from .annotation_grounding_checks import validate_grounding_spans
from .graphs import Record, Span


PROTOCOL = "source-anchored-v2"


class Quote(Record):
    quote: str = Field(min_length=1)
    occurrence: int = Field(ge=1)


class SourceText:
    """Only the available prefix can participate in exact quote matching."""

    def __init__(self, story, unit):
        self.story, self.unit = story, unit
        self.words = story["words"][:unit["end_token"]]
        self.tokens = [w["text"] for w in self.words]
        self.positions, self.match_cache = {}, {}
        for i, token in enumerate(self.tokens):
            self.positions.setdefault(token, []).append(i)

    def matches(self, quote, *, current):
        pieces = quote.split()
        if not pieces:
            return []
        key = (tuple(pieces), current)
        if key in self.match_cache:
            return self.match_cache[key]
        first = self.unit["start_token"] if current else 0
        result = [Span(start=i, end=i + len(pieces))
                  for i in self.positions.get(pieces[0], [])
                  if i >= first and self.tokens[i:i + len(pieces)] == pieces]
        self.match_cache[key] = result
        return result

    def resolve(self, value, *, current, path, records):
        anchor = Quote.model_validate(value)
        matches = self.matches(anchor.quote, current=current)
        if anchor.occurrence > len(matches):
            raise ValueError(f"{path}: quote {anchor.quote!r}, occurrence {anchor.occurrence}, "
                             f"has only {len(matches)} exact matches in the {'current unit' if current else 'prefix'}. "
                             "Copy source words exactly, including case and contractions; omit punctuation between words.")
        span = matches[anchor.occurrence - 1]
        if not current and span.end <= self.unit["start_token"]:
            raise ValueError(f"{path}: new evidence must reach the current unit, not only old text.")
        records.append({"path": path, "scope": "current" if current else "prefix",
                        "anchor": anchor.model_dump(), "span": span.model_dump(),
                        "verbatim_excerpt": self.story["text"][self.words[span.start]["char_start"]:
                                                                self.words[span.end - 1]["char_end"]]})
        return span.model_dump()

    def anchor_for_span(self, span, *, current):
        """Lossless source projection for replaying real saved records in checks."""
        span = Span.model_validate(span)
        quote = " ".join(self.tokens[span.start:span.end])
        matches = self.matches(quote, current=current)
        return {"quote": quote, "occurrence": matches.index(span) + 1}

    def excerpt(self, span):
        return " ".join(self.tokens[span["start"]:span["end"]])

    def input(self, history):
        counts, occurrences = Counter(), []
        for token in self.tokens[self.unit["start_token"]:]:
            counts[token] += 1
            occurrences.append({"word": token, "occurrence": counts[token]})
        ledger = compact_ledger(history)
        for mention in ledger["mentions"]:
            mention["quote"] = self.excerpt(mention["span"])
        for event in ledger["events"]:
            event["trigger_quote"] = self.excerpt(event["trigger"])
        return {"story_id": self.unit["story_id"], "unit_id": self.unit["id"],
                "prefix_text": self.story["text"][:self.unit["prefix_char_end"]],
                "current_text": self.story["text"][self.unit["char_start"]:self.unit["prefix_char_end"]],
                "current_word_occurrences": occurrences,
                "prefix_word_tape": " ".join(self.tokens), "prior_ledger": ledger}


def prompt_base(instructions, story, unit, history):
    return instructions + "\nANNOTATION INPUT (quoted data):\n" + json.dumps(
        SourceText(story, unit).input(history), ensure_ascii=False, separators=(",", ":"))


def replace_spans(schema):
    """Every requested model span becomes one exact quote plus occurrence."""
    schema = deepcopy(schema)
    schema["$defs"]["SourceQuote"] = normalize_schema(Quote.model_json_schema())

    def walk(value):
        if isinstance(value, dict):
            if value.get("$ref") in ("#/$defs/Span", "#/$defs/CurrentSpan"):
                value["$ref"] = "#/$defs/SourceQuote"
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    schema["$defs"].pop("Span", None)
    schema["$defs"].pop("CurrentSpan", None)
    return schema


def ground_schema(history):
    schema = replace_spans(normalize_schema(Grounding.model_json_schema()))
    if history.entities:
        schema["$defs"]["ExistingIdentity"]["properties"]["entity"] = selection(len(history.entities))
    else:
        schema["$defs"]["Referent"]["properties"]["identity"] = {"$ref": "#/$defs/NewIdentity"}
        del schema["$defs"]["ExistingIdentity"]
    return schema


def compile_grounding(draft, source, history):
    value, records = deepcopy(draft), []
    for ri, referent in enumerate(value["referents"]):
        for mi, mention in enumerate(referent["mentions"]):
            mention["span"] = source.resolve(mention["span"], current=True,
                path=f"referents/{ri}/mentions/{mi}/span", records=records)
    for ei, event in enumerate(value["events"]):
        for field in ("trigger", "evidence"):
            event[field] = source.resolve(event[field], current=field == "trigger",
                path=f"events/{ei}/{field}", records=records)
        if not event["evidence"]["start"] <= event["trigger"]["start"] < event["trigger"]["end"] <= event["evidence"]["end"]:
            raise ValueError(f"events/{ei}: quoted evidence must contain the quoted trigger; no automatic span expansion.")
    for ui, uncertainty in enumerate(value["uncertainties"]):
        uncertainty["span"] = source.resolve(uncertainty["span"], current=False,
            path=f"uncertainties/{ui}/span", records=records)
    catalog = Catalog(value, history, source.unit)
    spans = Counter((m.span.start, m.span.end) for m in catalog.mentions)
    if any(count > 1 for count in spans.values()):
        raise ValueError("The same mention span is declared more than once. Give each referring expression one identity; record ambiguity explicitly.")
    events = Counter((e.predicate, e.trigger.start, e.trigger.end) for e in catalog.events)
    if any(count > 1 for count in events.values()):
        raise ValueError("Duplicate predicate/trigger records; retain one supported event record per predicate occurrence.")
    validate_grounding_spans(catalog, source.story)
    return catalog, records


def bind_schema(catalog):
    bound = replace_spans(catalog.schema())
    definitions = bound.pop("$defs")
    result = closed({"result": {"anyOf": [closed({"links": bound}),
        closed({"grounding_correction": {"type": "string", "minLength": 1}})]}})
    result["$defs"] = definitions
    return result


def compile_bindings(links, source, catalog):
    value, records = deepcopy(links), []
    for field, span_field in (("relations", "evidence"), ("entity_updates", "evidence"), ("uncertainties", "span")):
        for i, record in enumerate(value[field]):
            record[span_field] = source.resolve(record[span_field], current=False,
                path=f"{field}/{i}/{span_field}", records=records)
    graph = catalog.compile(value)
    return graph, records


def surface_graph(graph, source):
    value = graph.model_dump()
    for field in ("entities", "mentions", "events", "relations", "entity_updates", "uncertainties"):
        for item in value[field]:
            item.pop("confidence", None)
            for key in ("span", "first_mention", "trigger", "evidence"):
                if key in item:
                    item[key + "_quote"] = source.excerpt(item[key])
    return value


def review_items(graph):
    grounding = {"coverage": "Check omitted mentions, predicates/states, and spurious annotations in the current unit."}
    bindings = {}
    for entity in graph.entities:
        grounding[entity.id] = "Check entity concept/type and distinctness from known referents."
    for mention in graph.mentions:
        grounding[mention.id] = "Check that the quoted expression really is a mention of this entity and has the stated form."
        bindings[mention.id] = "Check identity/coreference and antecedent against the prefix, including unresolved ambiguity."
    for event in graph.events:
        grounding[event.id] = "Check predicate, quoted trigger/evidence, polarity, status, and tense; do not infer actuality."
        bindings[event.id] = "Check scope and missing/unsupported roles, including argument-free events."
        for i, _ in enumerate(event.arguments):
            bindings[f"{event.id}/argument/{i}"] = "Check this role and its exact target; grammatical subject is not necessarily agent."
    for relation in graph.relations:
        bindings[relation.id] = "Check direction, type, evidence, and whether the relationship is asserted/inferred/uncertain."
    for i, _ in enumerate(graph.entity_updates):
        bindings[f"update/{i}"] = "Check that the update applies to this entity and is actually asserted, not desired or hypothetical."
    return {"grounding": grounding, "bindings": bindings}


def review_schema(graph):
    decision = closed({"verdict": {"type": "string", "enum": ["supported", "needs_correction"]},
                       "evidence": {"$ref": "#/$defs/SourceQuote"},
                       "reason": {"type": "string"}})
    schema = closed({family: closed({key: {"$ref": "#/$defs/ReviewDecision"} for key in items})
                     for family, items in review_items(graph).items()})
    schema["$defs"] = {"SourceQuote": normalize_schema(Quote.model_json_schema()), "ReviewDecision": decision}
    return schema


def validate_review(response, source, graph):
    from .annotation_catalog import exact_keys
    expected, records, failures = review_items(graph), [], []
    exact_keys(response, expected, "semantic review families")
    for family, items in expected.items():
        exact_keys(response[family], items, "semantic review " + family)
        for key in items:
            decision = response[family][key]
            exact_keys(decision, ("verdict", "evidence", "reason"), "review decision " + key)
            if decision["verdict"] not in ("supported", "needs_correction") or not isinstance(decision["reason"], str):
                raise ValueError("Invalid semantic review decision: " + key)
            # Review evidence may refer solely to the earlier prefix (e.g. a
            # coreference antecedent), unlike a newly asserted graph relation.
            anchor = Quote.model_validate(decision["evidence"])
            matches = source.matches(anchor.quote, current=False)
            if anchor.occurrence > len(matches):
                raise ValueError("Semantic review evidence is not an exact prefix quote: " + key)
            records.append({"claim": key, "anchor": anchor.model_dump(),
                            "span": matches[anchor.occurrence - 1].model_dump()})
            if decision["verdict"] == "needs_correction":
                if not decision["reason"].strip():
                    raise ValueError("A failed semantic review needs an explanation: " + key)
                failures.append({"family": family, "claim": key, "reason": decision["reason"],
                                 "evidence": anchor.model_dump()})
    return failures, records


GROUND = """
STAGE GROUND: return referents, events, uncertainties in the supplied schema.
Group current mentions by their actual referent, not by word proximity. Use an
existing entity index when identity is resolved, or a new entity description.
Do not create a mention for every word or for an unspoken participant. Entity
mentions are referring expressions; event triggers are predicates, not mentions
of their agent. Exact duplicate mention spans are forbidden. Distinct referents
remain distinct even if they share a concept. Record unresolved reference honestly.

Every span/trigger/evidence is {quote, occurrence}; NEVER generate token indices.
Copy a contiguous sequence from the word tape, with spaces between words. Copy
case and contractions exactly. Omit punctuation and transcription markers between
words. occurrence is ONE-BASED among exact matches: CURRENT UNIT for mention spans
and triggers; AVAILABLE PREFIX for event evidence and uncertainty spans. The
current_word_occurrences table identifies repeated single words in the unit.
Evidence must reach this unit, and event evidence must contain its trigger.
Put the introducing mention first for a new entity. Leave all relationships for
BIND. Empty categories are allowed only when no text-supported items are present.
"""


def grounding_prompt(base, history, feedback=""):
    entities = [{"index": i, **e.model_dump()} for i, e in enumerate(history.entities.values())]
    return base + "\n" + GROUND + "\nEXISTING ENTITIES:\n" + json.dumps(entities, ensure_ascii=False, separators=(",", ":")) + (
        "\nCORRECT THE FOLLOWING SOURCE/GROUNDING ERRORS:\n" + feedback if feedback else "")


def binding_prompt(base, catalog, source, feedback=""):
    instructions = BIND_INSTRUCTIONS.replace("TRANSPORT PROTOCOL ground-and-bind-v1", "TRANSPORT PROTOCOL " + PROTOCOL)
    instructions = instructions.replace("correction_required", "grounding_correction")
    payload = catalog.payload()
    payload["compiled_nodes"] = surface_graph(catalog.empty_graph(), source)
    return (base + "\n" + instructions + "\nBINDING TRANSPORT " + BINDING_TRANSPORT +
        "\nAll relation/update evidence and uncertainty spans use {quote, occurrence}, with ONE-BASED occurrence in the prefix word tape. "
        "Do not output integer token spans. Evidence must reach the current unit. "
        "grounding_correction is ONLY for a specific semantic/node defect; an invalid link must be fixed in links. "
        "If no correction is needed, return links, never a correction message saying everything is correct. "
        "Empty current-node lists are permitted for units without new mentions/events; do not invent missing catalogs.\nBINDING CATALOG:\n" +
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) +
        ("\nCORRECT THESE LINK ERRORS:\n" + feedback if feedback else ""))


def semantic_review_prompt(base, source, graph, feedback=""):
    return (base + "\nSTAGE REVIEW: audit the proposed annotation against the SOURCE, without trusting the generator. "
        "Return a decision for EVERY listed claim. Check actual quoted expressions, referent identity, event meaning, "
        "roles, scope, factuality, discourse direction, and missing/spurious items. Evidence must be an exact word-tape "
        "quote with ONE-BASED occurrence in the available prefix. Prefer unique multiword excerpts to avoid counting "
        "repeated pronouns in the prefix. Each decision must have source evidence; a failed "
        "decision also needs a concrete correction reason. The source may express genuine ambiguity: accepting an "
        "explicitly unresolved label is allowed, accepting a guessed confident referent is not. Check semantic "
        "content, not just schema validity. Confidence values are not supplied as evidence. For coverage, cite the "
        "current expression you checked. Supported means no defect identified by this review, not human gold.\n" +
        json.dumps({"claims": review_items(graph), "proposed_graph": surface_graph(graph, source)},
                   ensure_ascii=False, separators=(",", ":")) +
        ("\nFIX THESE REVIEW-FORMAT ERRORS:\n" + feedback if feedback else ""))
