"""Directly authored, evidence-linked annotations; no model or analysis calls.

The author chooses all semantic records. This module only locates quotations,
checks structure/availability, checkpoints them, and exports unchanged labels.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import re

from pydantic import BaseModel, ConfigDict, Field

from .corpus import transcript_tokens
from .io import object_hash, read_json, save_json, immutable_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/annotations/deniz-direct-v2"
CORPUS = ROOT / "data/processed/corpus"
GUIDE = ROOT / "docs/direct_annotation_conventions.md"
PROTOCOL = "direct-semantic-graph-v2"
COLLECTIONS = ("entities", "mentions", "literals", "contexts", "events", "relations",
               "properties", "qualifiers", "identity_links", "uncertainties")
NODE_COLLECTIONS = ("entities", "mentions", "literals", "contexts", "events")


def now():
    return datetime.now(timezone.utc).isoformat()


def log_event(sid, event):
    """Per-story append-only operational trace; never part of semantic labels."""
    path = OUTPUT / "traces" / (sid + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"utc": now(), **event}, ensure_ascii=False, sort_keys=True)+"\n")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(Strict):
    quote: str = Field(min_length=1)
    start: int | None = None
    within: str | None = None


class Entity(Strict):
    id: str
    label: str
    concept: str
    kind: str
    denotation: str
    introduction: str
    evidence: Evidence
    sense: str | None = None


class Mention(Strict):
    id: str
    target: str | None
    form: str
    support: str
    evidence: Evidence
    antecedent: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class Literal(Strict):
    id: str
    kind: str
    value: str | int | float | bool
    evidence: Evidence
    unit: str | None = None


class Context(Strict):
    id: str
    kind: str
    evidence: Evidence
    holder: str | None = None
    attribution: str | None = None
    parents: list[str] = Field(default_factory=list)
    support: str


class Argument(Strict):
    role: str
    target: str | None
    support: str
    mention: str | None = None


class Event(Strict):
    id: str
    predicate: str
    trigger: Evidence
    evidence: Evidence
    arguments: list[Argument]
    polarity: str
    tense: str
    mode: str
    contexts: list[str]
    support: str
    sense: str | None = None


class Relation(Strict):
    id: str
    type: str
    source: str
    target: str
    evidence: Evidence
    contexts: list[str]
    support: str


class Property(Strict):
    id: str
    target: str
    attribute: str
    value: str | int | float | bool | dict
    operation: str
    evidence: Evidence
    contexts: list[str]
    support: str


class Qualifier(Strict):
    id: str
    target: str
    dimension: str
    value: str | int | float | bool | dict
    evidence: Evidence
    contexts: list[str]
    support: str


class Identity(Strict):
    id: str
    left: str
    right: str
    relation: str
    evidence: Evidence
    contexts: list[str]
    support: str


class Uncertainty(Strict):
    id: str
    category: str
    targets: list[str]
    resolution: str
    explanation: str
    evidence: Evidence
    alternatives: list[dict] = Field(default_factory=list)


class Draft(Strict):
    protocol: str
    story_id: str
    unit_id: str
    summary: str
    coverage: str
    entities: list[Entity] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    literals: list[Literal] = Field(default_factory=list)
    contexts: list[Context] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    properties: list[Property] = Field(default_factory=list)
    qualifiers: list[Qualifier] = Field(default_factory=list)
    identity_links: list[Identity] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)


def corpus_index():
    index = read_json(CORPUS / "index.json")
    if object_hash({k: v for k, v in index.items() if k != "content_hash"}) != index["content_hash"]:
        raise ValueError("Corpus index checksum mismatch")
    return index


def story_source(sid):
    index = corpus_index()
    entry = next(s for s in index["stories"] if s["id"] == sid)
    story = read_json(CORPUS / entry["path"])
    if story["content_hash"] != entry["content_hash"] or object_hash(
            {k: v for k, v in story.items() if k != "content_hash"}) != entry["content_hash"]:
        raise ValueError("Story checksum mismatch")
    return entry, story


def initialize():
    index = corpus_index()
    manifest = {
        "protocol": PROTOCOL, "role": "production annotation attempt; human review pending",
        "corpus_hash": index["content_hash"], "corpus": "data/processed/corpus",
        "conventions_sha256": hashlib.sha256(GUIDE.read_bytes()).hexdigest(),
        "authoring": "Direct semantic decisions by isolated Codex agents; no CLI model runner",
        "model_provenance": {"family_reported_by_session": "GPT-6-based Codex",
                             "model_id": None, "reasoning_setting": None,
                             "visibility": "Exact model revision/settings not exposed to the annotation code; never inferred from filenames"},
        "availability": "Commit current unit before exposing the next unit; preserve earlier assertions",
        "development_evidence": "deniz-direct-v1 is nonblind diagnostic material, not imported production labels",
        "heldout": [s["id"] for s in index["stories"] if s["split"] == "heldout"],
        "scientific_independence": "No brain measurements or evaluated-model scores used for label decisions",
        "review_status": "unreviewed_by_humans",
    }
    immutable_json(OUTPUT / "manifest.json", manifest)
    immutable_json(OUTPUT / "schema.json", Draft.model_json_schema())
    return manifest


def records(sid):
    _, story = story_source(sid)
    result = []
    parent = None
    directory = OUTPUT / "records" / sid
    captured_names = {p.name for p in directory.glob("*.json")} if directory.exists() else set()
    for unit in story["units"]:
        p = OUTPUT / "records" / sid / (unit["id"] + ".json")
        if p.name not in captured_names:
            break
        record = read_json(p)
        if record["parent_graph_hash"] != parent or object_hash(record["graph"]) != record["graph_hash"]:
            raise ValueError("Broken annotation hash chain: " + unit["id"])
        result.append(record)
        parent = record["graph_hash"]
    if len(captured_names) != len(result):
        raise ValueError("Annotation gap detected: " + sid)
    return result


def register(sid, actor, exposure="fresh_story_context"):
    entry, _ = story_source(sid)
    if exposure != "fresh_story_context":
        raise ValueError("Production authoring requires a fresh story context")
    value = {"story_id": sid, "actor": actor, "exposure": exposure,
             "declaration": "This actor has not received later passages or pre-existing graphs for this story",
             "story_hash": entry["content_hash"], "conventions_sha256": initialize()["conventions_sha256"]}
    immutable_json(OUTPUT / "authors" / (sid + ".json"), value)
    return value


def next_packet(sid, actor):
    manifest = initialize()
    auth = read_json(OUTPUT / "authors" / (sid + ".json"))
    if auth["actor"] != actor:
        raise ValueError("Wrong registered story actor")
    entry, story = story_source(sid)
    prior = records(sid)
    if len(prior) == len(story["units"]):
        return {"story_id": sid, "complete": True, "units": len(prior)}
    unit = story["units"][len(prior)]
    value = {"protocol": PROTOCOL, "actor": actor, "story_id": sid, "unit": unit,
             "story_hash": entry["content_hash"], "conventions_sha256": manifest["conventions_sha256"],
             "parent_graph_hash": prior[-1]["graph_hash"] if prior else None,
             "current_text": story["text"][unit["char_start"]:unit["prefix_char_end"]],
             "current_words": [{"index": w["index"], "text": w["text"]}
                               for w in story["words"][unit["start_token"]:unit["end_token"]]],
             "prior_committed_units": len(prior),
             "information_condition": "Only this unit and earlier accepted text/annotations are available"}
    immutable_json(OUTPUT / "requests" / sid / (unit["id"] + ".json"), value)
    log_event(sid, {"operation": "expose_current_unit", "actor": actor, "unit_id": unit["id"],
                    "request_hash": object_hash(value), "parent_graph_hash": value["parent_graph_hash"]})
    return value


def available_ledger(sid):
    prior = records(sid)
    _, story = story_source(sid)
    end = story["units"][len(prior)-1]["prefix_char_end"] if prior else 0
    return {"story_id": sid, "available_text": story["text"][:end],
            "committed_units": len(prior), "graph_deltas": [r["graph"] for r in prior]}


def anchor(value, story, unit, *, current=False):
    value = Evidence.model_validate(value).model_dump(exclude_none=True)
    tokens = [w["text"] for w in story["words"][:unit["end_token"]]]
    pieces = [w["text"] for w in transcript_tokens(value["quote"])]
    if not pieces:
        raise ValueError("Empty quotation after transcript tokenization")
    first = unit["start_token"] if current else 0
    matches = [i for i in range(first, len(tokens)-len(pieces)+1)
               if tokens[i:i+len(pieces)] == pieces and i+len(pieces)>unit["start_token"]]
    if "within" in value:
        surroundings = [w["text"] for w in transcript_tokens(value["within"])]
        windows = [i for i in range(first, len(tokens)-len(surroundings)+1)
                   if tokens[i:i+len(surroundings)] == surroundings]
        matches = [i for i in matches if any(j <= i and i+len(pieces) <= j+len(surroundings) for j in windows)]
    if "start" in value:
        matches = [i for i in matches if i == value["start"]]
    if len(matches) != 1:
        raise ValueError(f"Quote {value['quote']!r} has {len(matches)} matches; select a start index from {matches}")
    start, end = matches[0], matches[0]+len(pieces)
    if end <= unit["start_token"]:
        raise ValueError("New observation evidence must reach the current unit")
    a, b = story["words"][start]["char_start"], story["words"][end-1]["char_end"]
    return {**value, "start_token": start, "end_token": end, "char_start": a,
            "char_end": b, "verbatim": story["text"][a:b]}


def validate_graph(raw, story, unit, prior):
    graph = Draft.model_validate(raw).model_dump()
    if (graph["protocol"], graph["story_id"], graph["unit_id"]) != (PROTOCOL, story["story_id"], unit["id"]):
        raise ValueError("Draft identity does not match the exposed passage")
    if graph["coverage"] not in {"annotated", "nonpropositional", "fragment"} or not graph["summary"].strip():
        raise ValueError("Supply an explicit coverage class and semantic summary")
    known = {item["id"]: family for r in prior for family in COLLECTIONS for item in r["graph"][family]}
    current = {}
    for family in COLLECTIONS:
        for item in graph[family]:
            ident = item["id"]
            if not re.fullmatch(re.escape(story["story_id"])+r":[a-zA-Z0-9_:-]+", ident):
                raise ValueError("IDs must be unique story-local keys: " + ident)
            if ident in known or ident in current:
                raise ValueError("ID already introduced: " + ident)
            current[ident] = family
    kinds = {**known, **current}
    def ref(ident, permitted=None):
        if ident is not None and (ident not in kinds or permitted is not None and kinds[ident] not in permitted):
            raise ValueError("Unavailable/wrong-kind reference: " + str(ident))
    def contexts(item):
        for c in item.get("contexts", []): ref(c, {"contexts"})
    def support(item):
        if "support" in item and item["support"] not in {"explicit", "inferred", "unresolved"}:
            raise ValueError("Invalid support class")
    def value_refs(value):
        if isinstance(value, dict):
            if "ref" in value: ref(value["ref"])
            for v in value.values(): value_refs(v)
        elif isinstance(value, list):
            for v in value: value_refs(v)
    for family in COLLECTIONS:
        for item in graph[family]:
            support(item); contexts(item)
            same_trigger_evidence = family == "events" and item["evidence"] == item["trigger"]
            item["evidence"] = anchor(item["evidence"], story, unit,
                                      current=family in {"entities", "mentions", "literals"} or same_trigger_evidence)
            if family == "entities":
                if item["denotation"] not in {"individual", "collective", "generic", "kind", "unspecified"}:
                    raise ValueError("Invalid denotation")
                if item["introduction"] not in {"explicit", "implicit", "unresolved"}:
                    raise ValueError("Invalid entity introduction")
                if not re.fullmatch(r"[a-z0-9][a-z0-9_'-]*", item["concept"]):
                    raise ValueError("Concept labels must be normalized keys, not descriptions")
            elif family == "mentions":
                ref(item["target"], {"entities", "events", "literals"})
                ref(item["antecedent"], {"mentions", "events", "entities"})
                for alt in item["alternatives"]: ref(alt, {"entities", "events", "literals"})
                if item["target"] is None and item["support"] != "unresolved":
                    raise ValueError("A null mention target must be unresolved")
            elif family == "contexts":
                ref(item["holder"], {"entities"}); ref(item["attribution"], {"events"})
                for p in item["parents"]: ref(p, {"contexts"})
            elif family == "events":
                item["trigger"] = anchor(item["trigger"], story, unit, current=True)
                if not (item["evidence"]["start_token"] <= item["trigger"]["start_token"] <
                        item["trigger"]["end_token"] <= item["evidence"]["end_token"]):
                    raise ValueError("Event evidence must contain its trigger")
                if item["polarity"] not in {"positive", "negative", "unresolved"} or item["tense"] not in {"past", "present", "future", "unspecified"}:
                    raise ValueError("Invalid polarity/tense")
                if item["mode"] not in {"episodic", "generic", "habitual", "unspecified"}:
                    raise ValueError("Invalid event mode")
                for arg in item["arguments"]:
                    ref(arg["target"], {"entities", "events", "literals"}); ref(arg["mention"], {"mentions"}); support(arg)
                    if arg["target"] is None and arg["support"] != "unresolved":
                        raise ValueError("Null role fillers must be explicitly unresolved")
            elif family == "relations":
                ref(item["source"], set(NODE_COLLECTIONS)); ref(item["target"], set(NODE_COLLECTIONS))
            elif family in {"properties", "qualifiers"}:
                ref(item["target"])
                value_refs(item["value"])
                if family == "properties" and item["operation"] not in {"assert", "retract"}:
                    raise ValueError("Invalid property operation")
            elif family == "identity_links":
                ref(item["left"], {"entities", "events"}); ref(item["right"], {"entities", "events"})
                if kinds[item["left"]] != kinds[item["right"]] or item["relation"] not in {"same", "different", "possible"}:
                    raise ValueError("Invalid identity claim")
            elif family == "uncertainties":
                for t in item["targets"]: ref(t)
                if item["resolution"] not in {"unresolved", "underspecified", "conservative_choice"}:
                    raise ValueError("Invalid uncertainty resolution")
    all_contexts = {c["id"]: c for r in prior for c in r["graph"]["contexts"]}
    all_contexts.update({c["id"]: c for c in graph["contexts"]})
    def visit(c, stack):
        if c in stack: raise ValueError("Cyclic nested context")
        for p in all_contexts[c]["parents"]: visit(p, stack | {c})
    for c in all_contexts: visit(c, set())
    mentions = [(m["evidence"]["start_token"], m["evidence"]["end_token"]) for m in graph["mentions"]]
    if len(mentions) != len(set(mentions)):
        raise ValueError("Duplicate mention span; use one unresolved mention with alternatives")
    return graph


def accept(raw, actor, author_file=None):
    sid = raw["story_id"]
    prior = records(sid)
    _, story = story_source(sid)
    if len(prior)==len(story["units"]): raise ValueError("Story already complete")
    unit = story["units"][len(prior)]
    packet_path = OUTPUT / "requests" / sid / (unit["id"]+".json")
    if not packet_path.exists(): raise ValueError("Current unit must be exposed before authoring/acceptance")
    packet = read_json(packet_path)
    if packet["actor"] != actor: raise ValueError("Wrong story author")
    if packet["parent_graph_hash"] != (prior[-1]["graph_hash"] if prior else None):
        raise ValueError("Exposure and accepted history disagree")
    if raw["unit_id"] != unit["id"]: raise ValueError("Cannot author an unexposed/future unit")
    validation_code = Path(__file__).read_bytes()
    validation_code_hash = hashlib.sha256(validation_code).hexdigest()
    immutable_json(OUTPUT/"support-code"/(validation_code_hash+".json"),
                   {"sha256": validation_code_hash, "module": validation_code.decode("utf-8")})
    attempt_path = OUTPUT / "attempts" / sid / unit["id"] / (object_hash({"draft": raw, "validation_code": validation_code_hash})+".json")
    attempt = {"actor": actor, "request_hash": object_hash(packet), "draft": raw,
               "validation_code_sha256": validation_code_hash,
               "author_source_sha256": hashlib.sha256(Path(author_file).read_bytes()).hexdigest() if author_file else None}
    try:
        graph = validate_graph(raw, story, unit, prior)
    except Exception as e:
        immutable_json(attempt_path, {**attempt, "structural_validation": "rejected", "error": str(e)})
        log_event(sid, {"operation": "structural_rejection", "unit_id": unit["id"], "draft_hash": object_hash(raw), "error": str(e)})
        raise
    immutable_json(attempt_path, {**attempt, "structural_validation": "passed"})
    response = {"actor": actor, "request_hash": object_hash(packet), "draft": raw}
    immutable_json(OUTPUT / "responses" / sid / (unit["id"] + ".json"), response)
    record = {"protocol": PROTOCOL, "story_id": sid, "unit_id": unit["id"],
              "available_at_token": unit["end_token"], "available_at_seconds": unit["offset_seconds"],
              "story_hash": packet["story_hash"], "request_hash": object_hash(packet),
              "response_hash": object_hash(response), "parent_graph_hash": packet["parent_graph_hash"],
              "graph": graph, "graph_hash": object_hash(graph), "actor": actor,
              "author_source": str(Path(author_file).resolve().relative_to(ROOT)).replace("\\", "/") if author_file else None,
              "author_source_sha256": hashlib.sha256(Path(author_file).read_bytes()).hexdigest() if author_file else None,
              "validation_code_sha256": validation_code_hash,
              "accepted_utc": now(), "structural_validation": "passed", "semantic_review": "human_review_pending"}
    immutable_json(OUTPUT / "records" / sid / (unit["id"] + ".json"), record)
    log_event(sid, {"operation": "accept_current_unit", "unit_id": unit["id"], "actor": actor,
                    "request_hash": record["request_hash"], "graph_hash": record["graph_hash"]})
    save_json(OUTPUT/"progress"/(sid+".json"), {"story_id": sid, "actor": actor,
        "accepted_units": len(prior)+1, "total_units": len(story["units"]),
        "last_unit": unit["id"], "last_graph_hash": record["graph_hash"],
        "next_unit": story["units"][len(prior)+1]["id"] if len(prior)+1<len(story["units"]) else None,
        "human_review": "pending"})
    return {"saved": unit["id"], "graph_hash": record["graph_hash"],
            "counts": {f: len(graph[f]) for f in COLLECTIONS}}


def q(quote, *, start=None, within=None):
    return {k: v for k, v in {"quote": quote, "start": start, "within": within}.items() if v is not None}


class Author:
    """Concise serialization helpers. Each helper invocation is an author decision.

    Defaults are declared conventions, never semantic guesses from the text.
    IDs/references are story-local author keys; quotation ambiguity is an error.
    """
    def __init__(self, sid, unit_number, actor):
        self.sid, self.actor = sid, actor
        self.unit_id = f"{sid}_u{unit_number:04d}"
        packet_path = OUTPUT / "requests" / sid / (self.unit_id + ".json")
        if not packet_path.exists(): raise ValueError("Call next to expose this unit before authoring it")
        self.packet = read_json(packet_path)
        if self.packet["actor"] != actor: raise ValueError("Author does not match the exposed passage")
        self.raw = {"protocol": PROTOCOL, "story_id": sid, "unit_id": self.unit_id,
                    **{f: [] for f in COLLECTIONS}, "decisions": []}
        self.prefix = f"u{unit_number:04d}_"

    def key(self, key):
        return key if key is None or key.startswith(self.sid+":") else self.sid+":"+key

    def evidence(self, quote):
        return q(quote) if isinstance(quote, str) else quote

    def add(self, family, key=None, **fields):
        key = key or self.prefix+family[:3]+str(len(self.raw[family]))
        self.raw[family].append({"id": self.key(key), **fields})
        return key

    def entity(self, key, label, concept, quote, *, kind="other", denotation="individual", introduction="explicit", sense=None):
        return self.add("entities", key, label=label, concept=concept, evidence=self.evidence(quote), kind=kind,
                        denotation=denotation, introduction=introduction, sense=sense)

    def mention(self, target, quote, *, key=None, form="description", support="explicit", antecedent=None, alternatives=()):
        return self.add("mentions", key, target=self.key(target), evidence=self.evidence(quote), form=form,
                        support=support, antecedent=self.key(antecedent), alternatives=[self.key(x) for x in alternatives])

    def mentions(self, target, quote, *, form="pronoun", support="explicit"):
        # The author explicitly assigns ALL occurrences of this expression in the
        # current unit. No automatic pronoun resolution or mention discovery.
        pieces = [w["text"] for w in transcript_tokens(quote)]
        tape = self.packet["current_words"]
        starts = [w["index"] for i,w in enumerate(tape) if [t["text"] for t in tape[i:i+len(pieces)]] == pieces]
        if not starts: raise ValueError("No current occurrences of " + quote)
        return [self.mention(target, q(quote, start=start), form=form, support=support) for start in starts]

    def literal(self, key, value, quote, *, kind="text", unit=None):
        return self.add("literals", key, value=value, kind=kind, unit=unit, evidence=self.evidence(quote))

    def context(self, key, kind, quote, *, holder=None, attribution=None, parents=(), support="explicit"):
        return self.add("contexts", key, kind=kind, evidence=self.evidence(quote), holder=self.key(holder),
                        attribution=self.key(attribution), parents=[self.key(x) for x in parents], support=support)

    def event(self, key, predicate, trigger, arguments=(), *, evidence=None, polarity="positive", tense="unspecified",
              mode="episodic", contexts=(), support="explicit", sense=None):
        args = list(arguments.items()) if isinstance(arguments, dict) else list(arguments)
        args = [{**a, "target": self.key(a["target"]), "mention": self.key(a.get("mention"))} if isinstance(a, dict) else {"role": a[0], "target": self.key(a[1]),
                 "support": "unresolved" if a[1] is None else support, "mention": None} for a in args]
        return self.add("events", key, predicate=predicate, trigger=self.evidence(trigger),
                        evidence=self.evidence(evidence or trigger), arguments=args, polarity=polarity, tense=tense,
                        mode=mode, contexts=[self.key(x) for x in contexts], support=support, sense=sense)

    def relation(self, type, source, target, quote, *, key=None, contexts=(), support="explicit"):
        return self.add("relations", key, type=type, source=self.key(source), target=self.key(target),
                        evidence=self.evidence(quote), contexts=[self.key(x) for x in contexts], support=support)

    def prop(self, target, attribute, value, quote, *, key=None, operation="assert", contexts=(), support="explicit"):
        return self.add("properties", key, target=self.key(target), attribute=attribute, value=value,
                        operation=operation, evidence=self.evidence(quote), contexts=[self.key(x) for x in contexts], support=support)

    def qualify(self, target, dimension, value, quote, *, key=None, contexts=(), support="explicit"):
        return self.add("qualifiers", key, target=self.key(target), dimension=dimension, value=value,
                        evidence=self.evidence(quote), contexts=[self.key(x) for x in contexts], support=support)

    def identity(self, left, right, quote, *, relation="same", key=None, contexts=(), support="explicit"):
        return self.add("identity_links", key, left=self.key(left), right=self.key(right), relation=relation,
                        evidence=self.evidence(quote), contexts=[self.key(x) for x in contexts], support=support)

    def uncertain(self, category, targets, quote, explanation, *, resolution="unresolved", alternatives=(), key=None):
        return self.add("uncertainties", key, category=category, targets=[self.key(t) for t in targets],
                        evidence=self.evidence(quote), explanation=explanation, resolution=resolution, alternatives=list(alternatives))

    def note(self, text):
        self.raw["decisions"].append(text)

    def finish(self, summary, *, coverage="annotated", author_file=None):
        self.raw.update(summary=summary, coverage=coverage)
        return accept(self.raw, self.actor, author_file)


def status():
    return [{"story_id": s["id"], "split": s["split"], "accepted": len(records(s["id"])),
             "total": s["units"]} for s in corpus_index()["stories"]]


def _prefix_receipt(package, sid, target, actor=None, story=None, original_package=None):
    """Verify the saved source-only recheck prefix, never supplying its suffix."""
    package = Path(package)
    if story is None: _, story = story_source(sid)
    original_package = original_package or package.relative_to(ROOT).as_posix()
    manifest = read_json(package/'manifest.json')
    author = read_json(package/'authors'/(sid+'.json'))
    if actor is not None and author['actor'] != actor: raise ValueError('Wrong prefix recheck actor')
    units = story['units'][:target+1]
    expected = {u['id']+'.json' for u in units}
    for family in ('records','requests','responses'):
        if {p.name for p in (package/family/sid).glob('*.json')} != expected:
            raise ValueError('Recheck must commit exactly the selected prefix, without exposing another unit')
    prior = []
    for unit in units:
        uid = unit['id']
        r = read_json(package/'records'/sid/(uid+'.json'))
        packet = read_json(package/'requests'/sid/(uid+'.json'))
        response = read_json(package/'responses'/sid/(uid+'.json'))
        source_hash = story.get('content_hash') or story['original_story_hash']
        _verify_record_inputs(r,packet,response,unit,story['words'][unit['start_token']:unit['end_token']],
                              author,manifest,source_hash,prior)
        if packet['current_text'] != story['text'][unit['char_start']:unit['prefix_char_end']]:
            raise ValueError('Recheck source differs')
        graph = validate_graph(response['draft'],story,unit,prior)
        if graph != r['graph'] or object_hash(graph) != r['graph_hash']:
            raise ValueError('Recheck draft replay differs')
        if r['author_source']:
            source = package/Path(r['author_source']).relative_to(Path(original_package))
            if hashlib.sha256(source.read_bytes()).hexdigest() != r['author_source_sha256']:
                raise ValueError('Recheck author source changed')
        prior.append(r)
    _verify_code_snapshots(prior,lambda digest:read_json(package/'support-code'/(digest+'.json')))
    return {'package':original_package,'story_id':sid,'actor':author['actor'],
            'source_only_committed_units':len(prior),'last_unit_id':units[-1]['id'],
            'request_hashes':[r['request_hash'] for r in prior],
            'graph_hashes':[r['graph_hash'] for r in prior],
            'story_hash':source_hash,'conventions_sha256':manifest['conventions_sha256'],
            'access_audit':'Saved prefix receipts and actor declaration; not an independent filesystem audit'}


def _alignment_input(receipt, current_text, current_draft, prior_graphs, original_graph_hash):
    return {'purpose':'Align an independently committed prefix interpretation to original IDs; no later source or graph supplied',
            'prefix_receipt':receipt,'current_text':current_text,
            'current_original_draft':current_draft,'original_prior_graphs':prior_graphs,
            'original_graph_hash':original_graph_hash}


def original_prefix_input(sid, target, prefix_package, actor):
    """After a blind prefix recheck, supply only matching original IDs/draft."""
    package = Path(prefix_package).resolve()
    if not package.is_relative_to((OUTPUT/'prefix-rechecks').resolve()):
        raise ValueError('Recheck package must stay under this production package')
    receipt = _prefix_receipt(package,sid,target,actor)
    if receipt['conventions_sha256'] != read_json(OUTPUT/'manifest.json')['conventions_sha256']:
        raise ValueError('Prefix recheck must use the pinned production conventions')
    rs = records(sid)
    if target >= len(rs): raise ValueError('Original unit not accepted')
    current = rs[target]
    packet = read_json(OUTPUT/'requests'/sid/(current['unit_id']+'.json'))
    value = _alignment_input(receipt,packet['current_text'],
             read_json(OUTPUT/'responses'/sid/(current['unit_id']+'.json'))['draft'],
             [r['graph'] for r in rs[:target]],current['graph_hash'])
    immutable_json(package/'integration-input.json',value)
    return value


def revision_digest(value):
    return object_hash({k:v for k,v in value.items() if k not in {'revision_hash','created_utc'}})


def save_revision(raw, actor, prefix_package, author_file, rationale):
    """Save a versioned prefix-only correction; never overwrite accepted history."""
    sid,uid = raw['story_id'],raw['unit_id']
    target = int(uid.rsplit('_u',1)[1])
    package = Path(prefix_package).resolve()
    if not package.is_relative_to((OUTPUT/'prefix-rechecks').resolve()):
        raise ValueError('Recheck package must stay under production prefix-rechecks')
    receipt = _prefix_receipt(package,sid,target,actor)
    if receipt['conventions_sha256'] != read_json(OUTPUT/'manifest.json')['conventions_sha256']:
        raise ValueError('Prefix recheck must use the pinned production conventions')
    integration = read_json(package/'integration-input.json')
    if integration['prefix_receipt'] != receipt: raise ValueError('Independent prefix receipt changed')
    rs = records(sid); original = rs[target]
    if integration['original_graph_hash'] != original['graph_hash']:
        raise ValueError('Original interpretation changed')
    expected_input = _alignment_input(receipt,
        read_json(OUTPUT/'requests'/sid/(uid+'.json'))['current_text'],
        read_json(OUTPUT/'responses'/sid/(uid+'.json'))['draft'],
        [r['graph'] for r in rs[:target]],original['graph_hash'])
    if integration != expected_input:
        raise ValueError('ID alignment must contain only the matching original prefix')
    _,story = story_source(sid)
    graph = validate_graph(raw,story,story['units'][target],rs[:target])
    old_ids = {x['id']:f for f in COLLECTIONS for x in original['graph'][f]}
    new_ids = {x['id']:f for f in COLLECTIONS for x in graph[f]}
    if any(new_ids.get(ident) != family for ident,family in old_ids.items()):
        raise ValueError('A prefix correction must retain original IDs/families for later reference; new supported nodes may be added')
    source = Path(author_file).resolve()
    value = {'protocol':'direct-prefix-revision-v1','story_id':sid,'unit_id':uid,
             'original_graph_hash':original['graph_hash'],'prefix_receipt':receipt,
             'integration_input_hash':object_hash(integration),'actor':actor,
             'draft':raw,'graph':graph,'graph_hash':object_hash(graph),'rationale':rationale,
             'author_source':source.relative_to(ROOT).as_posix(),
             'author_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'information_condition':'Independent source-only progressive prefix committed before viewing matching original prefix IDs; no later source or completed story graph',
             'human_review':'pending','semantic_accuracy_measured':False}
    digest = revision_digest(value)
    path = OUTPUT/'revisions'/uid/(digest+'.json')
    if path.exists():
        saved = read_json(path)
        if revision_digest(saved) != digest: raise ValueError('Revision digest collision')
    else:
        immutable_json(path,{**value,'revision_hash':digest,'created_utc':now()})
    index_path = OUTPUT/'revision-index.json'
    index = read_json(index_path) if index_path.exists() else {'protocol':'direct-prefix-revision-v1','active':{}}
    if uid in index['active'] and index['active'][uid] != digest:
        raise ValueError('A different active correction exists; select a new version explicitly')
    index['active'][uid] = digest
    save_json(index_path,index)
    return {'saved_revision':uid,'revision_hash':digest,'original_history_modified':False}


def load_revisions():
    path = OUTPUT/'revision-index.json'
    if not path.exists(): return {}
    result = {}
    for uid,digest in read_json(path)['active'].items():
        value = read_json(OUTPUT/'revisions'/uid/(digest+'.json'))
        if revision_digest(value) != digest or value['revision_hash'] != digest:
            raise ValueError('Revision checksum mismatch')
        if hashlib.sha256((ROOT/value['author_source']).read_bytes()).hexdigest() != value['author_source_sha256']:
            raise ValueError('Correction author source changed')
        result[uid] = value
    return result


def corrected_records(all_records, response_drafts, source_stories, revisions):
    """Recompile explicit versioned corrections, retaining original provenance."""
    result, priors = [], {}
    for original in all_records:
        sid,uid = original['story_id'],original['unit_id']
        prior = priors.setdefault(sid,[])
        story = source_stories[sid]; unit = story['units'][len(prior)]
        correction = revisions.get(uid)
        if correction and correction['original_graph_hash'] != original['graph_hash']:
            raise ValueError('Correction targets a different original graph')
        raw = correction['draft'] if correction else response_drafts[uid]
        graph = validate_graph(raw,story,unit,prior)
        if correction and (graph != correction['graph'] or object_hash(graph) != correction['graph_hash']):
            raise ValueError('Corrected draft replay differs from the saved correction')
        row = {'protocol':PROTOCOL,'interpretation_view':'author_corrected_prefix',
               'story_id':sid,'unit_id':uid,'story_hash':original['story_hash'],
               'available_at_token':original['available_at_token'],'available_at_seconds':original['available_at_seconds'],
               'request_hash':original['request_hash'],'graph':graph,'graph_hash':object_hash(graph),
               'original_record_hash':object_hash(original),'original_graph_hash':original['graph_hash'],
               'operational_parent_graph_hash':original['parent_graph_hash'],
               'compiled_parent_graph_hash':prior[-1]['graph_hash'] if prior else None,
               'revision_hash':correction['revision_hash'] if correction else None,
               'original_actor':original['actor'],'correction_actor':correction['actor'] if correction else None,
               'semantic_review':original['semantic_review']}
        result.append(row); prior.append(row)
    return result


def jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False)+"\n" for r in rows).encode("utf-8")


def anchored_graph_schema():
    """Describe saved graphs separately from raw authoring drafts."""
    schema = deepcopy(Draft.model_json_schema())
    schema['title'] = 'Accepted evidence-linked semantic graph'
    schema['description'] = ('The graph field of original and author-corrected records. '
        'Evidence coordinates are derived from author-selected quotations. '
        'Source matching, reference availability and context cycles require deterministic replay; '
        'schema conformity does not establish semantic correctness.')
    evidence = schema['$defs']['Evidence']
    evidence['title'] = 'Anchored evidence'
    for name in ('start_token','end_token','char_start','char_end'):
        evidence['properties'][name] = {'type':'integer','minimum':0}
    evidence['properties']['verbatim'] = {'type':'string','minLength':1}
    evidence['required'] = [*evidence['required'],'start_token','end_token','char_start','char_end','verbatim']
    return schema


def immutable_bytes(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable export differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def compile_records(all_records, packets, source_stories, splits):
    """Pure, lossless compilation of saved decisions. No label inference or fitting.

    Source-stories contain only each captured accepted prefix. Statement values,
    roles, context parents and unresolved references are retained verbatim.
    """
    nodes, statements, concepts, arguments, sources = [], [], [], [], []
    for r in all_records:
        g = r["graph"]
        sid, uid = g["story_id"], g["unit_id"]
        packet = packets[uid]
        unit = packet["unit"]
        source = {"story_id": sid, "unit_id": uid, "split": splits[sid],
                  "available_at_token": r["available_at_token"],
                  "available_at_seconds": r["available_at_seconds"],
                  "story_hash": r["story_hash"], "request_hash": r["request_hash"]}
        sources.append({**source, "text": packet["current_text"], "unit": unit,
                        "words": source_stories[sid]["words"][unit["start_token"]:unit["end_token"]]})
        for family in COLLECTIONS:
            for item in g[family]:
                row = {**source, "record_type": family, "record": item,
                       "graph_hash": r["graph_hash"], "semantic_review": r["semantic_review"]}
                (nodes if family in NODE_COLLECTIONS else statements).append(row)
                if family in {"entities", "events"}:
                    concepts.append({**source, "node_id": item["id"],
                        "kind": "concept" if family == "entities" else "predicate",
                        "label": item["concept"] if family == "entities" else item["predicate"],
                        "sense": item.get("sense"), "evidence": item["evidence"],
                        "semantic_review": r["semantic_review"]})
                if family == "events":
                    for position, arg in enumerate(item["arguments"]):
                        arguments.append({**source, "event_id": item["id"], "argument_index": position,
                            "argument": arg, "event_contexts": item["contexts"],
                            "event_polarity": item["polarity"], "event_evidence": item["evidence"],
                            "graph_hash": r["graph_hash"], "semantic_review": r["semantic_review"]})
    return {"annotations.jsonl": all_records, "nodes.jsonl": nodes, "statements.jsonl": statements,
            "arguments.jsonl": arguments, "concept-introductions.jsonl": concepts,
            "uncertainties.jsonl": [r for r in statements if r["record_type"] == "uncertainties"],
            "source-only.jsonl": sources}


def _review_page(title, content):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'+html.escape(title)+
        '</title><style>body{max-width:1100px;margin:35px auto;padding:0 15px;font:16px system-ui;line-height:1.5}'
        'article{border-top:1px solid #ccc;padding:20px 0}.source{background:#eff4ff;padding:15px;white-space:pre-wrap}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse}'
        'td,th{padding:6px 15px;text-align:left;border-bottom:1px solid #ddd}</style></head><body>'+content+'</body></html>')


def _review_outputs(all_records, sources, coverage, captured_files, revisions=None):
    revisions = revisions or {}
    outputs, links, blind_links = {}, [], []
    by_story = {s["story_id"]: [] for s in coverage}
    source_map = {s["unit_id"]: s for s in sources}
    for r in all_records: by_story[r["story_id"]].append(r)
    for s in coverage:
        sid = s["story_id"]
        if not by_story[sid]: continue
        source_pages, pages = [], []
        review_links = []
        for name in ("semantic_review.json", "review.md", "final_checkpoint.json"):
            target = "provenance/authoring/"+sid+"/"+name
            if target in captured_files:
                review_links.append('<a href="../'+target+'">'+html.escape(name)+'</a>')
        for r in by_story[sid]:
            g = r["graph"]; source = source_map[g["unit_id"]]; uid = g["unit_id"]
            meta = 'Tokens '+str(source["unit"]["start_token"])+':'+str(source["unit"]["end_token"])+\
                   '; available at '+str(source["available_at_seconds"])+ ' seconds.'
            base = '<article id="'+uid+'"><h2>'+uid+'</h2><p>'+meta+'</p><p class="source">'+html.escape(source["text"])+'</p>'
            source_pages.append(base+'</article>')
            counts = ', '.join(f'{f}: {len(g[f])}' for f in COLLECTIONS if g[f]) or 'No semantic records'
            correction_link = ('<p><b>Prefix-only author correction:</b> <a href="../provenance/revisions/'+uid+'/'+
                r['revision_hash']+'.json">Revision and independent prefix receipt</a></p>') if r.get('revision_hash') else ''
            if uid in revisions:
                branch = 'provenance/'+Path(revisions[uid]['prefix_receipt']['package']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                notes = [branch+'/'+name for name in ('revision_notes.html','revision_notes.md','revision_checkpoint.json')
                         if branch+'/'+name in captured_files]
                correction_link += '<p>'+' | '.join('<a href="../'+name+'">'+html.escape(Path(name).name)+'</a>' for name in notes)+'</p>'
            pages.append(base+correction_link+'<p><b>Author summary:</b> '+html.escape(g["summary"])+
                '</p><p>Coverage: '+html.escape(g["coverage"])+'. '+counts+'.</p><p><a href="../provenance/requests/'+
                sid+'/'+uid+'.json">Original input</a> · <a href="../provenance/responses/'+sid+'/'+uid+
                '.json">Original raw draft</a> · <a href="../provenance/records/'+sid+'/'+uid+
                '.json">Original accepted graph</a> · <a href="../provenance/authoring/'+sid+'/'+uid.replace(sid+'_','')+
                '.py">Original author decisions</a></p><details><summary>Complete structured graph</summary><pre>'+
                html.escape(json.dumps(g, ensure_ascii=False, indent=2))+'</pre></details></article>')
        heading = ('<h1>'+sid+' ('+s["split"]+')</h1><p>'+str(s["accepted"])+'/'+str(s["total"])+
                   ' units. Independent human review pending. Structural validity does not establish semantic correctness.</p>')
        outputs['review/'+sid+'.html'] = _review_page(sid+' annotations', heading+
            '<p><a href="../review.html">Index</a> · <a href="../source-review/'+sid+
            '.html">Source-only version</a></p><p>Author self-review: '+(' · '.join(review_links) or 'Not yet saved')+
            '</p>'+''.join(pages)).encode('utf-8')
        outputs['source-review/'+sid+'.html'] = _review_page(sid+' source only',
            '<h1>'+sid+' source only</h1><p>This file contains the captured source prefix and no annotation labels. '
            'For incremental review, supply one unit at a time to a fresh review context and commit its judgment before exposing the next.</p>'+''.join(source_pages)).encode('utf-8')
        links.append('<tr><td><a href="review/'+sid+'.html">'+sid+'</a></td><td>'+s['split']+
                     '</td><td>'+str(s['accepted'])+'/'+str(s['total'])+'</td></tr>')
        blind_links.append('<li><a href="source-review/'+sid+'.html">'+sid+'</a> · <a href="sources/'+sid+'.jsonl">JSONL</a></li>')
    outputs['review.html'] = _review_page('Deniz direct annotation review',
        '<h1>Deniz direct annotation review</h1><p>Human review pending. These are production author decisions, '
        'not gold labels. This view includes selected versioned corrections from fresh prefix rechecks; original accepted history is linked per unit. '
        'Review source evidence, role binding, scope, qualifications, reference and uncertainty separately from structural checks.</p>'
        '<p><a href="source-only.html">Source-only materials</a> · <a href="REVIEW.md">Review instructions</a> · '
        '<a href="report.json">Coverage and provenance</a></p><table><tr><th>Story</th><th>Split</th><th>Accepted/total</th></tr>'+
        ''.join(links)+'</table>').encode('utf-8')
    outputs['source-only.html'] = _review_page('Deniz source-only review',
        '<h1>Deniz source-only review</h1><p>No annotation labels appear in these source files. Avoid loading later units '
        'or completed graphs into a context used for incremental judgments.</p><ul>'+''.join(blind_links)+'</ul>').encode('utf-8')
    return outputs


def export_package(require_complete=False):
    manifest = initialize()
    index = corpus_index()
    all_records, coverage, packets, source_stories, files = [], [], {}, {}, {}
    def capture(name, path): files[name] = path.read_bytes()
    # Capture each immutable prefix once. Counts and exports use this same snapshot,
    # even when other actors accept another unit during export.
    for entry in index["stories"]:
        sid = entry["id"]; rs = records(sid)
        coverage.append({"story_id": sid, "split": entry["split"], "accepted": len(rs), "total": entry["units"]})
        if not rs: continue
        _, story = story_source(sid)
        last_unit = story["units"][len(rs)-1]
        source_stories[sid] = {"story_id": sid, "original_story_hash": entry["content_hash"],
            "text": story["text"][:last_unit["prefix_char_end"]],
            "words": story["words"][:last_unit["end_token"]], "units": story["units"][:len(rs)]}
        files['sources/'+sid+'.json'] = jsonl_bytes([source_stories[sid]]).rstrip(b'\n')+b'\n'
        capture('provenance/authors/'+sid+'.json', OUTPUT/'authors'/(sid+'.json'))
        accepted_ids = {r['unit_id'] for r in rs}
        trace_path = OUTPUT/'traces'/(sid+'.jsonl')
        if trace_path.exists():
            trace = []
            for line in trace_path.read_text(encoding='utf-8').splitlines():
                if not line.strip(): continue
                event = json.loads(line)
                if event.get('unit_id') in accepted_ids: trace.append(event)
            files['provenance/traces/'+sid+'.jsonl'] = jsonl_bytes(trace)
        if len(rs) == entry['units']:
            for path in sorted((OUTPUT/'authoring'/sid).glob('*')):
                if path.is_file() and path.suffix in {'.json','.py','.md','.txt','.html'}:
                    capture('provenance/authoring/'+sid+'/'+path.name,path)
        for r in rs:
            uid = r['unit_id']
            for family in ('requests','responses','records'):
                name = 'provenance/'+family+'/'+sid+'/'+uid+'.json'
                capture(name, OUTPUT/family/sid/(uid+'.json'))
            packet = json.loads(files['provenance/requests/'+sid+'/'+uid+'.json'])
            response = json.loads(files['provenance/responses/'+sid+'/'+uid+'.json'])
            if object_hash(packet) != r['request_hash'] or object_hash(response) != r['response_hash']:
                raise ValueError('Export request/response provenance mismatch: '+uid)
            packets[uid] = packet
            if r['author_source']:
                path = ROOT/r['author_source']
                name = 'provenance/'+path.relative_to(OUTPUT).as_posix()
                capture(name, path)
                if hashlib.sha256(files[name]).hexdigest() != r['author_source_sha256']:
                    raise ValueError('Author source changed after acceptance: '+uid)
            for path in sorted((OUTPUT/'attempts'/sid/uid).glob('*.json')):
                capture('provenance/'+path.relative_to(OUTPUT).as_posix(), path)
        all_records.extend(rs)
    if not all_records: raise ValueError('No actual annotations to export')
    revisions = load_revisions()
    revision_index = {'protocol':'direct-prefix-revision-v1','active':{uid:v['revision_hash'] for uid,v in revisions.items()}}
    files['revision-index.json'] = jsonl_bytes([revision_index])
    archive_paths = {}
    for uid,revision in revisions.items():
        if uid not in packets: raise ValueError('Correction is outside the captured accepted prefix')
        receipt = revision['prefix_receipt']; package = ROOT/receipt['package']
        fresh = _prefix_receipt(package,revision['story_id'],int(uid.rsplit('_u',1)[1]),revision['actor'])
        if fresh != receipt or fresh['conventions_sha256'] != manifest['conventions_sha256']:
            raise ValueError('Prefix recheck provenance changed')
        capture('provenance/revisions/'+uid+'/'+revision['revision_hash']+'.json',
                OUTPUT/'revisions'/uid/(revision['revision_hash']+'.json'))
        for path in sorted(package.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.jsonl','.py','.md','.txt','.html'}: continue
            relative = path.relative_to(OUTPUT).as_posix()
            name = 'provenance/'+relative
            if '/attempts/' in name:
                name = str(Path(name).with_name(path.stem[:20]+path.suffix)).replace('\\','/')
            if name in files and files[name] != path.read_bytes(): raise ValueError('Recheck archive path collision')
            capture(name,path); archive_paths[relative] = name
    files['recheck-archive-paths.json'] = jsonl_bytes([archive_paths])
    for path in sorted((OUTPUT/'support-code').glob('*.json')):
        capture('provenance/support-code/'+path.name, path)
    for name in ('neurosym/__init__.py','neurosym/direct_annotations.py','neurosym/corpus.py',
                 'neurosym/deniz.py','neurosym/io.py',
                  'scripts/direct_annotation.py','scripts/verify_direct_annotation.py','scripts/verify_direct_review.py',
                  'scripts/verify_direct_export_links.py','scripts/package_direct_annotation.py'):
        capture('code/'+name, ROOT/name)
    capture('conventions.md', GUIDE)
    capture('direct_annotation_conventions.md', GUIDE)
    capture('annotation-guide.md', ROOT/'docs/direct_annotation.md')
    for name in ('run-instructions.json','access-provenance.json'):
        path = OUTPUT/name
        if path.exists(): capture(name,path)
    for path in sorted((ROOT/'artifacts/direct-annotation-verification').glob('*.json')):
        capture('verification-history/'+path.name,path)
    structural_report = ROOT/'artifacts/direct-annotation-verification.json'
    if structural_report.exists() and read_json(structural_report).get('structurally_verified_units') == len(all_records):
        capture('structural-checks.json',structural_report)
    review_report = ROOT/'artifacts/direct-annotation-review-links.json'
    if review_report.exists() and read_json(review_report).get('complete_self_review_coverage') and all(s['accepted']==s['total'] for s in coverage):
        capture('review-link-checks.json',review_report)
    files['schema.json'] = jsonl_bytes([Draft.model_json_schema()])
    files['schemas/accepted-graph.json'] = jsonl_bytes([anchored_graph_schema()])
    files['runtime.json'] = jsonl_bytes([{'python':platform.python_version(),
        'dependencies':{name:importlib.metadata.version(name) for name in ('pydantic','numpy','h5py','praatio')},
        'annotation_model':manifest['model_provenance'], 'export_involves_model_call':False}])
    inputs = {'export_protocol':'lossless-direct-records-v3', 'manifest':manifest,
              'records':[object_hash(r) for r in all_records],
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}}
    fingerprint = object_hash(inputs)
    # Keep Windows paths below the legacy path limit; the full digest is retained
    # and checked in snapshot.json, report.json and latest.json.
    out = OUTPUT/'exports'/fingerprint[:20]
    out.mkdir(parents=True, exist_ok=True)
    for name,data in files.items(): immutable_bytes(out/name, data)
    compiled = compile_records(all_records, packets, source_stories, {s['story_id']:s['split'] for s in coverage})
    for name,rows in compiled.items(): immutable_bytes(out/name, jsonl_bytes(rows))
    response_drafts = {uid:json.loads(files['provenance/responses/'+packet['story_id']+'/'+uid+'.json'])['draft']
                       for uid,packet in packets.items()}
    effective = corrected_records(all_records,response_drafts,source_stories,revisions)
    corrected = compile_records(effective,packets,source_stories,{s['story_id']:s['split'] for s in coverage})
    for name,rows in corrected.items(): immutable_bytes(out/'author-corrected'/name,jsonl_bytes(rows))
    for sid in source_stories:
        immutable_bytes(out/'sources'/(sid+'.jsonl'), jsonl_bytes([s for s in compiled['source-only.jsonl'] if s['story_id']==sid]))
    for name,data in _review_outputs(effective, compiled['source-only.jsonl'], coverage, files, revisions).items():
        immutable_bytes(out/name, data)
    immutable_json(out/'manifest.json', manifest)
    immutable_json(out/'snapshot.json', {'build_hash':fingerprint,'inputs':inputs})
    report = {'protocol':PROTOCOL,'build_hash':fingerprint,'coverage':coverage,
              'complete':all(s['accepted']==s['total'] for s in coverage),
              'accepted_units':len(all_records),'source_words':sum(len(s['words']) for s in source_stories.values()),
              'counts':dict(Counter(row['record_type'] for row in compiled['nodes.jsonl']+compiled['statements.jsonl'])),
              'compiled_argument_rows':len(compiled['arguments.jsonl']),
              'selected_prefix_corrections':{uid:v['revision_hash'] for uid,v in revisions.items()},
              'recommended_annotation_view':'author-corrected/annotations.jsonl; root annotations.jsonl is the unchanged original attempt',
              'author_corrected_counts':dict(Counter(row['record_type'] for row in corrected['nodes.jsonl']+corrected['statements.jsonl'])),
              'accepted_records_missing_historical_code_hash':sum(r.get('validation_code_sha256') is None for r in all_records),
              'semantic_accuracy_measured':False,'human_review':'pending',
              'export_policy':'Lossless saved decisions; no inferred identities, flattened scopes, fitted vocabularies, numerical features or downstream analysis',
              'incremental_access_audit':'Saved requests and lineage; actor declarations and operational traces. Not a filesystem sandbox or an independent audit of all context exposure.'}
    immutable_json(out/'report.json', report)
    immutable_bytes(out/'REVIEW.md', (
        '# Reviewing the direct annotation attempt\n\n'
        'Open review.html for source-linked graphs and each author\'s self-review. '
        'This is the production attempt, not adjudicated gold data. Human review is pending.\n\n'
        'The author-corrected/ view includes explicitly selected prefix-only revisions and is the recommended annotation input. '
        'Root-level annotation rows retain the unchanged original attempt. Corrections are authored in fresh contexts that '
        'first commit source-only prefix interpretations, then see only matching original prefix records for ID alignment. '
        'No suffix or completed story graph is supplied. Original operational lineage and corrected compilation lineage are separate.\n\n'
        'Author self-reviews describe the original attempt. Some flagged issues may have been addressed by selected corrections; '
        'check per-unit revision links and report.json selected_prefix_corrections for their status.\n\n'
        'For fresh incremental review, use a new story context and one unit from sources/<story>.jsonl at a time. '
        'Do not preload later source text, source-review HTML, accepted labels, completed graphs or author review notes. '
        'Commit the prefix interpretation before providing the next unit. For retrospective review the complete source and graph views are available. '
        'Record review type, evidence, disputed IDs and availability point; distinguish a correction of an earlier mistake from later information.\n\n'
        'Inspect reference, semantic roles, nested scope, polarity, time, quantification, figurative senses and unresolved alternatives. '
        'Successful structural replay does not establish their correctness. The concept-introductions file contains introductions only; '
        'mentions and repeated predicates remain in full records. Source timing is copied as released, including nulls and unresolved flags.\n\n'
        'provenance/ contains actual requests, raw drafts, author decisions, attempted structural submissions, available traces and code snapshots. '
        'Stories 05 and 06 were authored progressively in the integration context with no prior target-story exposure; '
        'other stories used fresh isolated agents. The integration context had seen other-story development material. '
        'Early units lack some later-added trace/code-hash fields; these were not fabricated retrospectively. '
        'Runtime model variant/settings are unknown; runtime.json preserves nulls.\n\n'
        'Reproduce and verify from the repository in Windows CMD (no model call):\n\n'
        '```cmd\n.venv\\Scripts\\python.exe -B scripts\\direct_annotation.py verify\n'
        '.venv\\Scripts\\python.exe -B scripts\\direct_annotation.py export --require-complete\n'
        '.venv\\Scripts\\python.exe -B scripts\\direct_annotation.py verify-export\n```\n'
    ).encode('utf-8'))
    checksums = {p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.rglob('*')) if p.is_file() and p != out/'checksums.json'}
    immutable_json(out/'checksums.json', checksums)
    save_json(OUTPUT/'latest.json', {'build_hash':fingerprint,'directory':out.relative_to(ROOT).as_posix(),
                                   'complete':report['complete'],'accepted_units':len(all_records)})
    if require_complete and not report['complete']:
        raise ValueError('Actual annotation coverage remains incomplete; partial exports saved')
    return report


def _verify_record_inputs(r, packet, response, unit, words, author, manifest, original_story_hash, prior):
    uid = unit['id']
    if (r['protocol'],r['story_id'],r['unit_id']) != (PROTOCOL,unit['story_id'],uid):
        raise ValueError('Accepted record identity mismatch: '+uid)
    if packet['unit'] != unit or packet['current_words'] != [{'index':w['index'],'text':w['text']} for w in words]:
        raise ValueError('Source coordinates or exposed words differ: '+uid)
    if not (r['actor'] == packet['actor'] == response['actor'] == author['actor']):
        raise ValueError('Story actor mismatch: '+uid)
    if not (r['story_hash'] == packet['story_hash'] == author['story_hash'] == original_story_hash):
        raise ValueError('Source hash provenance mismatch: '+uid)
    if packet['conventions_sha256'] != manifest['conventions_sha256'] or author['conventions_sha256'] != manifest['conventions_sha256']:
        raise ValueError('Pinned conventions mismatch: '+uid)
    if object_hash(packet) != r['request_hash'] or response['request_hash'] != r['request_hash'] or object_hash(response) != r['response_hash']:
        raise ValueError('Request/response provenance mismatch: '+uid)
    if r['available_at_token'] != unit['end_token'] or r['available_at_seconds'] != unit['offset_seconds']:
        raise ValueError('Availability endpoint differs from source: '+uid)
    if r['parent_graph_hash'] != (prior[-1]['graph_hash'] if prior else None) or packet['parent_graph_hash'] != r['parent_graph_hash']:
        raise ValueError('Progressive lineage mismatch: '+uid)
    if packet['prior_committed_units'] != len(prior):
        raise ValueError('Exposed prefix count mismatch: '+uid)


def _verify_code_snapshots(all_records, read_snapshot):
    digests = sorted({r['validation_code_sha256'] for r in all_records if r.get('validation_code_sha256')})
    for digest in digests:
        if not re.fullmatch('[0-9a-f]{64}',digest): raise ValueError('Invalid historical code digest')
        saved = read_snapshot(digest)
        if saved['sha256'] != digest or hashlib.sha256(saved['module'].encode('utf-8')).hexdigest() != digest:
            raise ValueError('Historical validator code changed: '+digest)
    return len(digests)


def verify_export(directory=None):
    """Check a frozen bundle and replay its raw decisions without live corpus files."""
    out = Path(directory) if directory is not None else ROOT/read_json(OUTPUT/'latest.json')['directory']
    checksums = read_json(out/'checksums.json')
    actual_names = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p != out/'checksums.json'}
    if actual_names != set(checksums): raise ValueError('Export file inventory differs from checksums')
    for name,digest in checksums.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Export checksum mismatch: '+name)
    snapshot = read_json(out/'snapshot.json')
    if object_hash(snapshot['inputs']) != snapshot['build_hash']:
        raise ValueError('Export input fingerprint mismatch')
    for name,digest in snapshot['inputs']['files'].items():
        if checksums.get(name) != digest: raise ValueError('Captured input mismatch: '+name)
    def rows(name):
        return [json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines() if line]
    all_records = rows('annotations.jsonl')
    if [object_hash(r) for r in all_records] != snapshot['inputs']['records']:
        raise ValueError('Accepted record fingerprint mismatch')
    report = read_json(out/'report.json')
    manifest = read_json(out/'manifest.json')
    if (out/'schema.json').read_bytes() != jsonl_bytes([Draft.model_json_schema()]):
        raise ValueError('Frozen authoring schema differs')
    if (out/'schemas/accepted-graph.json').read_bytes() != jsonl_bytes([anchored_graph_schema()]):
        raise ValueError('Frozen accepted-graph schema differs')
    if manifest != snapshot['inputs']['manifest'] or hashlib.sha256((out/'conventions.md').read_bytes()).hexdigest() != manifest['conventions_sha256']:
        raise ValueError('Frozen source/conventions manifest differs')
    packets, source_stories, prior_by_story, authors = {}, {}, {}, {}
    for s in report['coverage']:
        sid = s['story_id']
        prior_by_story[sid] = []
        if s['accepted']:
            source_stories[sid] = read_json(out/'sources'/(sid+'.json'))
            authors[sid] = read_json(out/'provenance/authors'/(sid+'.json'))
    for r in all_records:
        sid, uid = r['story_id'], r['unit_id']; prior = prior_by_story[sid]
        story = source_stories[sid]; unit = story['units'][len(prior)]
        packet = read_json(out/'provenance/requests'/sid/(uid+'.json'))
        response = read_json(out/'provenance/responses'/sid/(uid+'.json'))
        saved_record = out/'provenance/records'/sid/(uid+'.json')
        if saved_record.exists() and read_json(saved_record) != r:
            raise ValueError('Frozen original record differs from its annotation row: '+uid)
        _verify_record_inputs(r,packet,response,unit,story['words'][unit['start_token']:unit['end_token']],
                              authors[sid],manifest,story['original_story_hash'],prior)
        if packet['current_text'] != story['text'][unit['char_start']:unit['prefix_char_end']] or packet['prior_committed_units'] != len(prior):
            raise ValueError('Frozen progressive source mismatch: '+uid)
        graph = validate_graph(response['draft'],story,unit,prior)
        if graph != r['graph'] or object_hash(graph) != r['graph_hash']:
            raise ValueError('Frozen structural replay differs: '+uid)
        if r['author_source']:
            name = 'provenance/'+Path(r['author_source']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
            if checksums[name] != r['author_source_sha256']: raise ValueError('Frozen author source mismatch: '+uid)
        packets[uid] = packet; prior.append(r)
    if len(all_records) != report['accepted_units'] or any(len(prior_by_story[s['story_id']]) != s['accepted'] for s in report['coverage']):
        raise ValueError('Export coverage count mismatch')
    expected = compile_records(all_records,packets,source_stories,{s['story_id']:s['split'] for s in report['coverage']})
    for name,value in expected.items():
        if (out/name).read_bytes() != jsonl_bytes(value): raise ValueError('Lossless compilation differs: '+name)
    for sid in source_stories:
        subset = [s for s in expected['source-only.jsonl'] if s['story_id']==sid]
        if (out/'sources'/(sid+'.jsonl')).read_bytes() != jsonl_bytes(subset):
            raise ValueError('Per-story source export differs: '+sid)
    revisions = {}
    for uid,digest in read_json(out/'revision-index.json')['active'].items():
        revision = read_json(out/'provenance/revisions'/uid/(digest+'.json'))
        if revision_digest(revision) != digest or revision['revision_hash'] != digest:
            raise ValueError('Frozen correction checksum mismatch')
        original_package = revision['prefix_receipt']['package']
        package = out/'provenance'/Path(original_package).relative_to(OUTPUT.relative_to(ROOT))
        receipt = _prefix_receipt(package,revision['story_id'],int(uid.rsplit('_u',1)[1]),revision['actor'],
                                  source_stories[revision['story_id']],original_package)
        if receipt != revision['prefix_receipt'] or receipt['conventions_sha256'] != manifest['conventions_sha256']:
            raise ValueError('Frozen independent recheck differs')
        integration = read_json(package/'integration-input.json')
        if object_hash(integration) != revision['integration_input_hash']:
            raise ValueError('Frozen ID-alignment input differs')
        sid = revision['story_id']; target = int(uid.rsplit('_u',1)[1])
        original = prior_by_story[sid][target]
        expected_input = _alignment_input(receipt,packets[uid]['current_text'],
            read_json(out/'provenance/responses'/sid/(uid+'.json'))['draft'],
            [r['graph'] for r in prior_by_story[sid][:target]],original['graph_hash'])
        if integration != expected_input:
            raise ValueError('Frozen alignment contains more than the matching original prefix')
        author = out/'provenance'/Path(revision['author_source']).relative_to(OUTPUT.relative_to(ROOT))
        if hashlib.sha256(author.read_bytes()).hexdigest() != revision['author_source_sha256']:
            raise ValueError('Frozen correction author changed')
        revisions[uid] = revision
    drafts = {uid:read_json(out/'provenance/responses'/packet['story_id']/(uid+'.json'))['draft'] for uid,packet in packets.items()}
    effective = corrected_records(all_records,drafts,source_stories,revisions)
    corrected = compile_records(effective,packets,source_stories,{s['story_id']:s['split'] for s in report['coverage']})
    for name,value in corrected.items():
        if (out/'author-corrected'/name).read_bytes() != jsonl_bytes(value):
            raise ValueError('Versioned correction compilation differs: '+name)
    code_count = _verify_code_snapshots(all_records,lambda digest:read_json(out/'provenance/support-code'/(digest+'.json')))
    return {'build_hash':snapshot['build_hash'],'checked_files':len(checksums),
            'structurally_replayed_units':len(all_records),'lossless_compilation_verified':True,
            'historical_code_snapshots_verified':code_count,
            'prefix_only_corrections_verified':len(revisions),
            'replay_validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'semantic_accuracy_measured':False,'human_review':'pending'}


def verify():
    manifest = initialize()
    count = 0
    all_records = []
    for s in corpus_index()["stories"]:
        prior = []
        _, story = story_source(s["id"])
        rs = records(s['id'])
        author = read_json(OUTPUT/'authors'/(s['id']+'.json')) if rs else None
        for i, r in enumerate(rs):
            unit = story["units"][i]
            packet = read_json(OUTPUT/"requests"/s["id"]/(unit["id"]+".json"))
            response = read_json(OUTPUT/"responses"/s["id"]/(unit["id"]+".json"))
            _verify_record_inputs(r,packet,response,unit,story['words'][unit['start_token']:unit['end_token']],
                                  author,manifest,story['content_hash'],prior)
            if packet["current_text"] != story["text"][unit["char_start"]:unit["prefix_char_end"]] or packet["prior_committed_units"] != i:
                raise ValueError("Progressive exposure record mismatch")
            if packet["parent_graph_hash"]!=(prior[-1]["graph_hash"] if prior else None):
                raise ValueError("Future input or broken exposure chain")
            replay = validate_graph(response["draft"], story, unit, prior)
            if replay != r["graph"] or object_hash(replay) != r['graph_hash']:
                raise ValueError("Graph replay or hash differs")
            if r["author_source"] and hashlib.sha256((ROOT/r["author_source"]).read_bytes()).hexdigest()!=r["author_source_sha256"]:
                raise ValueError("Author source changed after acceptance")
            prior.append(r); all_records.append(r); count += 1
    code_count = _verify_code_snapshots(all_records,lambda digest:read_json(OUTPUT/'support-code'/(digest+'.json')))
    return {"protocol": manifest["protocol"], "structurally_verified_units": count,
            "replay_validator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "progressive_request_chain_verified": True, "historical_code_snapshots_verified":code_count,
            "accepted_records_missing_historical_code_hash":sum(r.get('validation_code_sha256') is None for r in all_records),
            "semantic_accuracy_measured": False}
