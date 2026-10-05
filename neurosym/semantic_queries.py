"""Typed query construction and graph-side answer evaluation. No language-model calls."""
from collections import Counter
from itertools import combinations
from typing import get_args

from .graphs import Argument
from .io import object_hash
from .semantic_records import observed_concept, span_text, context_profile, scope_parents

RELATIONS = ["before", "after", "overlap", "causes", "motivates", "enables", "condition", "contrast", "elaboration", "same_event"]
STATUSES = ["hypothetical", "desired", "promised", "questioned", "reported", "uncertain", "believed", "negated"]
ROLES = set(get_args(Argument.model_fields["role"].annotation))


def event_selector(event):
    return {"predicate": event.predicate, "trigger": [event.trigger.start, event.trigger.end]}


def select_event(selector, history):
    found = [e for e in history.events.values() if event_selector(e) == selector]
    if len(found) != 1:
        raise ValueError("ambiguous_event_selector")
    return found[0]


def context_selectors(ids, history):
    return sorted([{"kind": history.contexts[c].kind,
                    "anchor": [history.contexts[c].evidence.start, history.contexts[c].evidence.end]}
                   for c in ids], key=object_hash)


def matching_contexts(ids, selectors, history):
    return context_selectors(ids, history) == selectors


def validate_ast(ast):
    op = ast.get("op")
    fields = {"concept": {"op", "concept"}, "role": {"op", "event", "role"},
              "reference": {"op", "mention"}, "status": {"op", "event"},
              "polarity": {"op", "event"}, "relation": {"op", "source", "target", "contexts"},
              "binding": {"op", "event", "arguments", "polarity"},
              "compose": {"op", "start", "steps"}, "identity": {"op", "left", "right", "contexts"}}
    if op not in fields or set(ast) != fields[op]:
        raise ValueError("Unsupported or malformed query AST.")
    for key in ("event", "source", "target", "start"):
        if key not in ast:
            continue
        selector = ast[key]
        if not isinstance(selector, dict) or set(selector) != {"predicate", "trigger"} or not isinstance(selector["predicate"], str):
            raise ValueError("Invalid event selector.")
        span = selector["trigger"]
        if not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1]:
            raise ValueError("Invalid event trigger selector.")
    if op == "role" and ast["role"] not in ROLES:
        raise ValueError("Invalid role selector.")
    if op == "reference":
        span = ast["mention"]
        if not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1]:
            raise ValueError("Invalid reference selector.")
    if op == "concept" and (not isinstance(ast["concept"], str) or not ast["concept"]):
        raise ValueError("Invalid concept selector.")
    if op == "identity":
        for key in ("left", "right"):
            if not isinstance(ast[key], dict) or set(ast[key]) != {"first_mention_word"} or type(ast[key]["first_mention_word"]) is not int:
                raise ValueError("Invalid entity identity selector.")
    def check_contexts(selectors):
        if not isinstance(selectors, list):
            raise ValueError("Context selectors must be a list.")
        for c in selectors:
            if (not isinstance(c, dict) or set(c) != {"kind", "anchor"} or c["kind"] not in STATUSES
                    or not isinstance(c["anchor"], list) or len(c["anchor"]) != 2
                    or any(type(x) is not int for x in c["anchor"]) or not 0 <= c["anchor"][0] < c["anchor"][1]):
                raise ValueError("Malformed context selector.")
    if "contexts" in ast:
        check_contexts(ast["contexts"])
    if op == "binding":
        if ast["polarity"] not in {"positive", "negative", "unresolved"} or not isinstance(ast["arguments"], list):
            raise ValueError("Invalid binding query.")
        for argument in ast["arguments"]:
            if set(argument) != {"role", "candidate"} or argument["role"] not in ROLES or not isinstance(argument["candidate"], str):
                raise ValueError("Invalid ordered argument.")
    if op == "compose":
        if not ast["steps"] or len(ast["steps"]) > 3:
            raise ValueError("A composed query needs one to three declared steps.")
        for step in ast["steps"]:
            allowed = {"argument_event": {"op", "role"}, "role": {"op", "role"},
                       "scope_parent": {"op"}, "event_relation": {"op", "relation", "direction", "contexts"}}
            if step.get("op") not in allowed or set(step) != allowed[step["op"]]:
                raise ValueError("Unsupported composition step.")
            if step["op"] == "event_relation" and step["direction"] not in {"out", "in"}:
                raise ValueError("Invalid event-link direction.")
            if "role" in step and step["role"] not in ROLES:
                raise ValueError("Invalid compositional role.")
            if "relation" in step and step["relation"] not in RELATIONS:
                raise ValueError("Invalid compositional event relation.")
            if "contexts" in step:
                check_contexts(step["contexts"])


def evaluate_query(ast, history, graph, story, private):
    """Return annotation-supported identities; None means no justified answer label."""
    validate_ast(ast)
    op = ast["op"]
    if op == "identity":
        def select(selector):
            values = [e.id for e in history.entities.values() if e.first_mention.start == selector["first_mention_word"]]
            return values[0] if len(values) == 1 else None
        left, right = select(ast["left"]), select(ast["right"])
        if left is None or right is None:
            return None
        labels = {link.relation for link in graph.identity_links
                  if {left, right} == {link.left_entity, link.right_entity} and link.status != "unresolved"
                  and matching_contexts(link.context_ids, ast["contexts"], history)}
        return labels if len(labels) == 1 else None
    if op == "concept":
        present = any(history.entities[m.entity_id].concept == ast["concept"] and
                      observed_concept(story, m, history.entities[m.entity_id]) for m in graph.mentions)
        return {"supported"} if present else None
    if op == "reference":
        matches = [m for m in graph.mentions if [m.span.start, m.span.end] == ast["mention"]]
        if len(matches) != 1 or matches[0].reference_status == "unresolved":
            return None
        return history.equivalent_entities(matches[0].entity_id)
    if op == "relation":
        left, right = select_event(ast["source"], history), select_event(ast["target"], history)
        labels = {r.type for r in history.relations.values() if r.source_event == left.id and
                  r.target_event == right.id and r.status != "unresolved"
                  and matching_contexts(r.context_ids, ast["contexts"], history)}
        return labels or None
    if op == "compose":
        current = {select_event(ast["start"], history).id}
        for step in ast["steps"]:
            following = set()
            for identity in current:
                if identity not in history.events:
                    continue
                event = history.events[identity]
                if step["op"] in {"role", "argument_event"}:
                    for argument in event.arguments:
                        if argument.role == step["role"] and (step["op"] == "role" or argument.target_kind == "event"):
                            if argument.mention_id and history.mentions[argument.mention_id].reference_status == "unresolved":
                                continue
                            following.add(argument.target_id)
                elif step["op"] == "scope_parent":
                    following.update(scope_parents(event, history))
                elif step["op"] == "event_relation":
                    for r in history.relations.values():
                        if (r.type != step["relation"] or r.status == "unresolved"
                                or not matching_contexts(r.context_ids, step["contexts"], history)):
                            continue
                        a, b = (r.source_event, r.target_event) if step["direction"] == "out" else (r.target_event, r.source_event)
                        if a == identity:
                            following.add(b)
            current = following
        return current or None
    event = select_event(ast["event"], history)
    if op == "status":
        return None if "uncertain" in history.event_statuses(event) else {object_hash(context_profile(event, history))}
    if op == "polarity":
        return None if event.polarity == "unresolved" else {event.polarity}
    if op == "role":
        return {a.target_id for a in event.arguments if a.role == ast["role"] and
                (not a.mention_id or history.mentions[a.mention_id].reference_status != "unresolved")} or None
    if op == "binding":
        actual = {(a.role, a.target_id) for a in event.arguments if not a.mention_id or
                  history.mentions[a.mention_id].reference_status != "unresolved"}
        requested = {(a["role"], private["bindings"][a["candidate"]]) for a in ast["arguments"]}
        if not requested or not requested <= actual or event.polarity == "unresolved" or "uncertain" in history.event_statuses(event):
            return None
        # Negation concerns the selected proposition within its recorded scope.
        # It does not establish whether any other event occurs in the world.
        return {"supported" if ast["polarity"] == event.polarity else "contradicted"}
    raise ValueError("Unimplemented query operator.")


def entity_descriptor(entity, history, story):
    first = [m for m in history.mentions.values() if m.entity_id == entity.id and m.span == entity.first_mention]
    name = span_text(story, first[0].span) if len(first) == 1 and first[0].form == "name" else None
    return {"type": "entity", "concept": entity.concept, "kind": entity.kind, "name": name,
            "first_mention_word": entity.first_mention.start}


def compile_queries(story, unit, graph, history, source, seed):
    queries, answers, private_records, skipped = [], [], [], Counter()
    seen = set()
    entities = {identity: entity_descriptor(e, history, story) for identity, e in history.entities.items()}
    events = {identity: {"type": "event", **event_selector(e)} for identity, e in history.events.items()}
    literals = {identity: {"type": "literal", "kind": v.kind, "value": v.value, "unit": v.unit}
                for identity, v in history.literals.items()}
    all_targets = {**entities, **events, **literals}
    # A candidate denotes the WHOLE recorded scope profile, not one member of a
    # multilabel target incorrectly treated as an interchangeable single answer.
    profiles = [[]] + [[{"kind": s, "holder": None, "attribution": None, "parents": []}] for s in STATUSES]
    profiles.extend(context_profile(e, history) for e in history.events.values())
    scope_candidates = {object_hash(p): {"type": "scope_profile", "contexts": p} for p in profiles}

    def add(family, ast, catalog, evidence, nodes, *, arguments=None, diagnostic=False, variant=None):
        try:
            validate_ast(ast)
            unique = object_hash({"source": unit["id"], "ast": ast, "arguments": arguments})
            if unique in seen:
                return
            seen.add(unique)
            order = sorted(catalog, key=lambda identity: object_hash([seed, unit["id"], family, identity]))
            descriptor_list = [catalog[identity] for identity in order]
            private = {"candidate_ids": order, "bindings": {}}
            if arguments is not None:
                binding_order = sorted({a[1] for a in arguments}, key=lambda x: object_hash([seed, unique, x]))
                private["bindings"] = {str(i): identity for i, identity in enumerate(binding_order)}
                binding_lookup = {identity: str(i) for i, identity in enumerate(binding_order)}
                ast = {**ast, "arguments": [{"role": role, "candidate": binding_lookup[identity]}
                                             for role, identity in arguments]}
            result = evaluate_query(ast, history, graph, story, private)
        except ValueError as error:
            if str(error) != "ambiguous_event_selector":
                raise
            skipped["ambiguous_event_selector"] += 1
            return
        if result is not None and not result <= set(catalog):
            raise ValueError("Answer identity is missing from the legal candidate catalog.")
        if len(catalog) < 2:
            skipped["no_alternative_candidates"] += 1
            return
        input_record = {"ast": ast, "candidates": descriptor_list}
        if arguments is not None:
            input_record["binding_candidates"] = [all_targets[private["bindings"][str(i)]] for i in range(len(private["bindings"]))]
        qid = object_hash({"source_id": unit["id"], "family": family, "input": input_record})
        labelled = result is not None
        queries.append({"id": qid, "source_id": unit["id"], "family": family, "input": input_record,
                        "ast_hash": object_hash(ast)})
        answers.append({"query_id": qid, "source_id": unit["id"], "family": family,
                        "acceptable_indices": sorted(order.index(identity) for identity in result) if labelled else [],
                        "label_state": "annotation_supported" if labelled else "needs_adjudication",
                        "label_interpretation": "Recorded textual support within the selected proposition/scope; not world truth.",
                        "diagnostic_only": diagnostic, "variant": variant,
                        "scoring_eligible": labelled and not diagnostic and source["decoder_timing_eligible"],
                        "annotation_review_status": source["annotation_review_status"],
                        "evidence": evidence, "nodes": nodes,
                        "review_flags": ["semantic_review_pending"] if source["annotation_review_status"] == "unreviewed" else []})
        private_records.append({"query_id": qid, **private})

    for concept in sorted({history.entities[m.entity_id].concept for m in graph.mentions
                           if observed_concept(story, m, history.entities[m.entity_id])}):
        add("concept_occurrence", {"op": "concept", "concept": concept},
            {"supported": {"label": "supported"}, "contradicted": {"label": "contradicted"}}, [], [], diagnostic=True,
            variant="positive_only_no_absence_inference")
    for mention in graph.mentions:
        if mention.reference_status in {"explicit", "inferred"} and mention.antecedent_id:
            # The candidate set is all available entities, not a target-filtered
            # list with the answer's type, gender, or role leaked into its shape.
            add("reference", {"op": "reference", "mention": [mention.span.start, mention.span.end]}, entities,
                [mention.span.model_dump()], [mention.id, mention.antecedent_id])
    for event in graph.events:
        selector = event_selector(event)
        for role in sorted({a.role for a in event.arguments}):
            add("event_role", {"op": "role", "event": selector, "role": role}, all_targets,
                [event.evidence.model_dump()], [event.id])
        add("event_scope", {"op": "status", "event": selector}, scope_candidates,
            [event.evidence.model_dump()], [event.id])
        add("event_polarity", {"op": "polarity", "event": selector},
            {x: {"label": x} for x in ["positive", "negative"]}, [event.evidence.model_dump()], [event.id])
        bindings = [(a.role, a.target_id) for a in event.arguments]
        if bindings:
            for polarity in ["positive", "negative"]:
                add("binding_verification", {"op": "binding", "event": selector, "arguments": [], "polarity": polarity},
                    {x: {"label": x} for x in ["supported", "contradicted"]}, [event.evidence.model_dump()], [event.id],
                    arguments=bindings, variant="recorded_binding_with_polarity_contrast")
            # Actual role-swapped candidates are saved, but a missing graph edge
            # is never promoted to contradiction or textual indeterminacy.
            for first, second in combinations(range(len(bindings)), 2):
                if bindings[first][0] == bindings[second][0] or bindings[first][1] == bindings[second][1]:
                    continue
                swapped = bindings.copy()
                swapped[first] = (bindings[first][0], bindings[second][1])
                swapped[second] = (bindings[second][0], bindings[first][1])
                add("binding_verification", {"op": "binding", "event": selector, "arguments": [], "polarity": event.polarity},
                    {x: {"label": x} for x in ["supported", "contradicted"]}, [event.evidence.model_dump()], [event.id],
                    arguments=swapped, variant="role_swap_requires_evidence")
        for argument in event.arguments:
            if argument.target_kind != "event":
                continue
            target = history.events[argument.target_id]
            for role in sorted({a.role for a in target.arguments}):
                add("composed", {"op": "compose", "start": selector,
                                 "steps": [{"op": "argument_event", "role": argument.role}, {"op": "role", "role": role}]},
                    all_targets, [event.evidence.model_dump(), target.evidence.model_dump()], [event.id, target.id])
        for parent_id in scope_parents(event, history):
            parent = history.events[parent_id]
            for role in sorted({a.role for a in parent.arguments}):
                add("composed", {"op": "compose", "start": selector,
                                 "steps": [{"op": "scope_parent"}, {"op": "role", "role": role}]}, all_targets,
                    [event.evidence.model_dump(), parent.evidence.model_dump()], [event.id, parent.id])
    for relation in graph.relations:
        if relation.status == "unresolved":
            skipped["unresolved_relation"] += 1
            continue
        left, right = history.events[relation.source_event], history.events[relation.target_event]
        selectors = context_selectors(relation.context_ids, history)
        add("event_relation", {"op": "relation", "source": event_selector(left), "target": event_selector(right), "contexts": selectors},
            {x: {"label": x} for x in RELATIONS}, [relation.evidence.model_dump()], [relation.id])
        for role in sorted({a.role for a in right.arguments}):
            add("composed", {"op": "compose", "start": event_selector(left),
                              "steps": [{"op": "event_relation", "relation": relation.type, "direction": "out", "contexts": selectors},
                                       {"op": "role", "role": role}]}, all_targets,
                [relation.evidence.model_dump(), right.evidence.model_dump()], [relation.id, right.id])
    for link in graph.identity_links:
        if link.status == "unresolved":
            skipped["unresolved_identity"] += 1
            continue
        left, right = history.entities[link.left_entity], history.entities[link.right_entity]
        add("identity_update", {"op": "identity", "left": {"first_mention_word": left.first_mention.start},
            "right": {"first_mention_word": right.first_mention.start}, "contexts": context_selectors(link.context_ids, history)},
            {x: {"label": x} for x in ("same", "different", "possible")}, [link.evidence.model_dump()], [link.id])
    return queries, answers, private_records, dict(skipped)
