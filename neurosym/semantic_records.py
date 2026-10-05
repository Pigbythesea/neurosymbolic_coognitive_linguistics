"""Replay real prefix graphs into semantic observations, feature atoms and review records."""
from collections import Counter
import re

from .io import object_hash


def span_text(story, span):
    words = story["words"][span.start:span.end]
    return " ".join(word["text"] for word in words)


def event_key(event):
    return {"predicate": event.predicate, "sense_id": event.sense_id}


def context_profile(event, history):
    """Transferable nested scope structure; no story-specific IDs or offsets."""
    def describe(identity):
        context = history.contexts[identity]
        holder = history.entities.get(context.holder_id)
        attribution = history.events.get(context.attribution_event_id)
        return {"kind": context.kind,
                "holder": {"concept": holder.concept, "kind": holder.kind} if holder else None,
                "attribution": event_key(attribution) if attribution else None,
                "parents": sorted([describe(c) for c in context.parent_ids], key=object_hash)}
    values = [describe(c) for c in event.context_ids]
    if event.status is not None and event.status != "asserted":
        values.append({"kind": event.status, "holder": None, "attribution": None, "parents": []})
    return sorted(values, key=object_hash)


def scope_parents(event, history):
    parents = {c.attribution_event_id for c in history.context_chain(event.context_ids) if c.attribution_event_id}
    if event.scope_parent_id:
        parents.add(event.scope_parent_id)
    return sorted(parents - {event.id})


def feature_key(group, parts, route="factorized"):
    definition = {"group": group, "parts": parts, "route": route}
    return object_hash(definition), definition


def local_mention(story, mention):
    # Surface material only: a pronoun never obtains its antecedent's type here.
    return {"surface": span_text(story, mention.span).casefold(), "form": mention.form}


def observed_concept(story, mention, entity):
    if mention.reference_status == "unresolved" or mention.form in {"pronoun", "deictic"}:
        return False
    if mention.reference_status == "first_mention":
        return True
    # Conservative lexical attestation of a repeated label. This is not a sense
    # disambiguator or synonym normalizer; all other resolutions stay in R.
    label = entity.concept.replace("_", " ").casefold()
    surface = " ".join(re.findall(r"[\w'-]+", span_text(story, mention.span).casefold()))
    return bool(re.search(r"(?<!\w)" + re.escape(label) + r"(?:s|es)?(?!\w)", surface))


def compile_unit(story, unit, graph, history, saved, unit_index, alignment):
    definitions, terms, occurrences, reviews = {}, Counter(), [], []
    source = unit["id"]
    audit = saved.get("semantic_audit_status", "unreviewed")

    def review(reason, nodes, evidence, detail):
        item = {"source_id": source, "reason": reason, "nodes": nodes,
                "evidence": evidence, "detail": detail, "annotation_review_status": audit}
        item["id"] = object_hash(item)
        reviews.append(item)

    def emit(group, parts, nodes, evidence, *, route="factorized", dependencies=()):
        key, definition = feature_key(group, parts, route)
        definitions[key] = definition
        terms[key] += 1
        return {"feature": key, "nodes": nodes, "evidence": evidence,
                "dependencies": sorted(set(dependencies))}

    evidence_records = []

    def add(*args, **kwargs):
        evidence_records.append(emit(*args, **kwargs))

    def occur(kind, label, node, span, view, **extra):
        item = {"source_id": source, "story_id": story["story_id"], "kind": kind,
                "label": label, "node": node, "evidence": span, "view": view,
                "unit_index": unit_index, "available_at_seconds": unit["offset_seconds"],
                "available_at_token": unit["end_token"], "annotation_review_status": audit, **extra}
        item["id"] = object_hash(item)
        occurrences.append(item)

    for mention in graph.mentions:
        entity = history.entities[mention.entity_id]
        span = mention.span.model_dump()
        local = local_mention(story, mention)
        add("L", ["mention", local], [mention.id], [span])
        if observed_concept(story, mention, entity):
            concept = {"concept": entity.concept, "kind": entity.kind}
            add("C", ["entity_concept", concept], [mention.id, entity.id], [span])
            occur("concept", concept, entity.id, span, "lexically_attested", mention_id=mention.id)
        if mention.reference_status == "unresolved":
            review("unresolved_reference", [mention.id], [span], "No reference answer is inferred.")
            continue
        if mention.reference_status != "first_mention":
            concept = {"concept": entity.concept, "kind": entity.kind}
            add("R", ["resolved_concept", concept, mention.form], [mention.id, entity.id], [span],
                dependencies=["reference_resolution"])
            anchor = span_text(story, entity.first_mention).casefold()
            add("R", ["reference_anchor", local, anchor], [mention.id, entity.id],
                [span, entity.first_mention.model_dump()], dependencies=["reference_resolution"])
            occur("concept", concept, entity.id, span, "reference_resolved", mention_id=mention.id)
            if mention.antecedent_id:
                antecedent = history.mentions[mention.antecedent_id]
                distance = mention.span.start - antecedent.span.start
                bucket = "same_unit" if antecedent.span.start >= unit["start_token"] else "previous_unit"
                add("R", ["reference_link", mention.form, antecedent.form, bucket, mention.reference_status],
                    [mention.id, antecedent.id], [span, antecedent.span.model_dump()],
                    dependencies=["reference_resolution"])
                occur("reference", {"form": mention.form, "antecedent_form": antecedent.form,
                                    "distance_class": bucket}, mention.id, span, "reference_resolved",
                      antecedent_id=antecedent.id, distance_words=distance,
                      reference_status=mention.reference_status)

    configurations = []
    for literal in graph.literals:
        label = {"kind": literal.kind, "value": literal.value, "unit": literal.unit}
        add("C", ["literal", label], [literal.id], [literal.evidence.model_dump()])
        occur("literal", label, literal.id, literal.evidence.model_dump(), "literal_value")
    for event in graph.events:
        predicate = event_key(event)
        statuses = history.event_statuses(event)
        profile = context_profile(event, history)
        ev = event.evidence.model_dump()
        add("C", ["predicate", predicate], [event.id], [event.trigger.model_dump()])
        add("S", ["event_status", predicate, event.polarity, statuses, event.tense], [event.id], [ev])
        add("S", ["context_profile", predicate, profile], [event.id], [ev])
        occur("predicate", predicate, event.id, ev, "event", polarity=event.polarity, statuses=statuses, contexts=profile)
        if event.sense_id is None:
            review("unspecified_predicate_sense", [event.id], [ev],
                   "Predicate lemma retained without claiming a disambiguated sense.")
        if event.polarity == "unresolved" or "uncertain" in statuses:
            review("uncertain_event", [event.id], [ev], "Unsupported truth labels are masked.")
        ordered = []
        for argument in event.arguments:
            role = argument.role
            if argument.target_kind == "event":
                target = history.events[argument.target_id]
                local = {"event": event_key(target)} if target.trigger.start >= unit["start_token"] else {"event": "previous_event"}
                transferable = {"event": event_key(target)}
                support = [ev, target.evidence.model_dump()]
                deps = ["event_link"] + (["reference_resolution"] if target.trigger.start < unit["start_token"] else [])
            elif argument.target_kind == "literal":
                target = history.literals[argument.target_id]
                transferable = {"literal": {"kind": target.kind, "value": target.value, "unit": target.unit}}
                local = transferable if target.evidence.start >= unit["start_token"] else {"literal": "previous_value"}
                support = [ev, target.evidence.model_dump()]
                deps = [] if target.evidence.start >= unit["start_token"] else ["reference_resolution"]
            else:
                target = history.entities[argument.target_id]
                mention = history.mentions.get(argument.mention_id)
                # Missing grounding stays missing. Do not insert an earlier
                # referent label into a supposedly local role feature.
                local_available = mention is not None and mention.span.start >= unit["start_token"]
                local = local_mention(story, mention) if local_available else {"mention": "unavailable"}
                transferable = {"concept": target.concept, "kind": target.kind}
                support = [ev] + ([mention.span.model_dump()] if mention else [])
                deps = [] if local_available and mention.reference_status == "first_mention" else ["reference_resolution"]
                if mention is None:
                    review("argument_without_mention", [event.id, target.id], [ev],
                           "Binding retained; no invented local mention representation.")
            ordered.append({"role": role, "frame_role": argument.frame_role, "filler": transferable})
            # Shared factors can support unseen combinations of familiar parts.
            add("B", ["predicate_role", predicate, role, argument.frame_role], [event.id], [ev])
            add("B", ["role_local_filler", role, local], [event.id, target.id], support)
            add("B", ["local_binding", predicate, role, local], [event.id, target.id], support)
            group = "BR" if "reference_resolution" in deps else "B"
            unresolved = argument.target_kind == "entity" and argument.mention_id and history.mentions[argument.mention_id].reference_status == "unresolved"
            if not unresolved:
                add("GBR" if "reference_resolution" in deps else "GB", ["role_filler", role, transferable],
                    [event.id, target.id], support, dependencies=deps)
                add("PBR" if "reference_resolution" in deps else "PB", ["predicate_role_filler", predicate, role, transferable],
                    [event.id, target.id], support, dependencies=deps)
                add(group, ["typed_binding", predicate, role, transferable], [event.id, target.id], support,
                    dependencies=deps)
                if argument.target_kind == "entity" and "reference_resolution" in deps:
                    add("BR", ["referent_binding", predicate, role, span_text(story, target.first_mention).casefold()],
                        [event.id, target.id], [*support, target.first_mention.model_dump()], dependencies=deps)
            occur("role", {"predicate": predicate, "role": role, "filler": transferable}, event.id,
                  ev, "bound", target_id=target.id, dependencies=deps,
                  polarity=event.polarity, statuses=statuses, contexts=profile, grounding_resolved=not bool(unresolved))
            if role == "cause":
                review("causal_interpretation", [event.id, target.id], support,
                       "A cause role needs semantic review; schema validity is not causal evidence.")
        signature = {"predicate": predicate, "arguments": sorted(ordered, key=object_hash),
                     "polarity": event.polarity, "statuses": statuses, "contexts": profile}
        uses_reference = any((a.target_kind == "entity" and (not a.mention_id or
                             history.mentions[a.mention_id].span.start < unit["start_token"] or
                             history.mentions[a.mention_id].reference_status != "first_mention")) or
                             (a.target_kind == "event" and history.events[a.target_id].trigger.start < unit["start_token"]) or
                             (a.target_kind == "literal" and history.literals[a.target_id].evidence.start < unit["start_token"])
                             for a in event.arguments)
        unresolved_binding = any(a.mention_id and history.mentions[a.mention_id].reference_status == "unresolved" for a in event.arguments)
        if not unresolved_binding:
            add("BR" if uses_reference else "B", ["event_configuration", signature], [event.id], [ev],
                route="exact_configuration", dependencies=["reference_resolution"] if uses_reference else [])
        occur("configuration", signature, event.id, ev, "event_configuration", grounding_resolved=not unresolved_binding)
        configurations.append({"key": object_hash(signature), "definition": signature,
                               "grounding_resolved": not bool(unresolved_binding),
                               "constituents": sorted(set([object_hash(predicate)] +
                                   [object_hash({"role": a["role"]}) for a in ordered] +
                                   [object_hash(a["filler"]) for a in ordered])), "event_id": event.id})
        for parent_id in scope_parents(event, history):
            parent = history.events[parent_id]
            add("S", ["scope", predicate, event_key(parent), profile], [event.id, parent.id],
                [ev, parent.evidence.model_dump()], dependencies=["scope_link"])
    for relation in graph.relations:
        left, right = history.events[relation.source_event], history.events[relation.target_event]
        definition = {"relation": relation.type, "source": event_key(left), "target": event_key(right),
                      "status": relation.status, "context_kinds": sorted(c.kind for c in history.context_chain(relation.context_ids))}
        ev = relation.evidence.model_dump()
        if relation.status == "unresolved":
            review("unresolved_relation", [relation.id], [ev], "Relation is recorded but not treated as supported.")
        else:
            add("D", ["relation_type", relation.type, relation.status, definition["context_kinds"]], [relation.id], [ev])
            add("D", ["relation_source", relation.type, event_key(left)], [relation.id, left.id], [ev])
            add("D", ["relation_target", relation.type, event_key(right)], [relation.id, right.id], [ev])
            add("D", ["directed_relation", definition], [relation.id, left.id, right.id], [ev],
                route="exact_configuration", dependencies=["ordered_event_link"])
            occur("discourse", definition, relation.id, ev, "ordered_event_link",
                  context_selectors=sorted([{"kind": history.contexts[c].kind,
                      "anchor": [history.contexts[c].evidence.start, history.contexts[c].evidence.end]}
                      for c in relation.context_ids], key=object_hash))
        if relation.type in {"causes", "motivates", "enables", "condition"}:
            review("causal_interpretation", [relation.id], [ev], "Review the supported direction and inference status.")
    for update in graph.entity_updates:
        target = history.entities[update.entity_id]
        ev = update.evidence.model_dump()
        if update.status != "unresolved":
            add("U", ["state_update", target.concept, update.attribute, update.value, update.operation, update.status,
                      sorted(c.kind for c in history.context_chain(update.context_ids))],
                [target.id], [ev], dependencies=["reference_resolution", "state_update"])
        occur("state_update", {"concept": target.concept, "attribute": update.attribute, "value": update.value,
                               "operation": update.operation, "status": update.status,
                               "context_kinds": sorted(c.kind for c in history.context_chain(update.context_ids))}, target.id, ev, "state_update")
    for link in graph.identity_links:
        left, right = history.entities[link.left_entity], history.entities[link.right_entity]
        label = {"relation": link.relation, "left_concept": left.concept, "right_concept": right.concept,
                 "status": link.status, "context_kinds": sorted(c.kind for c in history.context_chain(link.context_ids))}
        if link.status != "unresolved":
            add("R", ["identity_update", label], [link.id, left.id, right.id], [link.evidence.model_dump()],
                dependencies=["reference_resolution"])
        occur("identity", label, link.id, link.evidence.model_dump(), "identity_update",
              grounding_resolved=link.status != "unresolved")
    for uncertainty in graph.uncertainties:
        review("annotator_uncertainty:" + uncertainty.category, [], [uncertainty.span.model_dump()], uncertainty.explanation)

    slot = alignment["units"]["bin_indices"][unit_index]
    raw_rows = [slot + lag for lag in alignment["fir_delays_trs"]] if slot >= 0 else []
    response_rows = alignment["response_raw_indices"]
    complete_neural_support = bool(raw_rows) and all(row in response_rows for row in raw_rows)
    source_record = {"id": source, "story_id": story["story_id"], "unit_index": unit_index,
                     "annotation_hash": saved["graph_hash"], "annotation_review_status": audit,
                     "annotation_protocol": saved.get("protocol_context", {}).get("protocol", "legacy-graph-v1"),
                     "local_word_span": [unit["start_token"], unit["end_token"]],
                     "prefix_word_span": [0, unit["end_token"]], "available_at_seconds": unit["offset_seconds"],
                     "model_unit_row": unit_index, "raw_feature_bin": slot,
                     "decoder_raw_response_rows": raw_rows,
                     "decoder_trimmed_response_rows": [response_rows.index(row) for row in raw_rows if row in response_rows],
                     "decoder_timing_eligible": complete_neural_support,
                     "neural_interpretation": "Offline delayed samples can contain responses to later presented text.",
                     "configurations": configurations}
    feature_record = {"source_id": source, "terms": dict(sorted(terms.items())), "evidence": evidence_records}
    return source_record, feature_record, definitions, occurrences, reviews


def state_view(history):
    """Keep the full update log; expose the latest resolved assertion per property."""
    properties = {}
    for item in history.updates:
        if item["status"] == "unresolved":
            continue
        key = (tuple(sorted(item.get("context_ids", []))), item["entity_id"], item["attribute"], item["value"])
        properties[key] = item
    return {"ledger": history.ledger(), "active_properties":
            [item for _, item in sorted(properties.items()) if item["operation"] == "assert" and not item.get("context_ids")],
            "contextual_properties": [item for _, item in sorted(properties.items()) if item["operation"] == "assert" and item.get("context_ids")],
            "identity_classes": [list(group) for group in sorted({tuple(sorted(history.equivalent_entities(e))) for e in history.entities})]}
