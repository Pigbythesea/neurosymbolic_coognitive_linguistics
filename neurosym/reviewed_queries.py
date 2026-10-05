"""Open-vocabulary reviewed queries, with private provenance and answer-free selectors."""
from collections import Counter
from itertools import combinations

from .io import object_hash

TASKS = {"concept", "role", "reference", "scope", "polarity", "binding", "relation", "identity", "property", "qualifier", "compose"}


def validate_reviewed_ast(ast):
    if set(ast) != {"op", "task", "anchor", "operation", "steps"} or ast["op"] != "reviewed_choice" or ast["task"] not in TASKS:
        raise ValueError("Invalid reviewed query syntax.")
    if not isinstance(ast["operation"], dict) or not isinstance(ast["steps"], list):
        raise ValueError("Malformed reviewed operation.")
    # No raw identifiers, answer keys, graph expressions or evidence quotations are legal public syntax.
    def check(value):
        if isinstance(value, dict):
            if set(value) & {"id", "targets", "target_id", "graph", "answer", "evidence", "acceptable_indices", "uncertainty"}:
                raise ValueError("Private graph information in query.")
            for v in value.values():
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
        elif isinstance(value, str) and value.startswith("story_") and ":" in value:
            raise ValueError("Raw discourse identifier in public query.")
    check(ast)


def compile_reviewed_queries(history, current, source, seed):
    queries, answers, oracle = [], [], []
    skipped = Counter()
    catalogs, order_cache = {}, {}
    targets = {i: history.descriptor(i) for i, k in history.kinds.items() if k in {"entities", "events", "literals", "contexts"}}
    concepts = {object_hash(history.concept(i)): history.concept(i) for i, k in history.kinds.items() if k in {"entities", "events", "literals"}}
    # Every prefix candidate catalog is shared across anchors, with no answer-conditioned type filtering.
    relations = {a["canonical_type"]: {"label": a["canonical_type"]} for a in history.aliases.values() if "canonical_type" in a}
    def add(task, anchor, operation, catalog, gold, nodes, *, steps=(), uncertain=(), variant=None):
        # Several discourse IDs can have the same legal public description. Score that
        # observable equivalence class, never demand an inaccessible ID distinction.
        original = catalog
        cache_key = id(original)
        if cache_key not in catalogs:
            mapping = {key: object_hash(value) for key, value in original.items()}
            catalogs[cache_key] = (original, mapping, {mapping[key]: value for key, value in original.items()})
        _, mapping, catalog = catalogs[cache_key]
        gold = {mapping[key] for key in gold}
        if len(catalog) < 2:
            skipped[task + ":no_alternatives"] += 1
            return
        ast = {"op": "reviewed_choice", "task": task, "anchor": anchor, "operation": operation, "steps": list(steps)}
        validate_reviewed_ast(ast)
        if (cache_key, task) not in order_cache:
            order = sorted(catalog, key=lambda k: object_hash([seed, source["id"], task, k]))
            order_cache[(cache_key, task)] = (order, [catalog[i] for i in order], object_hash(order))
        order, candidates, candidate_set = order_cache[(cache_key, task)]
        gold = set(gold)
        if not gold <= set(order):
            raise ValueError("Reviewed answer outside prefix candidates.")
        if len(gold) == len(order):
            skipped[task + ":all_candidates_acceptable"] += 1
            return
        record = {"ast": ast, "candidates": candidates, "candidate_set": candidate_set}
        qid = object_hash({"source": source["id"], "ast": ast, "candidate_set": candidate_set})
        family = {"concept": "concept_identification", "role": "event_role", "reference": "reference",
                  "scope": "event_scope", "polarity": "event_polarity", "binding": "binding_assignment",
                  "relation": "event_relation", "identity": "identity_update", "compose": "composed",
                  "property": "state_update", "qualifier": "qualification"}[task]
        warnings = {w["id"]: w for w in uncertain}
        queries.append({"id": qid, "source_id": source["id"], "family": family, "input": record, "ast_hash": object_hash(ast)})
        supported = bool(gold) and not warnings
        answers.append({"query_id": qid, "source_id": source["id"], "family": family,
                        "acceptable_indices": sorted(order.index(i) for i in gold) if supported else [],
                        "label_state": "annotation_supported" if supported else "unresolved",
                        "label_interpretation": "Identify the recorded scoped interpretation; alternatives are not asserted false in the world.",
                        "diagnostic_only": False, "variant": variant, "scoring_eligible": supported and source["decoder_timing_eligible"],
                        "annotation_review_status": "accepted_reviewed", "review_flags": sorted(warnings),
                        "nodes": list(nodes), "evidence": [history.records[n]["evidence"] for n in nodes],
                        "source_weight": 0.0})
        oracle.append({"query_id": qid, "candidate_set": candidate_set, "annotated_candidate_ids": sorted(gold),
                       "uncertainty": list(warnings.values()), "dependencies": list(nodes),
                       "available_at_token": source["available_at_token"]})
    # Scope is the complete operator structure, not one arbitrarily selected context label.
    scopes = {object_hash(history.scope_descriptor(i)): history.scope_descriptor(i)
              for i, k in history.kinds.items() if k == "events"}
    for identity in current:
        r, kind = history.records[identity], history.kinds[identity]
        if kind in {"entities", "events", "literals"}:
            # This label task measures linguistic content, not resolved world existence or precise referent extension.
            add("concept", history.selector(identity, content=False), {"op": "concept_label"}, concepts,
                [object_hash(history.concept(identity))], [identity], variant="content_label_not_world_existence")
        if kind == "mentions" and (r.get("antecedent") is not None or r.get("form") in {"pronoun", "deictic", "relative", "demonstrative"}):
            deps = [identity]
            warnings = history.uncertainty(deps)
            if r.get("alternatives") or r.get("target") is None:
                warnings = warnings + [{"id": identity + ":unresolved_reference"}]
            add("reference", history.selector(identity, content=False), {"op": "reference"}, targets,
                [r["target"]] if r.get("target") else [], deps, uncertain=warnings)
        if kind == "events":
            selector = history.selector(identity)
            alias = history.aliases[identity]["argument_aliases"]
            for position, a in enumerate(r["arguments"]):
                deps = [identity] + ([a["mention"]] if a.get("mention") else [])
                # Target type uncertainty does not erase the recorded binding to that discourse node.
                warnings = history.uncertainty(deps)
                if a["target"] is None:
                    warnings += [{"id": identity + ":null_argument:" + str(position)}]
                add("role", selector, {"op": "role", "role": alias[position]["canonical_role"], "position": position},
                    targets, [a["target"]] if a["target"] else [], deps, uncertain=warnings)
            # Raw polarity is explicitly a field-recovery diagnostic, separate from the composed scope answer.
            if r["polarity"] in {"positive", "negative"}:
                add("polarity", selector, {"op": "recorded_polarity"}, {x: {"label": x} for x in ("positive", "negative")},
                    [r["polarity"]], [identity], uncertain=history.uncertainty([identity]), variant="local_field_not_proposition_truth")
            scope = history.scope_descriptor(identity)
            add("scope", selector, {"op": "scoped_expression"}, scopes, [object_hash(scope)], [identity],
                uncertain=history.uncertainty([identity], contextual=True), variant="complete_symbolic_scope")
            bindings = [{"role": alias[j]["canonical_role"], "position": j, "filler": targets.get(a["target"])}
                        for j, a in enumerate(r["arguments"])]
            if len(bindings) >= 2 and all(b["filler"] is not None for b in bindings):
                gold = {"assignment": bindings}
                catalog = {object_hash(gold): gold}
                for a, b in combinations(range(len(bindings)), 2):
                    if bindings[a]["role"] == bindings[b]["role"] or bindings[a]["filler"] == bindings[b]["filler"]:
                        continue
                    swapped = [dict(x) for x in bindings]
                    swapped[a]["filler"], swapped[b]["filler"] = swapped[b]["filler"], swapped[a]["filler"]
                    item = {"assignment": swapped}
                    catalog[object_hash(item)] = item
                deps = [identity] + [a["mention"] for a in r["arguments"] if a.get("mention")]
                add("binding", selector, {"op": "ordered_role_assignment"}, catalog, [object_hash(gold)], deps,
                    uncertain=history.uncertainty(deps), variant="recorded_assignment_choice_not_closed_world_contradiction")
            # Content-to-role compositions follow only explicit event arguments. No realization/identity inheritance.
            for position, a in enumerate(r["arguments"]):
                target = a["target"]
                if history.kinds.get(target) != "events":
                    continue
                for j, b in enumerate(history.records[target]["arguments"]):
                    if not b["target"]:
                        continue
                    steps = [{"op": "argument_event", "role": alias[position]["canonical_role"], "position": position},
                             {"op": "role", "role": history.aliases[target]["argument_aliases"][j]["canonical_role"], "position": j}]
                    deps = [identity, target] + [v for v in (a.get("mention"), b.get("mention")) if v]
                    add("compose", selector, {"op": "ordered_composition"}, targets, [b["target"]], deps,
                        steps=steps, uncertain=history.uncertainty(deps))
            holders = [history.records[c].get("holder") for c in r["contexts"]]
            holders = [h for h in holders if h in targets]
            if holders:
                add("compose", selector, {"op": "ordered_composition"}, targets, holders, [identity, *r["contexts"]],
                    steps=[{"op": "scope_parent"}, {"op": "context_holder"}],
                    uncertain=history.uncertainty([identity, *r["contexts"]]), variant="innermost_context_holder")
        elif kind == "relations":
            alias = history.aliases[identity]
            left, right = alias["canonical_source"], alias["canonical_target"]
            # Relation kind is the answer, not supplied in its anchor. Endpoints can be contexts.
            anchor = {"source": history.selector(left), "target": history.selector(right)}
            add("relation", anchor, {"op": "relation_type", "direction": "source_to_target"}, relations,
                [alias["canonical_type"]], [identity, left, right], uncertain=history.uncertainty([identity], contextual=True))
            if history.kinds[right] == "events":
                for j, argument in enumerate(history.records[right]["arguments"]):
                    if argument["target"] is None:
                        continue
                    steps = [{"op": "event_relation", "relation": alias["canonical_type"], "direction": "out"},
                             {"op": "role", "role": history.aliases[right]["argument_aliases"][j]["canonical_role"], "position": j}]
                    deps = [identity, right] + ([argument["mention"]] if argument.get("mention") else [])
                    add("compose", history.selector(left), {"op": "ordered_composition"}, targets, [argument["target"]], deps,
                        steps=steps, uncertain=history.uncertainty(deps, contextual=True),
                        variant="explicit_linked_expression_role_not_inherited_agency")
        elif kind == "identity_links":
            anchor = {"source": history.selector(r["left"]), "target": history.selector(r["right"])}
            add("identity", anchor, {"op": "scoped_identity"}, {x: {"label": x} for x in ("same", "different", "possible")},
                [r["relation"]], [identity], uncertain=history.uncertainty([identity], contextual=True))
        elif kind in {"qualifiers", "properties"}:
            field = "dimension" if kind == "qualifiers" else "attribute"
            values = {object_hash(history.abstract(v["value"])): {"value": history.abstract(v["value"])}
                      for i, v in history.records.items() if history.kinds[i] == kind and v[field] == r[field]}
            add("qualifier" if kind == "qualifiers" else "property", history.selector(r["target"]),
                {"op": "attached_value", field: r[field]}, values, [object_hash(history.abstract(r["value"]))], [identity],
                uncertain=history.uncertainty([identity], contextual=True), variant="symbolic_value_not_truth_projection")
    if len({q["id"] for q in queries}) != len(queries):
        # Same target/dimension can have two attachment records. Keep one query only if targets agree.
        unique = {}
        for q, a, o in zip(queries, answers, oracle, strict=True):
            if q["id"] in unique and unique[q["id"]][1]["acceptable_indices"] != a["acceptable_indices"]:
                for item in (a, unique[q["id"]][1]):
                    item.update(scoring_eligible=False, acceptable_indices=[], label_state="unresolved", source_weight=0.0)
            unique.setdefault(q["id"], (q, a, o))
        queries, answers, oracle = map(list, zip(*unique.values(), strict=True))
    counts = Counter(a["family"] for a in answers if a["scoring_eligible"])
    for answer in answers:
        if answer["scoring_eligible"]:
            answer["source_weight"] = 1 / (len(counts) * counts[answer["family"]])
    return queries, answers, oracle, dict(skipped)
