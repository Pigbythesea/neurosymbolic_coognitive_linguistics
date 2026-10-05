"""Prefix-scoped expression DAG over the accepted records; never global identity union."""
from collections import defaultdict
from copy import deepcopy

from .io import object_hash
from .reviewed_archive import RECORD_TYPES


def references(value):
    if isinstance(value, str):
        return {value} if value.startswith("story_") and ":" in value and " " not in value else set()
    if isinstance(value, dict):
        return set().union(*(references(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(references(v) for v in value)) if value else set()
    return set()


class ReviewedHistory:
    def __init__(self, constraints):
        self.constraints = constraints
        self.records, self.kinds, self.availability, self.aliases = {}, {}, {}, {}
        self.modifiers, self.warnings = defaultdict(list), defaultdict(list)
        self.endpoint = -1
        self.active_cases = []
        self._scope_cache = {}

    def append(self, saved, normalized):
        self.endpoint = saved["available_at_token"]
        self._scope_cache.clear()
        current = []
        for kind in RECORD_TYPES:
            for r in saved["graph"][kind]:
                identity = r["id"]
                if identity in self.records:
                    raise ValueError("An annotation observation ID was redefined: " + identity)
                self.records[identity], self.kinds[identity] = deepcopy(r), kind
                self.availability[identity] = {"unit_id": saved["unit_id"], "available_at_token": self.endpoint,
                                                "available_at_seconds": saved["available_at_seconds"]}
                self.aliases[identity] = {k: deepcopy(v) for k, v in normalized[identity].items()
                                         if k.startswith("canonical_") or k in {"argument_aliases", "endpoints_reversed"}}
                current.append(identity)
        for identity in current:
            r, kind = self.records[identity], self.kinds[identity]
            # Decisions/explanations can discuss unchosen alternatives; inspect structural reference fields only.
            structural = {k: v for k, v in r.items() if k not in {"evidence", "trigger", "explanation", "label", "sense"}}
            if references(structural) - set(self.records):
                raise ValueError("Unavailable reference in accepted prefix: " + identity + str(references(structural) - set(self.records)))
            if kind in {"qualifiers", "properties"}:
                self.modifiers[r["target"]].append(identity)
            if kind == "uncertainties" and r.get("resolution") not in {"resolved", "settled"}:
                for target in r["targets"]:
                    self.warnings[target].append({"id": identity, "origin": "graph_uncertainty", **deepcopy(r),
                                                  **self.availability[identity]})
        for q in self.constraints["questions"]:
            if q["unit_id"] == saved["unit_id"]:
                for target in q["affected_ids"]:
                    self.warnings[target].append({"id": q["issue_id"], "origin": "review_question", **deepcopy(q)})
        self.active_cases = [c for c in self.constraints["cases"] if c["available_at_token"] <= self.endpoint]
        # No later modifier is attached before its own availability endpoint.
        return current

    def cases(self, identity):
        return [c for c in self.active_cases if c.get("target") == identity or
                c.get("new_record") == identity or identity in c.get("joint_relation_records", [])]

    def active_modifiers(self, identity):
        superseded = {c["supersedes"] for c in self.active_cases if c["kind"] == "prospective_qualification_revision"}
        return [r for r in self.modifiers[identity] if r not in superseded]

    def concept(self, identity):
        r, kind = self.records[identity], self.kinds[identity]
        if kind == "entities":
            return {"concept": r["concept"], "kind": r["kind"], "sense": r.get("sense"), "denotation": r["denotation"]}
        if kind == "events":
            return {"predicate": self.aliases[identity]["canonical_predicate"], "sense": r.get("sense")}
        if kind == "literals":
            return {"literal": {"kind": r["kind"], "value": self.abstract(r["value"]), "unit": r.get("unit")}}
        if kind == "contexts":
            return {"context_kind": self.aliases[identity]["canonical_kind"]}
        return {"record_type": kind}

    def abstract(self, value):
        """Transferable descriptor; exact IDs/coordinates remain only in the linked expression DAG."""
        if isinstance(value, str) and value in self.records:
            return self.concept(value)
        if isinstance(value, dict):
            return {k: self.abstract(v) for k, v in value.items()
                    if k not in {"id", "evidence", "trigger", "source_evidence", "case_id",
                                 "unit_id", "available_at_token", "available_at_seconds", "integration_provenance", "human_status"}}
        if isinstance(value, list):
            return [self.abstract(v) for v in value]
        return value

    def selector(self, identity, *, content=True):
        r = self.records[identity]
        e = r.get("trigger", r["evidence"])
        selector = {"type": self.kinds[identity], "anchor": [e["start_token"], e["end_token"]]}
        if content:
            selector.update(self.concept(identity))
        return selector

    def descriptor(self, identity):
        # No role, outgoing edge, qualifier, correct referent or resolved scope in answer alternatives.
        return self.selector(identity)

    def context_paths(self, identity):
        def paths(cid, trail):
            if cid in trail or self.kinds.get(cid) != "contexts":
                raise ValueError("Invalid context DAG.")
            parents = self.records[cid]["parents"]
            return [p + [cid] for parent in parents for p in paths(parent, trail + [cid])] if parents else [[cid]]
        r = self.records[identity]
        roots = r["parents"] if self.kinds[identity] == "contexts" else r.get("contexts", [])
        return [p for c in roots for p in paths(c, [])]

    def target_context_paths(self, identity):
        paths = self.context_paths(identity)
        if self.kinds[identity] == "contexts":
            return [p + [identity] for p in paths] if paths else [[identity]]
        return paths

    def context_operator(self, cid):
        r = self.records[cid]
        cases = self.cases(cid)
        attribution = r.get("attribution")
        result = {"id": cid, "kind": r["kind"], "operator_family": self.aliases[cid]["canonical_kind"],
                  "holder": r.get("holder"), "attribution": attribution, "parents": r["parents"],
                  "content_of_attribution": attribution is not None,
                  "projection": "none_automatically", "support": r.get("support")}
        if attribution:
            event = self.records[attribution]
            result["attribution_semantics"] = {"predicate": event.get("predicate"), "polarity": event.get("polarity"),
                                                "sense": event.get("sense"), "mode": event.get("mode")}
        result["qualifications"] = [{"id": q, "dimension": self.records[q].get("dimension", self.records[q].get("attribute")),
                                      "value": deepcopy(self.records[q]["value"]), "context_paths": self.context_paths(q)}
                                     for q in self.active_modifiers(cid)]
        for case in cases:
            if case["kind"] == "context_continuation":
                result["same_operator_as"] = case["same_operator_as"]
            elif case["kind"] == "context_attribution_semantics":
                result.update(operator=case["operator"], projection=case["projection"])
            elif case["kind"] == "context_formula":
                result["formula"] = case["formula"]
            elif case["kind"] == "compound_condition":
                result["formula"] = {"op": "if", "antecedent": {"op": "and", "items":
                    [{"op": "expression_ref", "id": i} for i in case["antecedents"]]},
                    "consequent": {"op": "expression_ref", "id": case["consequent"]}}
        return result

    def expression(self, identity):
        r, kind = self.records[identity], self.kinds[identity]
        cases = self.cases(identity)
        mods = self.active_modifiers(identity)
        consumed = [v for c in cases for v in c.get("consume_fields", [])]
        equivalent = [c for c in cases if c["kind"] == "operator_equivalence"]
        consumed += [{"id": q, "field": "value"} for c in equivalent for q in c["representations"]
                     if self.kinds[q] == "qualifiers"]
        consumed = list({(v["id"], v["field"]): v for v in consumed}.values())
        local = {"op": "typed_record", "record_type": kind, "id": identity}
        if kind == "events":
            local = {"op": "atom", "id": identity, **self.concept(identity),
                     "arguments": [{**a, "position": i, "canonical_role": self.aliases[identity]["argument_aliases"][i]["canonical_role"]}
                                   for i, a in enumerate(r["arguments"])], "tense": r["tense"], "mode": r["mode"]}
        formula_cases = [c for c in cases if c["kind"] == "local_formula"]
        modal = [c for c in cases if c["kind"] == "negative_modal_encoding"]
        if formula_cases:
            local = deepcopy(formula_cases[0]["formula"])
        elif modal:
            m = modal[0]
            operator = m["operator"]
            inner = {"op": "qualified_atom", "id": identity}
            if operator in {"not_possible", "not_able", "not_can_force_unspecified"}:
                local = {"op": "not", "body": {"op": operator.removeprefix("not_"), "body": inner}}
            elif operator == "unresolved_prohibition_or_inability":
                local = {"op": "unresolved_alternatives", "items": [{"op": "forbidden", "body": inner},
                         {"op": "not", "body": {"op": "able", "body": inner}}]}
            else:
                local = {"op": operator, "body": inner}
        elif kind == "events":
            # A polarity+quantifier ambiguity is explicitly unsimplified, never multiplied into two NOTs.
            if any(self.records[q].get("dimension") in {"quantification", "frequency", "modality"}
                   and {"id": q, "field": "value"} not in consumed for q in mods):
                local = {"op": "qualified_local_expression", "polarity": r["polarity"], "body": local,
                         "operator_order": "symbolic_unresolved_without_explicit_case"}
            elif r["polarity"] == "negative":
                local = {"op": "not", "body": local}
            elif r["polarity"] != "positive":
                local = {"op": "unresolved_polarity", "value": r["polarity"], "body": local}
        elif kind == "contexts":
            local = self.context_operator(identity)
        for case in cases:
            if case["kind"] == "structured_value_interpretation":
                local = {"op": case["coordination"], "items": [{"op": "expression_ref", "id": n} for n in case["members"]],
                         "raw_value": deepcopy(r["value"])}
        qualifications = []
        for qid in mods:
            q = self.records[qid]
            qualifications.append({"id": qid, "record_type": self.kinds[qid],
                "dimension": q.get("dimension", q.get("attribute")), "value": deepcopy(q["value"]),
                "operation": q.get("operation"), "support": q.get("support"),
                "context_paths": self.context_paths(qid),
                "consumed_in_formula": {"id": qid, "field": "value"} in consumed,
                "interpretation": "typed_symbolic_operator"})
        # An explicit operator-equivalence links representations instead of duplicating them.
        paths = self.context_paths(identity)
        context_ops = {c: self.context_operator(c) for p in paths for c in p}
        effective_paths = []
        for path in paths:
            seen, effective = set(), []
            for cid in path:
                canonical = cid
                while self.context_operator(canonical).get("same_operator_as"):
                    canonical = self.context_operator(canonical)["same_operator_as"]
                if canonical not in seen:
                    effective.append(canonical); seen.add(canonical)
            effective_paths.append(effective)
        endpoints = {}
        if kind == "relations":
            a = self.aliases[identity]
            endpoints = {"source": a["canonical_source"], "target": a["canonical_target"]}
        elif kind in {"properties", "qualifiers", "mentions"}:
            endpoints = {"target": r.get("target")}
        elif kind == "identity_links":
            endpoints = {"left": r["left"], "right": r["right"]}
        return {"id": identity, "record_type": kind, "as_of_token": self.endpoint,
                "introduced_at": self.availability[identity], "local": local, "qualifications": qualifications,
                "raw_context_paths": paths, "effective_context_paths": effective_paths, "context_operators": context_ops,
                "target_expression_context_paths": self.target_context_paths(r["target"]) if kind == "qualifiers" else [],
                "endpoint_expressions": {k: {"op": "expression_ref", "id": v, "as_of_token": self.endpoint,
                                            "record_type": self.kinds[v]} for k, v in endpoints.items() if v},
                "cases": deepcopy(cases), "consumed_fields": consumed,
                "operator_equivalences": deepcopy(equivalent),
                "uncertainty": self.uncertainty([identity]),
                "assertion_semantics": "scoped_textual_claim; no global world-truth or existence projection"}

    def uncertainty(self, ids, *, contextual=False):
        dependencies = set(i for i in ids if i)
        if contextual:
            for identity in list(dependencies):
                dependencies.update(c for p in self.context_paths(identity) for c in p)
                dependencies.update(self.active_modifiers(identity))
        found = {w["id"]: w for i in dependencies for w in self.warnings[i]}
        for identity in dependencies:
            for case in self.cases(identity):
                if case["kind"] == "assertion_eligibility" and case["status"] == "unresolved_commitment":
                    found[case["case_id"]] = {"id": case["case_id"], "origin": "scope_case", **case}
        return list(found.values())

    def scope_descriptor(self, identity):
        if identity in self._scope_cache:
            return self._scope_cache[identity]
        x = self.expression(identity)
        # Scope recovery must not reveal its answer through the event predicate/arguments.
        # Atom indices preserve co-reference within a formula without supplying its lexical content.
        atoms = {}
        def operator_shape(value):
            if isinstance(value, dict):
                if value.get("op") in {"atom", "qualified_atom", "expression_ref"}:
                    node = value.get("id", identity)
                    return {"op": value["op"], "atom_index": atoms.setdefault(node, len(atoms))}
                return {k: operator_shape(v) for k, v in value.items() if k not in {"id", "predicate", "sense", "arguments"}}
            if isinstance(value, list):
                return [operator_shape(v) for v in value]
            return self.abstract(value)
        result = {"local": operator_shape(x["local"]),
                "contexts": [[self.abstract(self.context_operator(c)) for c in p] for p in x["effective_context_paths"]],
                "qualifications": self.abstract(x["qualifications"])}
        if self.kinds[identity] in {"properties", "qualifiers"}:
            target = self.records[identity]["target"]
            result["target_contexts"] = [[self.abstract(self.context_operator(c)) for c in p]
                                         for p in self.target_context_paths(target)]
        if self.kinds[identity] == "relations":
            alias = self.aliases[identity]
            result["endpoint_scopes"] = {name: self.scope_descriptor(alias["canonical_" + name]) for name in ("source", "target")}
        self._scope_cache[identity] = result
        return result

    def state(self):
        return {"as_of_token": self.endpoint, "records": deepcopy(self.records), "kinds": self.kinds.copy(),
                "availability": deepcopy(self.availability), "identity_policy": "scoped edges only; no global quotient",
                "active_modifier_ids": {k: self.active_modifiers(k) for k in self.modifiers},
                "uncertainty": {k: deepcopy(v) for k, v in self.warnings.items() if v}}
