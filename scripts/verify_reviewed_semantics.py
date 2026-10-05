"""Full real-corpus invariants for the accepted, scoped downstream interface."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from neurosym.analysis_data import latest
from neurosym.analysis_runs import partition
from neurosym.io import object_hash, read_json, save_json
from neurosym.reviewed_archive import ReviewedArchive, RECORD_TYPES
from neurosym.reviewed_graph import ReviewedHistory, references
from neurosym.reviewed_queries import validate_reviewed_ast
from neurosym.semantic_features import SemanticDataset, SemanticVectorizer
from neurosym.semantics import digest_file
from neurosym.temporal import checked_alignment, fir_design


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def public_check(value, endpoint=None):
    if isinstance(value, dict):
        require(not set(value) & {"id", "graph", "evidence", "answer", "acceptable_indices", "source_evidence", "uncertainty"},
                "Private record field in public decoder input")
        if endpoint is not None and isinstance(value.get("anchor"), list):
            require(0 <= value["anchor"][0] < value["anchor"][1] <= endpoint, "Future/invalid candidate anchor")
        for v in value.values():
            public_check(v, endpoint)
    elif isinstance(value, list):
        for v in value:
            public_check(v, endpoint)
    else:
        require(not references(value), "Story-local graph identifier in decoder input")


def verify(build=None):
    config = read_json(Path(build) / "identity.json")["config"] if build is not None else read_json(ROOT / "configs/semantics.json")
    archive = ReviewedArchive(ROOT, config)
    dataset = SemanticDataset(Path(build) if build is not None else latest(ROOT / config["output"]))
    require(dataset.identity["reviewed_build_hash"] == archive.build_hash, "Wrong accepted archive")
    for name, digest in dataset.identity["code_hashes"].items():
        require(digest_file(ROOT / "neurosym" / name) == digest, "Compiler changed after compilation: " + name)
    totals, tasks, eligible, cases_checked, story_stats = Counter(), Counter(), Counter(), [], {}
    uncertainty_without_other_exclusion = 0
    evidence_before_availability = 0
    candidate_last_anchor = {}
    def max_anchor(value):
        if isinstance(value, list):
            return max((max_anchor(v) for v in value), default=0)
        if isinstance(value, dict):
            return max([value["anchor"][1] if isinstance(value.get("anchor"), list) else 0,
                        *[max_anchor(v) for v in value.values()]])
        return 0
    descriptor_anchors = {}
    require(dataset.candidate_sets.pooled, "Expected shared reviewed candidate storage")
    for key, candidate in dataset.candidate_sets.descriptors.items():
        require(object_hash(candidate) == key, "Candidate descriptor hash changed")
        public_check(candidate)
        descriptor_anchors[key] = max_anchor(candidate)
    for key, ids in dataset.candidate_sets.sets.items():
        require(object_hash(ids) == key and len(set(ids)) == len(ids) >= 2, "Invalid/duplicate candidate set")
        candidate_last_anchor[key] = max(descriptor_anchors[i] for i in ids)
    for sid in dataset.story_ids:
        history = ReviewedHistory(archive.constraints(sid))
        sources = {s["id"]: s for s in dataset.records(sid, "sources")}
        features = {s["source_id"]: s for s in dataset.records(sid, "features")}
        outputs = {(x["source_id"], x["id"]): x for x in dataset.records(sid, "expressions")}
        deltas = dataset.records(sid, "history")
        queries_by_source = {}
        for query in dataset.records(sid, "queries"):
            queries_by_source.setdefault(query["source_id"], []).append(query)
        question_ids = set()
        prior_state = None
        for saved, delta in zip(archive.by_story[sid], deltas, strict=True):
            source = sources[saved["unit_id"]]
            require(delta["graph"] == saved["graph"] and delta["graph_hash"] == saved["graph_hash"], "Lossless history changed")
            current = history.append(saved, archive.normalized)
            # Reconstruct allowed catalogs from THIS prefix, independently of the
            # persisted catalogs. Position checks alone miss future concept labels.
            targets = {object_hash(history.descriptor(i)) for i, k in history.kinds.items()
                       if k in {"entities", "events", "literals", "contexts"}}
            allowed = {
                "concept": {object_hash(history.concept(i)) for i, k in history.kinds.items() if k in {"entities", "events", "literals"}},
                "scope": {object_hash(history.scope_descriptor(i)) for i, k in history.kinds.items() if k == "events"},
                "relation": {object_hash({"label": a["canonical_type"]}) for a in history.aliases.values() if "canonical_type" in a},
                "polarity": {object_hash({"label": v}) for v in ("positive", "negative")},
                "identity": {object_hash({"label": v}) for v in ("same", "different", "possible")}}
            for query in queries_by_source.get(source["id"], []):
                ast = query["input"]["ast"]
                task = ast["task"]
                ids = set(dataset.candidate_sets.sets[query["input"]["candidate_set"]])
                if task in {"role", "reference", "compose"}:
                    expected = targets
                elif task in {"property", "qualifier"}:
                    field, kind = ("attribute", "properties") if task == "property" else ("dimension", "qualifiers")
                    expected = {object_hash({"value": history.abstract(r["value"])}) for i, r in history.records.items()
                                if history.kinds[i] == kind and r[field] == ast["operation"][field]}
                elif task == "binding":
                    for candidate in dataset.candidate_sets[query["input"]["candidate_set"]]:
                        require(all(object_hash(a["filler"]) in targets for a in candidate["assignment"]), "Future binding filler")
                    continue
                else:
                    expected = allowed[task]
                require(ids == expected, "Candidate catalog uses future or answer-conditioned information: " + query["id"])
            require(source["available_at_token"] == saved["available_at_token"] and
                    source["interpretation_availability"]["seconds"] == saved["available_at_seconds"], "Availability moved")
            require(source["text_exposure"]["word_span"] == source["local_word_span"], "Text exposure changed")
            for identity in current:
                record, kind = history.records[identity], history.kinds[identity]
                totals[kind] += 1
                expression = outputs[(source["id"], identity)]
                require(expression["introduced_at"] == history.availability[identity], "Observation time changed")
                require(expression["as_of_token"] == source["available_at_token"], "Expression assigned too early")
                require(set(expression["endpoint_expressions"][k]["id"] for k in expression["endpoint_expressions"]) <= set(history.records),
                        "Unavailable endpoint expression")
                evidence_before_availability += record["evidence"]["end_token"] < source["available_at_token"]
                if kind == "events":
                    aliases = history.aliases[identity]["argument_aliases"]
                    require([a["argument_index"] for a in aliases] == list(range(len(record["arguments"]))), "Lost ordered/repeated role")
                if kind in {"qualifiers", "properties"} and isinstance(record["value"], dict) and "interpretation" in record["value"]:
                    require(history.abstract(record["value"])["interpretation"] == record["value"]["interpretation"],
                            "Semantic interpretation in a qualifier payload was discarded")
                if kind == "relations":
                    alias = history.aliases[identity]
                    endpoints = [alias["canonical_source"], alias["canonical_target"]]
                    if alias["endpoints_reversed"]:
                        endpoints.reverse()
                    require(endpoints == [record["source"], record["target"]], "Nonreversible relation direction")
                if identity in archive.scope_bindings:
                    sidecar = archive.scope_bindings[identity]
                    require(expression["raw_context_paths"] == sidecar["context_paths_outer_to_inner"], "Context DAG sidecar mismatch")
                    require(expression["target_expression_context_paths"] == sidecar["qualifier_target_context_paths_outer_to_inner"],
                            "Qualifier escaped target scope: " + identity)
                for attachment in expression["qualifications"]:
                    require(attachment["value"] == history.records[attachment["id"]]["value"], "Qualifier value tree changed")
                    require(history.availability[attachment["id"]]["available_at_token"] <= source["available_at_token"], "Future modifier")
            for case in history.active_cases:
                if case["unit_id"] != source["id"]:
                    continue
                kind, target = case["kind"], case.get("target", case.get("new_record"))
                x = outputs[(source["id"], target)]
                if kind in {"local_formula", "context_formula"}:
                    require(x["local"] == case["formula"] if kind == "local_formula" else x["local"]["formula"] == case["formula"],
                            "Explicit reviewed formula not applied")
                elif kind == "negative_modal_encoding":
                    require(x["consumed_fields"] == case["consume_fields"], "Modal polarity consumed incorrectly")
                    if case["operator"].startswith("not_") and not any(c["kind"] == "local_formula" for c in x["cases"]):
                        require(x["local"]["op"] == "not" and x["local"]["body"]["body"]["op"] == "qualified_atom",
                                "Modal double negation")
                elif kind == "context_continuation":
                    require(x["local"]["same_operator_as"] == case["same_operator_as"], "Continued report duplicated")
                elif kind == "operator_equivalence":
                    mods = [q for q in x["qualifications"] if q["id"] in case["representations"]]
                    require(mods and all(q["consumed_in_formula"] for q in mods), "Redundant possibility counted twice")
                elif kind == "structured_value_interpretation":
                    require(x["local"]["op"] == "and", "Conjoined goals converted to disjunction")
                elif kind == "compound_condition":
                    require(x["local"]["formula"]["antecedent"]["op"] == "and", "Independent sufficient conditions invented")
                elif kind == "prospective_qualification_revision":
                    require(prior_state is not None, "Missing earlier prefix")
                    old_target = history.records[case["supersedes"]]["target"]
                    require(case["supersedes"] in prior_state["active_modifier_ids"][old_target], "Earlier annotation retroactively revised")
                    require(case["supersedes"] not in history.active_modifiers(old_target), "Later update not applied")
                elif kind == "assertion_eligibility":
                    require(history.uncertainty([target]), "Unresolved commitment lost")
                elif kind == "context_attribution_semantics":
                    require(x["local"]["operator"] == case["operator"] and x["local"]["projection"] == case["projection"],
                            "Attribution/factivity scope lost")
                cases_checked.append(case["case_id"])
            # A compact earlier-prefix snapshot suffices to test the real prospective revision.
            prior_state = {"active_modifier_ids": {i: history.active_modifiers(i) for i in history.modifiers}}
            for q in archive.questions:
                if q["unit_id"] == source["id"]:
                    question_ids.add(q["issue_id"])
                    require(all(any(w["id"] == q["issue_id"] for w in history.warnings[i]) for i in q["affected_ids"]),
                            "Review uncertainty was not attached to its named targets")
        require(len(history.records) == sum(len(r["graph"][k]) for r in archive.by_story[sid] for k in RECORD_TYPES), "Record loss")
        answers = {a["query_id"]: a for a in dataset.records(sid, "answers")}
        oracle = {a["query_id"]: a for a in dataset.records(sid, "oracle")}
        by_source = Counter()
        for query in dataset.records(sid, "queries"):
            source, answer = sources[query["source_id"]], answers[query["id"]]
            ast = query["input"]["ast"]
            validate_reviewed_ast(ast); public_check(ast, source["available_at_token"])
            key = query["input"]["candidate_set"]
            require(candidate_last_anchor[key] <= source["available_at_token"], "Future candidate exposure")
            require(query["ast_hash"] == object_hash(ast), "AST digest changed")
            require(oracle[query["id"]]["available_at_token"] == source["available_at_token"], "Answer availability changed")
            require(not answer["scoring_eligible"] or source["decoder_timing_eligible"], "Unmeasured target scored")
            require(not answer["review_flags"] or not answer["acceptable_indices"], "Uncertainty turned into a target")
            require(not answer["scoring_eligible"] or 0 < len(answer["acceptable_indices"]) < len(dataset.candidate_sets[key]), "Invalid scored choice")
            tasks[ast["task"]] += 1
            if answer["scoring_eligible"]:
                eligible[ast["task"]] += 1
                by_source[source["id"]] += 1
        uncertain_sources = {s["source_id"] for s in dataset.records(sid, "review")}
        uncertainty_without_other_exclusion += sum(by_source[s] > 0 for s in uncertain_sources)
        # Null endpoint seconds remain unavailable; BOLD is delayed, not an online prefix observation.
        timing = read_json(dataset.build / "stories" / sid / "timing.json")
        alignment = checked_alignment(ROOT / config["alignment"], sid, archive.index["content_hash"])
        for source in sources.values():
            slot = source["raw_feature_bin"]
            require((source["available_at_seconds"] is None) == (slot == -1), "Unknown timing imputed")
            require(source["decoder_raw_response_rows"] == ([slot + d for d in timing["fir_delays_trs"]] if slot >= 0 else []), "Wrong FIR measurement clock")
            require(not source["decoder_timing_eligible"] or all(t > source["available_at_seconds"] for t in source["fmri_measurement"]["sample_story_seconds"]),
                    "Neural measurement not later than interpretation")
        indices = np.asarray(timing["response_raw_indices"])
        valid = np.ones(len(indices), dtype=bool)
        common = np.asarray(timing["comparison_known_raw_rows"], dtype=bool)
        for d in timing["fir_delays_trs"]:
            valid &= common[indices - d] & np.asarray(alignment["words"]["known_raw_rows"])[indices - d]
        require(valid.any(), "All real encoding rows excluded: " + sid)
        story_stats[sid] = {"units": len(sources), "queries": len(answers), "encoding_comparison_rows": int(valid.sum()),
                           "trimmed_response_rows": len(valid), "unresolved_endpoint_units": sum(s["available_at_seconds"] is None for s in sources.values()),
                           "review_questions_bound": len(question_ids)}
        # Free large expression lists between stories. Dataset caches remain useful for vocabulary verification.
        dataset._cache.pop((sid, "expressions"), None)
    require(len(cases_checked) == len(archive.cases), "Missing scope cases")
    require(sum(s["units"] for s in story_stats.values()) == 1217, "Incomplete corpus")
    require(sum(s["unresolved_endpoint_units"] for s in story_stats.values()) == 4, "Unknown endpoint count changed")
    require(sum(s["review_questions_bound"] for s in story_stats.values()) == 29, "Missing review questions")
    require(uncertainty_without_other_exclusion > 0, "Uncertainty excluded entire units")
    earlier = dataset.state_at("story_06", "story_06_u0043")
    later = dataset.state_at("story_06", "story_06_u0044")
    require("story_06:u0044_qua0" not in earlier["records"] and "story_06:u0044_qua0" in later["records"], "Prefix state API leaked later clarification")
    require(dataset.state_at("story_06", "story_06_u0043") == earlier, "Reading later state mutated earlier state")
    require(dataset.splits["heldout"] == ["story_11"], "Final story was changed")
    fold_reports = []
    for fold in [str(i) for i in range(len(dataset.splits["outer_folds"]))] + ["final"]:
        split = partition(dataset, fold)
        for inner in split["inner"]:
            require(set(inner["train"]).isdisjoint(inner["validation"]) and
                    set(inner["train"] + inner["validation"]) <= set(split["train"]), "Nested-fold leakage")
        require("story_11" not in split["train"], "Heldout story used for fitting")
        vectorizer = SemanticVectorizer.fit(dataset, split["train"])
        training_keys = {k for sid in split["train"] for f in dataset.records(sid, "features") for k in f["terms"]}
        require(set(vectorizer.vocabulary) <= training_keys, "Vocabulary used test observations")
        fold_reports.append({"fold": fold, "train": split["train"], "test": split["test"], "columns": len(vectorizer.vocabulary)})
    try:
        SemanticVectorizer.fit(dataset, ["story_11"])
    except ValueError:
        pass
    else:
        raise AssertionError("Heldout vocabulary fitting was allowed")
    return {"status": "verified", "semantic_build_hash": dataset.build_hash, "reviewed_build_hash": archive.build_hash,
            "verification_script_sha256": digest_file(Path(__file__)),
            "archive_files_verified_unchanged": archive.verify_files(), "records": dict(totals), "stories": story_stats,
            "queries_by_task": dict(tasks), "eligible_queries_by_task": dict(eligible), "explicit_scope_cases_checked": cases_checked,
            "evidence_before_interpretation_records": evidence_before_availability,
            "uncertain_units_retaining_other_scored_queries": uncertainty_without_other_exclusion,
            "folds": fold_reports, "annotation_review_closed": True, "scientific_fits_executed": False,
            "checks": ["accepted archive hashes", "all 1217 real graph deltas", "all ordered roles and relation reversals",
                       "scope DAG and qualifier target dependencies", "explicit modal/conditional/factivity/continuation cases",
                       "prospective updates", "targeted uncertainties", "public-query separation and prefix anchors",
                       "three distinct clocks and four unresolved endpoints", "story-disjoint nested folds and story 11 exclusion"]}


if __name__ == "__main__":
    result = verify()
    destination = ROOT / "artifacts/reviewed-semantics-verification.json"
    save_json(destination, result)
    print(json.dumps(result, indent=2))
    print("REPORT:", destination)
