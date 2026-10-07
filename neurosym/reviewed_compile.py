"""Compile the accepted review into lossless expressions, features, queries and timing."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

from .extraction_inputs import story_folds
from .io import object_hash, read_json, save_json
from .reviewed_archive import MODULES, RECORD_TYPES, ReviewedArchive
from .reviewed_graph import ReviewedHistory
from .reviewed_queries import compile_reviewed_queries
from .semantics import digest_file, verify_build, write_jsonl, compilation_report_html
from .temporal import checked_alignment

GROUPS = ("L", "C", "BC", "B", "BR", "GB", "GBR", "PB", "PBR", "R", "S", "D", "U")


def source_record(story, unit, unit_index, saved, alignment):
    slot = alignment["units"]["bin_indices"][unit_index]
    raw = [slot + lag for lag in alignment["fir_delays_trs"]] if slot >= 0 else []
    indices = alignment["response_raw_indices"]
    clock = dict(zip(indices, alignment["tr_centers_story_seconds"], strict=True))
    return {"id": unit["id"], "story_id": story["story_id"], "unit_index": unit_index,
            "annotation_hash": saved["graph_hash"], "annotation_review_status": "accepted_reviewed",
            "annotation_protocol": "accepted-independent-review-v1", "local_word_span": [unit["start_token"], unit["end_token"]],
            "prefix_word_span": [0, unit["end_token"]], "available_at_token": unit["end_token"],
            "available_at_seconds": saved["available_at_seconds"], "endpoint_timing_resolved": unit["endpoint_timing_resolved"],
            "timing_counts": unit["timing_counts"], "model_unit_row": unit_index, "raw_feature_bin": slot,
            "decoder_raw_response_rows": raw, "decoder_trimmed_response_rows": [indices.index(r) for r in raw if r in clock],
            "decoder_timing_eligible": bool(raw) and all(r in clock for r in raw), "configurations": [],
            "text_exposure": {"word_span": [unit["start_token"], unit["end_token"]],
                              "onset_seconds": unit["onset_seconds"], "endpoint_seconds": unit["offset_seconds"],
                              "word_alignment_source": "unchanged corpus words; no imputed unknown times"},
            "interpretation_availability": {"token": saved["available_at_token"], "seconds": saved["available_at_seconds"],
                                            "rule": "full annotation-unit endpoint, not evidence trigger"},
            "fmri_measurement": {"raw_rows": raw, "sample_story_seconds": [clock.get(r) for r in raw],
                                 "fir_delays_trs": alignment["fir_delays_trs"],
                                 "interpretation": "delayed mixed BOLD; later presented text can contribute"},
            "neural_interpretation": "Offline delayed samples can contain responses to later presented text."}


def compile_features(history, current, source):
    definitions, terms, occurrences, evidence = {}, Counter(), [], []
    unknown_groups = set()
    def emit(group, parts, nodes, *, route="factorized", uncertain=()):
        definition = {"group": group, "parts": parts, "route": route}
        key = object_hash(definition)
        definitions[key] = definition
        if uncertain:
            unknown_groups.add(group)
        else:
            terms[key] += 1
        evidence.append({"feature": key, "nodes": list(nodes), "uncertainty_ids": [u["id"] for u in uncertain],
                         "included": not bool(uncertain), "available_at_token": source["available_at_token"]})
    def occur(kind, label, node, *, resolved=True, view="reviewed_scoped", **extra):
        item = {"source_id": source["id"], "story_id": source["story_id"], "unit_index": source["unit_index"],
                "kind": kind, "label": label, "node": node, "evidence": history.records[node]["evidence"], "view": view,
                "available_at_token": source["available_at_token"], "available_at_seconds": source["available_at_seconds"],
                "annotation_review_status": "accepted_reviewed", "grounding_resolved": resolved,
                "expression_ref": {"id": node, "as_of_token": source["available_at_token"]}, **extra}
        item.setdefault("scope", history.scope_descriptor(node))
        item["id"] = object_hash(item); occurrences.append(item)
    for identity in current:
        r, kind = history.records[identity], history.kinds[identity]
        concept = history.concept(identity)
        if kind in {"entities", "events", "literals"}:
            emit("C", [kind, concept], [identity])
            occur({"entities": "concept", "events": "predicate", "literals": "literal"}[kind], concept, identity,
                  view="textual_content_not_world_existence")
        if kind == "mentions":
            emit("L", ["mention", r["form"], r["evidence"]["verbatim"].casefold()], [identity])
            if r.get("target"):
                warnings = history.uncertainty([identity])
                if r.get("alternatives"):
                    warnings += [{"id": identity + ":alternatives"}]
                target = r["target"]
                label = {"form": r["form"], "target": history.concept(target),
                         "antecedent_supplied": r.get("antecedent") is not None}
                emit("R", ["reference", label], [identity, target], uncertain=warnings)
                occur("reference", label, identity, resolved=not warnings,
                      grounding_query={"task": "reference", "anchor": history.selector(identity, content=False),
                                       "filler": history.descriptor(target), "operation": {"op": "reference"}})
                occur({"entities": "concept", "events": "predicate", "literals": "literal", "contexts": "scope"}[history.kinds[target]],
                      history.concept(target), identity, resolved=not warnings, view="reference_resolved", scope=history.scope_descriptor(target))
            else:
                unknown_groups.add("R")
        elif kind == "events":
            warnings = history.uncertainty([identity])
            scope_warnings = history.uncertainty([identity], contextual=True)
            scope = history.scope_descriptor(identity)
            emit("S", ["scope", scope], [identity], uncertain=scope_warnings)
            occur("scope", scope, identity, resolved=not scope_warnings)
            for path in scope["contexts"]:
                emit("S", ["operator_path", [{k: c[k] for k in ("kind", "operator_family", "operator", "projection") if k in c} for c in path]],
                     [identity], uncertain=scope_warnings)
            emit("S", ["polarity_mode_tense", r["polarity"], r["mode"], r["tense"]], [identity], uncertain=warnings)
            ordered = []
            for position, a in enumerate(r["arguments"]):
                role = history.aliases[identity]["argument_aliases"][position]["canonical_role"]
                target = a["target"]
                reference = bool(target and history.availability[target]["available_at_token"] < source["available_at_token"])
                if a.get("mention"):
                    reference |= history.records[a["mention"]].get("form") in {"pronoun", "relative", "deictic", "demonstrative"}
                group = "BR" if reference else "B"
                deps = [identity] + ([a["mention"]] if a.get("mention") else [])
                uncertain = history.uncertainty(deps)
                if target is None:
                    uncertain += [{"id": identity + ":null_argument:" + str(position)}]
                filler = history.concept(target) if target else {"unknown": True}
                ordered.append({"role": role, "position": position, "filler": filler})
                # Marginals of the SAME argument incidences as PB/PBR. No atom
                # binds a predicate, role and filler together. Counts and the
                # reference channel are retained, so these cannot explain the
                # gain from adding their categorical conjunctions.
                for atom in (["predicate", concept, reference], ["role", role, reference],
                             ["filler", filler, reference]):
                    emit("BC", atom, deps, uncertain=uncertain)
                emit("B", ["predicate_role", concept, role], [identity], uncertain=warnings)
                emit(group, ["typed_binding", concept, role, filler], deps, uncertain=uncertain)
                emit(group, ["scoped_binding", concept, role, filler, scope], deps, uncertain=scope_warnings + uncertain)
                emit("GBR" if reference else "GB", ["role_filler", role, filler], deps, uncertain=uncertain)
                emit("PBR" if reference else "PB", ["predicate_role_filler", concept, role, filler], deps, uncertain=uncertain)
                if target:
                    occur("role", {"predicate": concept, "role": role, "filler": filler}, identity, resolved=not uncertain,
                          position=position, scope=scope,
                          grounding_query={"task": "role", "anchor": history.selector(identity), "filler": history.descriptor(target),
                                           "operation": {"op": "role", "role": role, "position": position}})
            signature = {"predicate": concept, "arguments": ordered, "scope": scope}
            resolved = not history.uncertainty([identity] + [a["mention"] for a in r["arguments"] if a.get("mention")], contextual=True) and all(a["target"] for a in r["arguments"])
            emit("B", ["configuration", signature], [identity], route="exact_configuration",
                 uncertain=[] if resolved else [{"id": identity + ":unresolved_configuration"}])
            source["configurations"].append({"key": object_hash(signature), "definition": signature, "event_id": identity,
                "selector": history.selector(identity), "grounding_resolved": resolved,
                "constituents": sorted(set([object_hash(concept)] + [object_hash({"role": a["role"]}) for a in ordered] +
                                           [object_hash(a["filler"]) for a in ordered]))})
            occur("configuration", signature, identity, resolved=resolved)
        elif kind == "relations":
            a = history.aliases[identity]
            left, right = a["canonical_source"], a["canonical_target"]
            warnings = history.uncertainty([identity], contextual=True)
            label = {"relation": a["canonical_type"], "source": history.concept(left), "target": history.concept(right)}
            joint = [c for c in history.cases(identity) if c["kind"] == "compound_condition"]
            if joint:
                label["joint_condition"] = history.abstract(history.context_operator(joint[0]["target"])["formula"])
                label["independently_sufficient"] = False
            emit("D", ["relation_type", a["canonical_type"]], [identity], uncertain=warnings)
            emit("D", ["directed_relation", label], [identity, left, right], uncertain=warnings)
            emit("D", ["scoped_relation", label, history.scope_descriptor(identity)], [identity, left, right], uncertain=warnings)
            # Typed endpoint expressions stay in the symbolic DAG; do not inherit roles through correspondence.
            occur("discourse", label, identity, resolved=not warnings,
                  grounding_query={"task": "relation", "anchor": history.selector(left), "filler": history.selector(right),
                      "operation": {"op": "event_relation", "relation": a["canonical_type"], "direction": "out"}})
        elif kind == "identity_links":
            warnings = history.uncertainty([identity], contextual=True)
            label = {"relation": r["relation"], "left": history.concept(r["left"]), "right": history.concept(r["right"])}
            emit("R", ["scoped_identity", label, history.abstract(history.context_paths(identity))], [identity], uncertain=warnings)
            occur("identity", label, identity, resolved=not warnings,
                  grounding_query={"task": "identity", "anchor": history.selector(r["left"]), "filler": history.selector(r["right"]),
                                   "operation": {"op": "identity", "relation": r["relation"], "direction": "out"}})
        elif kind in {"qualifiers", "properties"}:
            warnings = history.uncertainty([identity], contextual=True)
            label = {"target": history.concept(r["target"]), "dimension": r.get("dimension", r.get("attribute")),
                     "value": history.abstract(r["value"]), "operation": r.get("operation"),
                     "target_scope": history.abstract(history.target_context_paths(r["target"])),
                     "attachment_scope": history.abstract(history.context_paths(identity))}
            emit("S" if kind == "qualifiers" else "U", [kind, label], [identity, r["target"]], uncertain=warnings)
            emit("S" if kind == "qualifiers" else "U", ["attachment_dimension", kind, label["dimension"], r.get("operation")],
                 [identity], uncertain=warnings)
            occur("qualification" if kind == "qualifiers" else "state_update", label, identity, resolved=not warnings,
                  grounding_query={"task": "qualifier" if kind == "qualifiers" else "property", "anchor": history.selector(r["target"]),
                                   "filler": {"value": history.abstract(r["value"])},
                                   "operation": {"op": "attached_value", "dimension" if kind == "qualifiers" else "attribute": label["dimension"]}})
    return {"source_id": source["id"], "terms": dict(terms), "evidence": evidence,
            "uncertain_groups": sorted(unknown_groups)}, definitions, occurrences


def attach_latent_queries(history, occurrences, queries):
    """Link each item to its own public retrieval query, never to its neighbours.

    Links are analysis provenance only; no target or answer is added to decoder
    inputs. Missing alternatives/queries stay missing, with no passage fallback.
    """
    lookup = defaultdict(list)
    for query in queries:
        ast = query["input"]["ast"]
        lookup[object_hash([ast["task"], ast["anchor"], ast["operation"]])].append(query["id"])
    tasks = {"concept": "concept", "predicate": "concept", "literal": "concept",
             "scope": "scope", "role": "role", "reference": "reference",
             "configuration": "binding", "discourse": "relation", "identity": "identity",
             "qualification": "qualifier", "state_update": "property"}
    for record in occurrences:
        node, kind = record["node"], record["kind"]
        task = "reference" if record["view"] == "reference_resolved" else tasks[kind]
        if task in {"concept", "reference"}:
            anchor = history.selector(node, content=False)
            operation = {"op": "concept_label" if task == "concept" else "reference"}
        elif task in {"scope", "binding"}:
            anchor = history.selector(node)
            operation = {"op": "scoped_expression" if task == "scope" else "ordered_role_assignment"}
        elif task in {"relation", "identity"}:
            grounding = record["grounding_query"]
            anchor = {"source": grounding["anchor"], "target": grounding["filler"]}
            operation = ({"op": "relation_type", "direction": "source_to_target"} if task == "relation"
                         else {"op": "scoped_identity"})
        else:
            grounding = record["grounding_query"]
            anchor, operation = grounding["anchor"], grounding["operation"]
        record["latent_query_link"] = {
            "policy": "same_occurrence_public_query_v1", "task": task,
            "query_ids": sorted(set(lookup.get(object_hash([task, anchor, operation]), []))),
            "selection": "annotation occurrence and query identity; independent of prediction correctness"}
        record["id"] = object_hash({k: v for k, v in record.items() if k != "id"})


def compile_reviewed(root, config_path):
    root, config = Path(root).resolve(), read_json(config_path)
    archive = ReviewedArchive(root, config)
    identity = {"format_version": 2, "semantic_interface": "reviewed-scoped-expressions-v1", "config": config,
                "corpus_hash": archive.index["content_hash"], "coverage": archive.coverage,
                "reviewed_build_hash": archive.build_hash, "annotation_run_hash": archive.build_hash,
                "reviewed_checksums_sha256": archive.snapshot["checksums_sha256"],
                "alignment_hash": read_json(root / config["alignment"] / "index.json")["content_hash"],
                "code_hashes": {n: digest_file(Path(__file__).parent / n) for n in (*MODULES, "semantics.py", "semantic_features.py", "temporal.py", "extraction_inputs.py", "io.py")},
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pydantic")}}
    build_hash = object_hash(identity)
    output = root / config["output"]
    if not output.resolve().is_relative_to(root) or output.resolve() != output:
        raise ValueError("Reviewed compiler destination must stay inside the workspace.")
    output.mkdir(parents=True, exist_ok=True)
    build = output / "builds" / build_hash
    if (build / "complete.json").exists():
        verify_build(build)
        save_json(output / "latest.json", {"build_hash": build_hash, "path": "builds/" + build_hash})
        return build
    staging = Path(tempfile.mkdtemp(prefix=".compiling-", dir=output))
    save_json(staging / "identity.json", identity)
    save_json(staging / "splits.json", story_folds(archive.index))
    # Exact accepted annotations/constraints are copied as read-only provenance, never modified in place.
    files = ["snapshot.json", "checksums.json", "manifest.json", "verification.json", "annotations.jsonl", "source-only.jsonl",
             "unresolved.jsonl", "uncertainties.jsonl", "question-bindings.json", "interface/definitions.json", "interface/scope-cases.json",
             "interface/modal-scope-bindings.jsonl", "interface/scope-bindings.jsonl", "interface/relation-orientations.jsonl",
             "interface/normalized-records.jsonl"]
    for name in files:
        dest = staging / "accepted_review" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive.path / name, dest)
    definitions, candidate_sets, candidate_descriptors, catalog = {}, {}, {}, {}
    totals, query_counts, skips = Counter(), defaultdict(Counter), Counter()
    for entry in archive.index["stories"]:
        sid = entry["id"]
        story, constraints = archive.stories[sid], archive.constraints(sid)
        history = ReviewedHistory(constraints)
        alignment = checked_alignment(root / config["alignment"], sid, archive.index["content_hash"])
        result = {k: [] for k in ("sources", "features", "queries", "answers", "oracle", "occurrences", "review", "history", "expressions")}
        for i, saved in enumerate(archive.by_story[sid]):
            current = history.append(saved, archive.normalized)
            source = source_record(story, story["units"][i], i, saved, alignment)
            feat, defs, occ = compile_features(history, current, source)
            qs, ans, private, skipped = compile_reviewed_queries(history, current, source, config["candidate_seed"])
            attach_latent_queries(history, occ, qs)
            definitions.update(defs); skips.update(skipped)
            for q in qs:
                candidates = q["input"].pop("candidates")
                key = q["input"]["candidate_set"]
                if key not in candidate_sets:
                    ids = [object_hash(c) for c in candidates]
                    if object_hash(ids) != key:
                        raise ValueError("Candidate catalog identity mismatch.")
                    candidate_sets[key] = ids
                    candidate_descriptors.update(zip(ids, candidates, strict=True))
            for a in ans:
                query_counts[a["family"]].update({"total": 1, a["label_state"]: 1,
                    "scoring_eligible_against_annotations": int(a["scoring_eligible"]), "diagnostic_only": 0})
            # Rebuild only newly introduced expressions and targets changed NOW by new modifiers.
            affected = set(current)
            affected.update(history.records[n]["target"] for n in current if history.kinds[n] in {"qualifiers", "properties"})
            expressions = [{"source_id": source["id"], **history.expression(n)} for n in sorted(affected)]
            result["sources"].append(source); result["features"].append(feat)
            result["queries"].extend(qs); result["answers"].extend(ans); result["oracle"].extend(private)
            result["occurrences"].extend(occ); result["expressions"].extend(expressions)
            result["history"].append({"source_id": source["id"], "graph": saved["graph"], "graph_hash": saved["graph_hash"],
                                      "available_at_token": saved["available_at_token"], "available_at_seconds": saved["available_at_seconds"],
                                      "aliases": {n: history.aliases[n] for n in current}})
            result["review"].extend({"source_id": source["id"], "record": history.records[n]} for n in current if history.kinds[n] == "uncertainties")
            result["review"].extend({"source_id": source["id"], "origin": "accepted_review_question", "record": q}
                                    for q in constraints["questions"] if q["unit_id"] == source["id"])
            totals.update({"units": 1, "queries": len(qs), "occurrences": len(occ), "expressions": len(expressions),
                           **{k: len(saved["graph"][k]) for k in RECORD_TYPES}})
            for occurrence in occ:
                key = object_hash({"kind": occurrence["kind"], "label": occurrence["label"], "view": occurrence["view"]})
                item = catalog.setdefault(key, {"kind": occurrence["kind"], "label": occurrence["label"], "view": occurrence["view"],
                                                "occurrences": 0, "source_ids": set(), "story_ids": set()})
                item["occurrences"] += 1; item["source_ids"].add(source["id"]); item["story_ids"].add(sid)
        directory = staging / "stories" / sid
        for name, values in result.items():
            write_jsonl(directory / (name + ".jsonl"), values)
        save_json(directory / "constraints.json", constraints)
        save_json(directory / "state.json", history.state())
        masks = {g: list(alignment["units"]["known_raw_rows"]) for g in GROUPS}
        for source, feat in zip(result["sources"], result["features"], strict=True):
            slot = source["raw_feature_bin"]
            if slot >= 0:
                for group in feat["uncertain_groups"]:
                    masks[group][slot] = False
        # A common comparison mask keeps every semantic/model ablation on the same measurements.
        common = [all(masks[g][i] for g in GROUPS) for i in range(alignment["raw_response_rows"])]
        save_json(directory / "timing.json", {"story_id": sid, "raw_response_rows": alignment["raw_response_rows"],
                  "response_raw_indices": alignment["response_raw_indices"], "fir_delays_trs": alignment["fir_delays_trs"],
                  "known_raw_rows": list(alignment["units"]["known_raw_rows"]), "group_known_raw_rows": masks,
                  "comparison_known_raw_rows": common, "coverage_complete": True, "alignment_hash": alignment["content_hash"]})
        print("COMPILED REVIEWED", sid, len(result["sources"]), "units;", len(result["queries"]), "queries", flush=True)
    for c in catalog.values():
        c["source_ids"], c["story_ids"] = sorted(c["source_ids"]), sorted(c["story_ids"])
        c["distinct_source_units"], c["stories"] = len(c["source_ids"]), len(c["story_ids"])
    save_json(staging / "feature_catalog.json", definitions)
    save_json(staging / "candidate_sets.json", {"format_version": 2, "descriptors": candidate_descriptors, "sets": candidate_sets})
    save_json(staging / "occurrence_catalog.json", catalog)
    report = {"format_version": 2, "build_hash": build_hash, "reviewed_build_hash": archive.build_hash,
              "compilation_status": "complete", "corpus_coverage_complete": True, "semantic_accuracy_measured": False,
              "human_validation": False, "annotation_review_closed": True, "accepted_by_researcher": True,
              "neural_fits_executed": False, "llm_calls": 0, "counts": dict(totals), "expected_units": len(archive.records),
              "stories": archive.coverage, "queries": {k: dict(v) for k, v in query_counts.items()}, "query_skips": dict(skips),
              "feature_groups": dict(Counter(d["group"] for d in definitions.values())),
              "feature_routes": dict(Counter(d["route"] for d in definitions.values())),
              "annotation_protocols": {"accepted-independent-review-v1": 1217},
              "uncertainty_policy": "Targeted uncertainty retained; affected targets withheld, no unit-wide decoder exclusion or false negatives.",
              "scope_policy": "Typed symbolic expressions; explicit reviewed formulas precede polarity, full qualifiers retained, no world projection.",
              "timing_policy": "Text exposure, unit-end interpretation availability and delayed BOLD samples are separate.",
              "archive_files_verified": archive.verify_files()}
    save_json(staging / "report.json", report)
    (staging / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Accepted semantic compilation</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>Accepted semantic compilation</h1><p>Review closed; the accepted archive is preserved. '
        'This report verifies compilation, not annotation accuracy or scientific brain results.</p>'
        '<p>Features are scoped annotation updates at unit endpoints. Decoder choices recover recorded content and structure; '
        'missing facts are never negatives. All raw records, uncertainty, context paths and qualifier payloads remain linked.</p>'
        '<p>Text exposure, interpretation availability and delayed fMRI measurements are distinct. '
        'All vocabulary, scaling, projection and readout fitting belongs inside training folds; story 11 remains held out.</p>'
        '<pre>' + html.escape(json.dumps(report, indent=2, ensure_ascii=False)) + '</pre>', encoding="utf-8")
    artifacts = {p.relative_to(staging).as_posix(): digest_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    save_json(staging / "complete.json", {"build_hash": build_hash, "artifact_hashes": artifacts,
                                          "completed_utc": datetime.now(timezone.utc).isoformat()})
    build.parent.mkdir(parents=True, exist_ok=True)
    if build.exists():
        raise ValueError("Immutable output already exists; retained staging for inspection.")
    os.replace(staging, build)
    save_json(output / "latest.json", {"build_hash": build_hash, "path": "builds/" + build_hash})
    return build
