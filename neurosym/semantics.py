"""Immutable, deterministic compilation of a snapshot of accepted real annotations."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from .extraction_inputs import checked_index, checked_story, story_folds
from .graphs import GraphDelta, GraphHistory
from .io import object_hash, read_json, save_json
from .semantic_records import compile_unit, state_view
from .semantic_queries import compile_queries
from .temporal import checked_alignment


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def capture_inputs(root, config):
    corpus = root / config["corpus"]
    index = checked_index(corpus)
    annotation_root = root / config["annotations"]
    run = read_json(annotation_root / "run.json")
    if run["corpus_hash"] != index["content_hash"]:
        raise ValueError("Annotations and compiler corpus differ.")
    if config["annotation_pass"] not in run["configuration"]["passes"]:
        raise ValueError("Unknown annotation pass.")
    run_hash = object_hash(run)
    captured, files, coverage = {}, {}, []
    for entry in index["stories"]:
        story = checked_story(corpus, entry)
        records, gap, missing, blocked = [], False, [], []
        for unit in story["units"]:
            path = annotation_root / config["annotation_pass"] / story["story_id"] / (unit["id"] + ".json")
            if not path.is_file():
                gap = True
                missing.append(unit["id"])
                continue
            if gap:
                blocked.append(unit["id"])
                continue
            if path.is_symlink():
                raise ValueError("Annotation record must not be a symlink.")
            data = path.read_bytes()
            saved = json.loads(data)
            identity = saved["input_identity"]
            if (saved["input_hash"] != object_hash(identity) or saved["graph_hash"] != object_hash(saved["graph"])
                    or identity["run_hash"] != run_hash or identity["story_hash"] != entry["content_hash"]
                    or identity["unit"] != unit or identity["pass"] != config["annotation_pass"]
                    or saved["available_at_token"] != unit["end_token"]
                    or saved["available_at_seconds"] != unit["offset_seconds"]):
                raise ValueError("Annotation identity/timing mismatch: " + str(path))
            files[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
            records.append(saved)
        captured[entry["id"]] = (story, records)
        coverage.append({"story_id": entry["id"], "split": entry["split"], "total_units": len(story["units"]),
                         "compiled_units": len(records), "missing_units": missing, "blocked_after_gap": blocked,
                         "coverage_complete": len(records) == len(story["units"])})
    if not files:
        raise ValueError("No contiguous accepted annotation records exist. Finish/save real annotation units first.")
    return index, captured, files, coverage, run_hash


def compilation_report_html(report):
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(item[k])) + "</td>" for k in
                        ("story_id", "compiled_units", "total_units", "coverage_complete")) + "</tr>" for item in report["stories"])
    families = "".join("<tr><td>" + html.escape(name) + "</td><td>" + html.escape(json.dumps(counts, sort_keys=True)) + "</td></tr>"
                       for name, counts in sorted(report["queries"].items()))
    return """<!doctype html><html lang="en"><meta charset="utf-8"><title>Semantic compilation report</title>
<style>body{max-width:1050px;margin:40px auto;font:16px/1.5 system-ui;padding:0 20px;color:#172536}
table{border-collapse:collapse;width:100%;margin:20px 0}td,th{text-align:left;border-bottom:1px solid #ccd5df;padding:9px}
code,pre{background:#f0f3f7;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:28px}</style>
<h1>Semantic compilation report</h1><p>Deterministic compilation of real saved annotations. No LLM calls or neural fits.</p>
<p>Compilation: complete. Corpus coverage: """ + ("complete" if report["corpus_coverage_complete"] else "incomplete") + """.
Annotation semantic accuracy has not been measured by this compiler.</p><h2>Story coverage</h2>
<table><tr><th>Story</th><th>Compiled units</th><th>Total units</th><th>Full coverage</th></tr>""" + rows + """</table>
<h2>Query coverage</h2><p>Scores against annotation targets remain distinct from human-reviewed accuracy.
Concept-presence positives alone are diagnostic; missing graph facts never generate false labels.
Binding polarity contrasts and role-swap candidates are separate variants.</p><table>""" + families + """</table>
<h2>Representation definitions</h2><p>L: surface mentions; C: locally attested concepts/predicates;
B: local role binding; BR: binding requiring reference; R: reference; S: status and scope;
D: ordered discourse links; U: state updates. Exact conjunctions are distinct from shared factors.</p>
<p>Features are annotation update counts at unit endpoints, not persistent counts of every known entity.
Missing and uncompiled support is masked. The concept catalog describes coverage; numerical vocabularies are fitted on training stories only.</p>
<h2>Interpretation and review</h2><p>Candidate descriptors use concepts, names where explicitly available,
and mention/event selectors. They can carry priors: query-only comparisons must receive identical candidates.
No full graph, answer indices, evidence, or review flags enter the decoder-input interface.</p>
<p>Review queues include causal claims, unresolved references, unspecified senses, and explicit annotation uncertainty.
These are review requests, not automatic semantic corrections. Full review records are in each story's review.jsonl.</p>
<h2>Machine-readable summary</h2><pre>""" + html.escape(json.dumps(report, indent=2, ensure_ascii=False)) + "</pre></html>"


def compile_semantics(root, config_path):
    root = Path(root).resolve()
    config = read_json(config_path)
    policy = {"format_version": 1, "feature_reduction": "sum", "normalization": "preserve_annotation_labels_and_senses",
              "availability": "annotation_unit_endpoint", "query_label_policy": "annotation_supported_answers_no_closed_world_negatives"}
    if any(config.get(key) != value for key, value in policy.items()):
        raise ValueError("Unsupported semantic compilation policy.")
    index, captured, files, coverage, annotation_hash = capture_inputs(root, config)
    alignment_index = read_json(root / config["alignment"] / "index.json")
    modules = ["semantic_records.py", "semantic_queries.py", "semantics.py", "semantic_features.py", "graphs.py", "io.py", "temporal.py", "extraction_inputs.py"]
    identity = {"format_version": 1, "config": config, "corpus_hash": index["content_hash"],
                "annotation_run_hash": annotation_hash, "annotation_files": files,
                "alignment_hash": alignment_index["content_hash"], "coverage": coverage,
                "code_hashes": {name: digest_file(Path(__file__).parent / name) for name in modules},
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pydantic"]}}
    build_hash = object_hash(identity)
    output = root / config["output"]
    if not output.resolve().is_relative_to(root) or output.resolve() != output:
        raise ValueError("Compiler output must be a nonsymlink directory inside the project.")
    build = output / "builds" / build_hash
    output.mkdir(parents=True, exist_ok=True)
    if (build / "complete.json").is_file():
        verify_build(build)
        save_json(output / "latest.json", {"build_hash": build_hash, "path": "builds/" + build_hash})
        print("CACHED SEMANTIC BUILD", build, flush=True)
        return build
    staging = Path(tempfile.mkdtemp(prefix=".compiling-", dir=output))
    save_json(staging / "identity.json", identity)
    save_json(staging / "splits.json", story_folds(index))
    definitions, candidate_sets, catalog = {}, {}, {}
    query_counts, review_counts, protocol_counts = defaultdict(Counter), Counter(), Counter()
    review_status, totals, skipped_counts = Counter(), Counter(), Counter()
    source_example = None
    for entry, story_coverage in zip(index["stories"], coverage, strict=True):
        story, records = captured[entry["id"]]
        alignment = checked_alignment(root / config["alignment"], entry["id"], index["content_hash"])
        history = GraphHistory()
        sources, features, queries, answers, private, occurrences, reviews, deltas = [], [], [], [], [], [], [], []
        for unit_index, saved in enumerate(records):
            unit = story["units"][unit_index]
            graph = GraphDelta.model_validate(saved["graph"])
            history.append(graph, unit)
            source, feat, defs, occ, rev = compile_unit(story, unit, graph, history, saved, unit_index, alignment)
            qs, ans, hidden, skipped = compile_queries(story, unit, graph, history, source, config["candidate_seed"])
            for query in qs:
                candidates = query["input"].pop("candidates")
                key = object_hash(candidates)
                candidate_sets[key] = candidates
                query["input"]["candidate_set"] = key
            for answer in ans:
                for warning in graph.uncertainties:
                    if any(warning.span.start < span["end"] and span["start"] < warning.span.end for span in answer["evidence"]):
                        answer["review_flags"].append("overlapping_annotator_uncertainty:" + warning.category)
                        answer["scoring_eligible"] = False
                if any(item["reason"] == "causal_interpretation" and set(item["nodes"]) & set(answer["nodes"]) for item in rev):
                    answer["review_flags"].append("causal_interpretation")
                query_counts[answer["family"]].update({"total": 1, answer["label_state"]: 1,
                    "scoring_eligible_against_annotations": int(answer["scoring_eligible"]),
                    "diagnostic_only": int(answer["diagnostic_only"])})
            family_sizes = Counter(a["family"] for a in ans if a["scoring_eligible"])
            for answer in ans:
                answer["source_weight"] = 1.0 / (len(family_sizes) * family_sizes[answer["family"]]) if answer["scoring_eligible"] else 0.0
            skipped_counts.update(skipped)
            sources.append(source); features.append(feat); definitions.update(defs)
            queries.extend(qs); answers.extend(ans); private.extend(hidden)
            occurrences.extend(occ); reviews.extend(rev)
            deltas.append({"source_id": unit["id"], "graph": saved["graph"], "graph_hash": saved["graph_hash"]})
            protocol_counts[source["annotation_protocol"]] += 1
            review_status[source["annotation_review_status"]] += 1
            review_counts.update(item["reason"] for item in rev)
            totals.update({"units": 1, "queries": len(qs), "occurrences": len(occ),
                           **{name: len(getattr(graph, name)) for name in ["entities", "mentions", "events", "relations", "entity_updates", "literals", "contexts", "identity_links"]}})
            for occurrence in occ:
                key = object_hash({"kind": occurrence["kind"], "label": occurrence["label"], "view": occurrence["view"]})
                item = catalog.setdefault(key, {"kind": occurrence["kind"], "label": occurrence["label"], "view": occurrence["view"],
                                                "occurrences": 0, "source_ids": set(), "story_ids": set()})
                item["occurrences"] += 1; item["source_ids"].add(unit["id"]); item["story_ids"].add(story["story_id"])
            if source_example is None or unit["id"] == "story_01_u0002":
                source_example = {"source_id": unit["id"], "text": story["text"][unit["char_start"]:unit["prefix_char_end"]],
                                  "graph": saved["graph"], "features": feat, "queries": qs, "answers": ans, "reviews": rev}
        known = list(alignment["units"]["known_raw_rows"])
        endpoint = alignment["units"]["bin_indices"][len(records) - 1] if records else -1
        # Prefix coverage ends at the last safely observed annotation endpoint.
        # An unresolved final endpoint never licenses the unknown suffix.
        if len(records) < len(story["units"]):
            if endpoint < 0:
                endpoint = next((alignment["units"]["bin_indices"][i] for i in range(len(records) - 2, -1, -1)
                                 if alignment["units"]["bin_indices"][i] >= 0), -1)
            known = [valid and i <= endpoint for i, valid in enumerate(known)]
        if not records:
            known = [False] * len(known)
        for unit_index in range(len(records), len(story["units"])):
            slot = alignment["units"]["bin_indices"][unit_index]
            if slot >= 0:
                known[slot] = False
        directory = staging / "stories" / entry["id"]
        for filename, values in [("sources", sources), ("features", features), ("queries", queries), ("answers", answers),
                                 ("oracle", private), ("occurrences", occurrences), ("review", reviews), ("history", deltas)]:
            write_jsonl(directory / (filename + ".jsonl"), values)
        save_json(directory / "state.json", state_view(history))
        save_json(directory / "timing.json", {"story_id": entry["id"], "raw_response_rows": alignment["raw_response_rows"],
                  "response_raw_indices": alignment["response_raw_indices"], "fir_delays_trs": alignment["fir_delays_trs"],
                  "known_raw_rows": known, "coverage_complete": story_coverage["coverage_complete"],
                  "alignment_hash": alignment["content_hash"]})
        print("COMPILED", entry["id"], len(records), "/", len(story["units"]), "units;", len(queries), "queries", flush=True)
    for value in catalog.values():
        value["source_ids"] = sorted(value["source_ids"]); value["story_ids"] = sorted(value["story_ids"])
        value["distinct_source_units"] = len(value["source_ids"])
        value["stories"] = len(value["story_ids"])
    save_json(staging / "feature_catalog.json", definitions)
    save_json(staging / "candidate_sets.json", candidate_sets)
    save_json(staging / "occurrence_catalog.json", catalog)
    save_json(staging / "worked_example.json", source_example)
    report = {"format_version": 1, "build_hash": build_hash, "compilation_status": "complete",
              "corpus_coverage_complete": all(s["coverage_complete"] for s in coverage),
              "semantic_accuracy_measured": False, "neural_fits_executed": False, "llm_calls": 0,
              "counts": dict(totals), "expected_units": sum(s["total_units"] for s in coverage), "stories": coverage,
              "feature_groups": dict(Counter(d["group"] for d in definitions.values())),
              "feature_routes": dict(Counter(d["route"] for d in definitions.values())),
              "queries": {key: dict(value) for key, value in query_counts.items()}, "query_skips": dict(skipped_counts),
              "review_flags": dict(review_counts), "annotation_protocols": dict(protocol_counts),
              "annotation_review_statuses": dict(review_status),
              "concept_occurrence_policy": "Only supported positives are emitted; these are diagnostic until justified negative labels exist.",
              "binding_policy": "Polarity complements label the selected proposition. Role swaps without explicit support remain unlabelled.",
              "candidate_policy": "Same available catalogs for all legal queries/priors; semantic and location priors must be evaluated.",
              "geometry_policy": "Occurrences and context support only. No concept signatures or neural geometry have been estimated."}
    save_json(staging / "report.json", report)
    (staging / "report.html").write_text(compilation_report_html(report), encoding="utf-8")
    for name, digest in files.items():
        if digest_file(root / name) != digest:
            raise ValueError("A captured annotation changed during compilation: " + name)
    artifacts = {path.relative_to(staging).as_posix(): digest_file(path) for path in sorted(staging.rglob("*")) if path.is_file()}
    save_json(staging / "complete.json", {"build_hash": build_hash, "artifact_hashes": artifacts,
                                          "completed_utc": datetime.now(timezone.utc).isoformat()})
    build.parent.mkdir(parents=True, exist_ok=True)
    if build.exists():
        raise ValueError("Destination already exists; retained staging files for inspection: " + str(staging))
    os.replace(staging, build)
    save_json(output / "latest.json", {"build_hash": build_hash, "path": "builds/" + build_hash})
    return build


def verify_build(build):
    build = Path(build)
    receipt = read_json(build / "complete.json")
    if object_hash(read_json(build / "identity.json")) != receipt["build_hash"] or build.name != receipt["build_hash"]:
        raise ValueError("Semantic build identity mismatch.")
    for name, digest in receipt["artifact_hashes"].items():
        path = build / name
        if not path.resolve().is_relative_to(build.resolve()) or digest_file(path) != digest:
            raise ValueError("Semantic artifact changed: " + name)
    return receipt
