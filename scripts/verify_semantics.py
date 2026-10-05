"""Verify the compiler on a captured real annotation snapshot; no generated fixtures."""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from neurosym.extraction_inputs import checked_index, checked_story
from neurosym.graphs import GraphDelta, GraphHistory
from neurosym.io import object_hash, read_json, save_json
from neurosym.semantic_features import SemanticDataset, SemanticVectorizer
from neurosym.semantic_queries import evaluate_query
from neurosym.semantics import digest_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(build):
    if read_json(Path(build) / "identity.json").get("semantic_interface") == "reviewed-scoped-expressions-v1":
        from verify_reviewed_semantics import verify as verify_reviewed
        report = verify_reviewed(build)
        destination = ROOT / "artifacts/reviewed-semantics-verification.json"
        save_json(destination, report)
        print(report)
        print("VERIFICATION REPORT:", destination)
        return
    dataset = SemanticDataset(build)
    config = dataset.identity["config"]
    corpus = ROOT / config["corpus"]
    index = checked_index(corpus)
    require(index["content_hash"] == dataset.identity["corpus_hash"], "Corpus differs from captured build.")
    for name, digest in dataset.identity["annotation_files"].items():
        require(digest_file(ROOT / name) == digest, "Captured annotation changed: " + name)
    for name, digest in dataset.identity["code_hashes"].items():
        require(digest_file(ROOT / "neurosym" / name) == digest, "Recompile after code changes: " + name)
    for key, definition in dataset.definitions.items():
        require(object_hash(definition) == key, "Feature definition hash mismatch.")
    for key, candidates in dataset.candidate_sets.items():
        require(object_hash(candidates) == key, "Candidate catalog hash mismatch.")
    counts = Counter()
    for entry in index["stories"]:
        sid = entry["id"]
        story = checked_story(corpus, entry)
        sources = dataset.records(sid, "sources")
        features = dataset.records(sid, "features")
        deltas = dataset.records(sid, "history")
        require(len(sources) == len(features) == len(deltas), "Source/feature/history coverage mismatch.")
        require([s["id"] for s in sources] == [u["id"] for u in story["units"][:len(sources)]], "Noncontiguous prefix.")
        queries = dataset.records(sid, "queries")
        answers = {a["query_id"]: a for a in dataset.records(sid, "answers")}
        private = {a["query_id"]: a for a in dataset.records(sid, "oracle")}
        require(len({q["id"] for q in queries}) == len(queries), "Duplicate query ID.")
        require(set(answers) == set(private) == {q["id"] for q in queries}, "Query/target join mismatch.")
        by_source = defaultdict(list)
        for q in queries:
            by_source[q["source_id"]].append(q)
        history = GraphHistory()
        first_state = None
        for i, (source, feature, delta) in enumerate(zip(sources, features, deltas, strict=True)):
            unit = story["units"][i]
            graph = GraphDelta.model_validate(delta["graph"])
            require(object_hash(delta["graph"]) == delta["graph_hash"] == source["annotation_hash"], "Graph provenance mismatch.")
            history.append(graph, unit)
            if i == 0:
                first_state = deepcopy(history.ledger())
            require(source["prefix_word_span"] == [0, unit["end_token"]], "Future text in prefix support.")
            require(source["local_word_span"] == [unit["start_token"], unit["end_token"]], "Local support mismatch.")
            require(source["model_unit_row"] == i and source["available_at_seconds"] == unit["offset_seconds"], "Model/timing join mismatch.")
            require(feature["source_id"] == source["id"] and delta["source_id"] == source["id"], "Source join mismatch.")
            evidence_counts = Counter(e["feature"] for e in feature["evidence"])
            require(evidence_counts == feature["terms"], "Feature counts lack matching evidence.")
            for e in feature["evidence"]:
                definition = dataset.definitions[e["feature"]]
                require(all(0 <= s["start"] < s["end"] <= unit["end_token"] for s in e["evidence"]), "Future feature evidence.")
                if definition["group"] == "B":
                    require("reference_resolution" not in e["dependencies"], "Reference-dependent binding leaked into B.")
            for q in by_source[source["id"]]:
                a, p = answers[q["id"]], private[q["id"]]
                inputs = dataset.decoder_inputs(q)
                require(set(inputs) <= {"ast", "candidates", "binding_candidates"}, "Supervision entered decoder inputs.")
                require(object_hash({"source_id": source["id"], "family": q["family"], "input": inputs}) == q["id"], "Rehydrated query identity mismatch.")
                require(len(inputs["candidates"]) == len(p["candidate_ids"]), "Candidate identity join mismatch.")
                result = evaluate_query(inputs["ast"], history, graph, story, p)
                expected = sorted(p["candidate_ids"].index(x) for x in result) if result else []
                require(a["acceptable_indices"] == expected, "Graph-side query answer mismatch.")
                require((a["label_state"] == "annotation_supported") == (result is not None), "Unsupported query labelled.")
                if result is None or a["diagnostic_only"] or not source["decoder_timing_eligible"]:
                    require(not a["scoring_eligible"], "Unlabelled, diagnostic or untimed query entered scoring.")
                # Independent direct check of role-target direction in the saved graph.
                if inputs["ast"]["op"] == "role":
                    selector = inputs["ast"]["event"]
                    event = next(e for e in graph.events if e.predicate == selector["predicate"] and
                                 [e.trigger.start, e.trigger.end] == selector["trigger"])
                    targets = {arg.target_id for arg in event.arguments if arg.role == inputs["ast"]["role"] and
                               (not arg.mention_id or history.mentions[arg.mention_id].reference_status != "unresolved")}
                    require(set(p["candidate_ids"][j] for j in a["acceptable_indices"]) == targets, "Role direction/target changed.")
                require(all(0 <= s["start"] < s["end"] <= unit["end_token"] for s in a["evidence"]), "Future query evidence.")
                counts["queries"] += 1
                counts["unlabelled_queries"] += int(result is None)
            counts["units"] += 1
        if sources:
            require(dataset.state_at(sid, sources[0]["id"])["ledger"] == first_state, "Prefix replay includes later graph nodes.")
            require(dataset.state_at(sid, sources[-1]["id"]) == read_json(build / "stories" / sid / "state.json"), "Final replay differs.")
        for occurrence in dataset.records(sid, "occurrences"):
            require(object_hash({k: v for k, v in occurrence.items() if k != "id"}) == occurrence["id"], "Occurrence identity mismatch.")
            require(occurrence["source_id"] in {s["id"] for s in sources}, "Occurrence has no source.")
            require(occurrence["evidence"]["end"] <= occurrence["available_at_token"], "Future occurrence support.")
            counts["occurrences"] += 1
        weights = defaultdict(float)
        for example in dataset.decoder_examples(sid):
            weights[example["source_id"]] += example["weight"]
        require(all(math.isclose(w, 1.0, abs_tol=1e-9) for w in weights.values()), "Query multiplicity changes source weighting.")
        require(all(e["annotation_review_status"] in {"reviewed", "adjudicated", "human_reviewed"}
                    for e in dataset.decoder_examples(sid, reviewed_only=True)), "Unreviewed target entered reviewed subset.")
        print("VERIFIED", sid, len(sources), "units;", len(queries), "queries", flush=True)

    # Numerical verification uses actual compiled feature vectors and actual TR mappings.
    training_story = next(sid for sid in dataset.splits["development"] if dataset.records(sid, "sources"))
    vectorizer = SemanticVectorizer.fit(dataset, [training_story])
    training_terms = {key for f in dataset.records(training_story, "features") for key in f["terms"]
                      if dataset.definitions[key]["group"] in vectorizer.identity["groups"] and dataset.definitions[key]["route"] == "factorized"}
    require(set(vectorizer.vocabulary) == training_terms, "Vocabulary uses nontraining sources.")
    artifact_dir = ROOT / "artifacts" / "semantic-verification" / dataset.build_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vectorizer.save(artifact_dir / "vectorizer.json")
    require(SemanticVectorizer.load(artifact_dir / "vectorizer.json").identity == vectorizer.identity, "Fitted vocabulary round-trip changed.")
    for sid in dataset.story_ids:
        values, coverage = vectorizer.transform(dataset, sid)
        design, valid, _ = vectorizer.design(dataset, sid)
        sources = dataset.records(sid, "sources")
        timing = read_json(build / "stories" / sid / "timing.json")
        width = len(vectorizer.vocabulary)
        require(design.shape == (len(timing["response_raw_indices"]), width * len(timing["fir_delays_trs"])), "FIR shape mismatch.")
        total_assigned = sum(values[i].sum() for i, s in enumerate(sources) if s["raw_feature_bin"] >= 0)
        by_bin = defaultdict(list)
        for i, s in enumerate(sources):
            if s["raw_feature_bin"] >= 0:
                by_bin[s["raw_feature_bin"]].append(i)
        require(math.isclose(float(total_assigned), sum(float(values[rows].sum()) for rows in by_bin.values())), "Feature mass not conserved.")
        for row, raw in enumerate(timing["response_raw_indices"]):
            expected_valid = True
            for j, delay in enumerate(timing["fir_delays_trs"]):
                source_row = raw - delay
                expected_valid &= source_row >= 0 and timing["known_raw_rows"][source_row]
                block = design[row, j * width:(j + 1) * width]
                positions = by_bin.get(source_row, [])
                if positions:
                    require(np.array_equal(block, values[positions].sum(axis=0)), "FIR changed lag direction or story alignment.")
                else:
                    require(not np.any(block), "FIR invented a feature in an empty bin.")
            require(bool(valid[row]) == expected_valid, "FIR validity ignores missing support.")
        if not sources:
            require(not np.any(valid), "Unannotated story was treated as observed zeros.")
        if not timing["coverage_complete"]:
            last = max((s["raw_feature_bin"] for s in sources), default=-1)
            require(not any(timing["known_raw_rows"][last + 1:]), "Uncompiled suffix became known support.")
        counts["fir_rows"] += len(valid)
        counts["oov_feature_tokens"] += coverage["unseen_feature_tokens"]
    # The holdout must purge every training story containing the actual test combination.
    test_story = next(sid for sid in dataset.splits["development"] if dataset.records(sid, "sources"))
    target = next(c for s in dataset.records(test_story, "sources") for c in s["configurations"])
    holdout = dataset.composition_holdout([target["key"]], [s for s in dataset.splits["development"] if s != test_story], [test_story])
    complete_stories = {s["story_id"] for s in dataset.identity["coverage"] if s["coverage_complete"]}
    require(set(holdout["train_stories"]) <= complete_stories, "Incomplete annotations certify held-out combination absence.")
    require(all(c["key"] != target["key"] for sid in holdout["train_stories"] for s in dataset.records(sid, "sources")
                for c in s["configurations"]), "Held-out combination remains in training prefix.")
    save_json(artifact_dir / "composition_example.json", holdout)
    for name in ["semantic_records.py", "semantic_queries.py", "semantics.py", "semantic_features.py"]:
        ast.parse((ROOT / "neurosym" / name).read_text(encoding="utf-8"), feature_version=(3, 11))
    for name in ["compile_semantics.py", "verify_semantics.py"]:
        ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 11))
    report = {"status": "passed", "build_hash": dataset.build_hash, "counts": dict(counts),
              "checks": ["captured real annotation and artifact hashes", "prefix graph replay", "query oracle and role direction",
                         "separate decoder inputs and targets", "open-world unknown labels", "source-balanced query weighting",
                         "training-only vocabulary", "actual semantic FIR values and missing-support masks",
                         "whole-story composition purging", "Python 3.11 syntax"],
              "synthetic_data_used": False, "llm_calls": 0, "neural_fits_executed": False,
              "semantic_annotation_accuracy_validated": False}
    save_json(artifact_dir / "report.json", report)
    print(report)
    print("VERIFICATION REPORT:", artifact_dir / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/semantics.json")
    args = parser.parse_args()
    if args.build is None:
        output = ROOT / read_json(args.config)["output"]
        args.build = output / read_json(output / "latest.json")["path"]
    verify(args.build.resolve())


if __name__ == "__main__":
    main()
