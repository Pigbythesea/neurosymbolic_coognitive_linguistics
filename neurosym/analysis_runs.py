"""Explicit partitions, immutable run identities, and dependency-aware scores."""
from collections import defaultdict
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from .io import immutable_json, object_hash, read_json, save_json


def partition(dataset, fold):
    if fold.startswith("stories:"):
        test = fold.removeprefix("stories:").split(",")
        development = dataset.splits["development"]
        if len(set(test)) != len(test) or not set(test) <= set(development):
            raise ValueError("Custom context holdouts must be unique development story IDs.")
        train = [s for s in development if s not in test]
        if not test or len(train) < 3:
            raise ValueError("Nested context holdouts need at least three remaining training stories.")
        inner = [{"train": [s for s in train if s != val], "validation": [val]} for val in train]
    elif fold == "final":
        train, test = dataset.splits["development"], dataset.splits["heldout"]
        inner = dataset.splits["selection_folds"]
    else:
        index = int(fold)
        if not 0 <= index < len(dataset.splits["outer_folds"]):
            raise ValueError("Outer fold outside the registered story splits.")
        selected = dataset.splits["outer_folds"][index]
        train, test, inner = selected["train"], selected["test"], selected["inner_folds"]
    return {"train": train, "test": test, "inner": inner}


def composition_partition(dataset, split, keys):
    """Purge complete training stories; select queries on the exact heldout event."""
    report = dataset.composition_holdout(keys, split["train"], split["test"])
    train = report["train_stories"]
    if not report["available"] or len(train) < 3:
        raise ValueError("Requested composition holdout lacks enough complete, constituent-covered training stories for nested evaluation.")
    eligible = {(r["source_id"], r["configuration"]) for r in report["eligible_tests"]}
    query_ids = []
    for story in split["test"]:
        events = {e["id"]: e for h in dataset.records(story, "history") for e in h["graph"]["events"]}
        selectors = defaultdict(set)
        for source in dataset.records(story, "sources"):
            for config in source["configurations"]:
                if (source["id"], config["key"]) in eligible:
                    if "selector" in config:
                        selectors[source["id"]].add(object_hash(config["selector"]))
                        continue
                    event = events[config["event_id"]]
                    selectors[source["id"]].add(object_hash({"predicate": event["predicate"],
                         "trigger": [event["trigger"]["start"], event["trigger"]["end"]]}))
        for query in dataset.records(story, "queries"):
            ast = query["input"]["ast"]
            if ast["op"] == "reviewed_choice" and ast["task"] in {"role", "binding", "scope", "polarity", "compose"}:
                if object_hash(ast["anchor"]) in selectors[query["source_id"]]:
                    query_ids.append(query["id"])
                continue
            if ast["op"] in {"role", "binding", "status", "polarity"} and object_hash(ast["event"]) in selectors[query["source_id"]]:
                query_ids.append(query["id"])
    if not query_ids:
        raise ValueError("Heldout compositions have no matching event-specific queries.")
    result = {"train": train, "test": split["test"],
              "inner": [{"train": [s for s in train if s != val], "validation": [val]} for val in train],
              "composition": report, "test_query_ids": sorted(query_ids)}
    return result


ANALYSIS_MODULES = ["analysis_runs.py", "analysis_data.py", "decoders.py", "decoder_fit.py", "decoder_batch.py", "decoder_minibatch.py", "protocol.py", "study_jobs.py", "study_reports.py", "grounding_checks.py",
               "encoding.py", "encoding_support.py", "geometry.py", "spatial.py", "semantic_features.py",
               "semantic_queries.py", "dataset.py", "model_features.py", "temporal.py", "reviewed_queries.py",
               "reviewed_graph.py", "reviewed_archive.py", "reviewed_compile.py", "compute.py", "storage.py", "execution.py", "pca.py", "runtime.py",
               "trace_store.py", "trace_geometry.py", "decoder_stages.py", "fit_reuse.py"]


def run_identity(data, kind, options):
    identity = {"format_version": 1, "kind": kind, "options": options,
                "config": data.config, "semantic_build_hash": data.semantics.build_hash,
                "spatial_hash": object_hash(data.spatial.identity),
                "data_contract_hash": object_hash(data.reader.contract), "partial_annotations": data.partial,
                "experiment_definition_hash": object_hash(read_json(data.root / "configs/experiments.json")),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py", "torch", "nibabel", "pydantic")},
                "code": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                         for name in ANALYSIS_MODULES}}
    if options.get("model"):
        identity["model_alignment"] = data.model(options["model"]).aligned_run
    if options.get("comparison_support"):
        identity["comparison_model_alignments"] = {
            m["model"]: data.model(m["model"]).aligned_run for m in options["comparison_support"]["models"]}
    return identity


def run_directory(data, kind, options):
    identity = run_identity(data, kind, options)
    from .fit_reuse import reused_directory
    reused = reused_directory(data.root, identity)
    if reused is not None:
        return reused
    directory = data.root / data.config["output"] / kind / object_hash(identity)
    immutable_json(directory / "identity.json", identity)
    return directory, identity


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_report(path, report):
    save_json(Path(path), json_safe(report))


def column_correlation(first, second):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation needs finite, identically aligned matrices.")
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide((a * b).sum(0), denominator, out=np.full(a.shape[1], np.nan), where=denominator > 1e-12)


def decoder_metrics(rows):
    """Repetitions -> queries -> weighted sources -> stories; never query IID."""
    queries = defaultdict(list)
    for row in rows:
        queries[(row["story_id"], row["source_id"], row["query_id"])].append(row)
    sources, families = defaultdict(list), defaultdict(list)
    for (story, source, _), repetitions in queries.items():
        row = repetitions[0]
        value = {name: float(np.mean([r[name] for r in repetitions])) for name in ("correct", "nll", "chance")}
        sources[(story, source)].append((row["weight"], value))
        families[(story, source, row["family"])].append(value)
    story_values = defaultdict(list)
    for (story, _), values in sources.items():
        total = sum(w for w, _ in values)
        story_values[story].append({k: sum(w * v[k] for w, v in values) / total for k in ("correct", "nll", "chance")})
    by_story = {s: {k: float(np.mean([v[k] for v in vals])) for k in ("correct", "nll", "chance")}
                for s, vals in story_values.items()}
    for values in by_story.values():
        values["accuracy_minus_chance"] = values["correct"] - values["chance"]
    by_family = defaultdict(lambda: defaultdict(list))
    for (story, _, family), vals in families.items():
        by_family[family][story].append({k: float(np.mean([v[k] for v in vals])) for k in ("correct", "nll", "chance")})
    family_stories = {family: {story: {k: float(np.mean([v[k] for v in values])) for k in ("correct", "nll", "chance")}
                              for story, values in stories.items()} for family, stories in by_family.items()}
    for stories in family_stories.values():
        for values in stories.values():
            values["accuracy_minus_chance"] = values["correct"] - values["chance"]
    return {"stories": by_story,
            "story_macro": {k: float(np.mean([v[k] for v in by_story.values()])) if by_story else None
                            for k in ("correct", "nll", "chance", "accuracy_minus_chance")},
            "family_stories": family_stories,
            "family_story_macro_accuracy": {f: float(np.mean([v["correct"] for v in stories.values()]))
                                            for f, stories in family_stories.items()},
            "n_stories": len(by_story), "n_sources": len(sources), "n_queries": len(queries),
            "n_repeat_predictions": len(rows), "inference_unit": "story and participant, not question"}
