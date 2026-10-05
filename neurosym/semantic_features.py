"""Model-facing semantic inputs, training-only vectorization, and actual support masks."""
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from .graphs import GraphDelta, GraphHistory
from .io import object_hash, read_json, save_json
from .semantic_records import state_view
from .semantics import read_jsonl, verify_build
from .temporal import causal_bin_features, fir_design


def _immutable_change(*args, **kwargs):
    raise TypeError("Public candidate descriptors are immutable.")


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable_change

    def __deepcopy__(self, memo):
        return self


class FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable_change

    def __deepcopy__(self, memo):
        return self


def freeze_descriptor(value):
    if isinstance(value, dict):
        return FrozenDict({k: freeze_descriptor(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_descriptor(v) for v in value)
    return value


class CandidateSets(Mapping):
    """Share immutable descriptor storage; materialize only a requested ordered set."""
    def __init__(self, record):
        self.pooled = record.get("format_version") == 2
        self.sets = record["sets"] if self.pooled else record
        self.descriptors = {k: freeze_descriptor(v) for k, v in record["descriptors"].items()} if self.pooled else {}
        self._resolved = {}

    def __getitem__(self, key):
        if not self.pooled:
            return self.sets[key]
        if key not in self._resolved:
            self._resolved[key] = FrozenList(self.descriptors[d] for d in self.sets[key])
        return self._resolved[key]

    def __iter__(self):
        return iter(self.sets)

    def __len__(self):
        return len(self.sets)


class SemanticDataset:
    def __init__(self, build: Path, *, verify=True):
        self.build = Path(build)
        if verify:
            verify_build(self.build)
        self.identity = read_json(self.build / "identity.json")
        self.build_hash = object_hash(self.identity)
        self.splits = read_json(self.build / "splits.json")
        self.definitions = read_json(self.build / "feature_catalog.json")
        self.candidate_sets = CandidateSets(read_json(self.build / "candidate_sets.json"))
        self.story_ids = [s["story_id"] for s in self.identity["coverage"]]
        self._cache = {}

    def records(self, story_id, kind):
        if story_id not in self.story_ids or kind not in {"sources", "features", "queries", "answers", "oracle", "occurrences", "review", "history", "expressions"}:
            raise ValueError("Unknown semantic story or artifact kind.")
        key = story_id, kind
        if key not in self._cache:
            self._cache[key] = list(read_jsonl(self.build / "stories" / story_id / (kind + ".jsonl")))
        return deepcopy(self._cache[key])

    def decoder_inputs(self, query):
        """Only these values may enter a decoder or its query-only comparator."""
        if set(query) != {"id", "source_id", "family", "input", "ast_hash"}:
            raise ValueError("Unexpected query fields; do not pass arbitrary records to a decoder.")
        allowed = {"ast", "candidate_set", "binding_candidates"}
        if set(query["input"]) - allowed or not {"ast", "candidate_set"} <= set(query["input"]):
            raise ValueError("Unexpected model-facing query fields.")
        inputs = query["input"]
        if query["ast_hash"] != object_hash(inputs["ast"]):
            raise ValueError("Query AST changed.")
        result = {"ast": deepcopy(inputs["ast"]), "candidates": deepcopy(self.candidate_sets[inputs["candidate_set"]])}
        if "binding_candidates" in inputs:
            result["binding_candidates"] = deepcopy(inputs["binding_candidates"])
        return result

    def decoder_examples(self, story_id, *, reviewed_only=False, families=None):
        """Yield separate inputs/targets; weights are recomputed after selection."""
        answers = {a["query_id"]: a for a in self.records(story_id, "answers")}
        selected = []
        for query in self.records(story_id, "queries"):
            answer = answers[query["id"]]
            if not answer["scoring_eligible"] or (families is not None and query["family"] not in families):
                continue
            if reviewed_only and answer["annotation_review_status"] not in {"reviewed", "adjudicated", "human_reviewed", "accepted_reviewed"}:
                continue
            selected.append((query, answer))
        counts = defaultdict(Counter)
        for query, _ in selected:
            counts[query["source_id"]][query["family"]] += 1
        for query, answer in selected:
            groups = counts[query["source_id"]]
            yield {"query_id": query["id"], "source_id": query["source_id"],
                   "inputs": self.decoder_inputs(query), "acceptable_indices": answer["acceptable_indices"],
                   "weight": 1.0 / (len(groups) * groups[query["family"]]),
                   "annotation_review_status": answer["annotation_review_status"]}

    def state_at(self, story_id, source_id):
        sources = {s["id"]: s for s in self.records(story_id, "sources")}
        if source_id not in sources:
            raise ValueError("The requested annotation endpoint is not compiled.")
        if self.identity.get("semantic_interface") == "reviewed-scoped-expressions-v1":
            from .reviewed_graph import ReviewedHistory
            history = ReviewedHistory(read_json(self.build / "stories" / story_id / "constraints.json"))
            for delta in self.records(story_id, "history"):
                history.append({**delta, "unit_id": delta["source_id"]}, delta["aliases"])
                if delta["source_id"] == source_id:
                    return history.state()
            raise ValueError("Missing reviewed history endpoint.")
        history = GraphHistory()
        for delta in self.records(story_id, "history"):
            source = sources[delta["source_id"]]
            unit = {"id": source["id"], "story_id": story_id,
                    "start_token": source["local_word_span"][0], "end_token": source["local_word_span"][1]}
            history.append(GraphDelta.model_validate(delta["graph"]), unit)
            if source["id"] == source_id:
                return state_view(history)
        raise ValueError("Missing graph history endpoint.")

    def composition_holdout(self, heldout_keys, train_stories, test_stories):
        """Conservative story-level purging covers full-prefix and delayed BOLD access."""
        train_stories, test_stories, heldout_keys = set(train_stories), set(test_stories), set(heldout_keys)
        if train_stories & test_stories or not train_stories <= set(self.splits["development"]):
            raise ValueError("Invalid compositional training/test story separation.")
        if not test_stories <= set(self.story_ids) or not heldout_keys:
            raise ValueError("Unknown test stories or empty combination holdout.")
        sources = {sid: self.records(sid, "sources") for sid in train_stories | test_stories}
        known_keys = {c["key"] for records in sources.values() for s in records for c in s["configurations"]}
        if not heldout_keys <= known_keys:
            raise ValueError("Requested combination does not occur in the supplied real sources.")
        purged = {sid for sid in train_stories if any(c["key"] in heldout_keys for s in sources[sid] for c in s["configurations"])}
        # A still unannotated suffix cannot certify the absence of a held-out
        # combination. Strict compositional splits use complete training stories.
        incomplete = {s["story_id"] for s in self.identity["coverage"] if not s["coverage_complete"]} & train_stories
        kept = train_stories - purged - incomplete
        components = {part for sid in kept for s in sources[sid] for c in s["configurations"]
                      if c["grounding_resolved"] for part in c["constituents"]}
        eligible, unsupported = [], []
        for sid in sorted(test_stories):
            for source in sources[sid]:
                matches = [c for c in source["configurations"] if c["key"] in heldout_keys]
                for match in matches:
                    target = {"source_id": source["id"], "configuration": match["key"],
                              "grounding_resolved": match["grounding_resolved"],
                              "unseen_constituents": sorted(set(match["constituents"]) - components)}
                    (unsupported if target["unseen_constituents"] or not target["grounding_resolved"] else eligible).append(target)
        return {"build_hash": self.build_hash, "policy": "Whole training stories containing heldout combinations are purged, including all prefix and neural support.",
                "train_stories": sorted(kept), "purged_stories": sorted(purged), "test_stories": sorted(test_stories),
                "incomplete_training_stories_excluded": sorted(incomplete),
                "heldout_keys": sorted(heldout_keys), "eligible_tests": eligible, "unsupported_tests": unsupported,
                "available": bool(kept and eligible)}


class SemanticVectorizer:
    def __init__(self, identity):
        self.identity = identity
        self.vocabulary = {key: i for i, key in enumerate(identity["vocabulary"])}

    @classmethod
    def fit(cls, dataset, train_stories, *, groups=("C", "B", "BR", "R", "S", "D", "U"),
            include_exact=False, min_source_count=1):
        train_stories = sorted(set(train_stories))
        if not train_stories or not set(train_stories) <= set(dataset.splits["development"]):
            raise ValueError("Fit vocabulary on explicit development training stories only.")
        if type(min_source_count) is not int or min_source_count < 1:
            raise ValueError("Source-frequency threshold must be a positive integer.")
        legal_groups = {"L", "C", "B", "BR", "GB", "GBR", "PB", "PBR", "R", "S", "D", "U"}
        if not groups or not set(groups) <= legal_groups:
            raise ValueError("Unknown semantic feature group.")
        counts, source_ids = Counter(), []
        for sid in train_stories:
            for record in dataset.records(sid, "features"):
                source_ids.append(record["source_id"])
                for key in record["terms"]:
                    definition = dataset.definitions[key]
                    if definition["group"] in groups and (include_exact or definition["route"] == "factorized"):
                        counts[key] += 1
        vocabulary = sorted(key for key, count in counts.items() if count >= min_source_count)
        if not vocabulary:
            raise ValueError("No observed training features satisfy the requested groups/coverage.")
        return cls({"format_version": 1, "build_hash": dataset.build_hash, "train_stories": train_stories,
                    "training_source_ids": sorted(source_ids), "groups": sorted(set(groups)), "include_exact": include_exact,
                    "min_source_count": min_source_count, "vocabulary": vocabulary,
                    "source_frequency": {key: counts[key] for key in vocabulary},
                    "representation": "Additive shared predicate/role/filler factors; optional exact conjunctions. No neural fitting or pretrained semantic embedding."})

    def transform(self, dataset, story_id):
        if dataset.build_hash != self.identity["build_hash"]:
            raise ValueError("A vectorizer cannot silently move to another annotation snapshot.")
        records = dataset.records(story_id, "features")
        values = np.zeros((len(records), len(self.vocabulary)), dtype=np.float32)
        unseen, observed = Counter(), 0
        for i, record in enumerate(records):
            for key, value in record["terms"].items():
                definition = dataset.definitions[key]
                if definition["group"] not in self.identity["groups"] or (not self.identity["include_exact"] and definition["route"] != "factorized"):
                    continue
                observed += value
                if key in self.vocabulary:
                    values[i, self.vocabulary[key]] = value
                else:
                    unseen[key] += value
        return values, {"source_ids": [r["source_id"] for r in records], "unseen_feature_columns": len(unseen),
                        "unseen_feature_tokens": sum(unseen.values()), "observed_feature_tokens": observed}

    def design(self, dataset, story_id, *, delays=None):
        values, coverage = self.transform(dataset, story_id)
        sources = dataset.records(story_id, "sources")
        timing = read_json(dataset.build / "stories" / story_id / "timing.json")
        full, counts = causal_bin_features(values, [s["raw_feature_bin"] for s in sources], timing["raw_response_rows"], reduction="sum")
        known = np.asarray(timing["known_raw_rows"], dtype=bool)
        for group in self.identity["groups"]:
            known &= np.asarray(timing.get("group_known_raw_rows", {}).get(group, timing["known_raw_rows"]), dtype=bool)
        design, valid = fir_design(full, timing["response_raw_indices"],
                                  timing["fir_delays_trs"] if delays is None else delays, known)
        return design, valid, {**coverage, "observed_unit_counts": counts.tolist()}

    def save(self, path):
        save_json(Path(path), {"identity": self.identity, "content_hash": object_hash(self.identity)})

    @classmethod
    def load(cls, path):
        record = read_json(Path(path))
        if record["content_hash"] != object_hash(record["identity"]):
            raise ValueError("Semantic vectorizer changed.")
        return cls(record["identity"])
