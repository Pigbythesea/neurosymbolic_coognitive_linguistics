"""Outcome-independent temporal support shared by every condition of a contrast."""
from collections import defaultdict

import h5py
import numpy as np

from .io import object_hash, read_json
from .temporal import checked_alignment

SEMANTIC_GROUPS = ("L", "C", "B", "BR", "GB", "GBR", "PB", "PBR", "R", "S", "D", "U")
ENCODING_GROUPS = set(SEMANTIC_GROUPS) | {"presentation", "legacy", "model"}


def delayed_support(known, rows, delays):
    """Intersect original-timeline support at each FIR lag, without constructing X."""
    known, rows = np.asarray(known, dtype=bool), np.asarray(rows, dtype=np.int64)
    if known.ndim != 1 or rows.ndim != 1 or not delays or any(type(d) is not int or d < 0 for d in delays):
        raise ValueError("Invalid FIR support contract.")
    if len(set(delays)) != len(delays) or np.any(rows < 0) or np.any(rows >= len(known)):
        raise ValueError("Duplicate delays or response indices outside original story.")
    valid = np.ones(len(rows), dtype=bool)
    for delay in delays:
        indices = rows - delay
        inside = indices >= 0
        valid &= inside
        valid[inside] &= known[indices[inside]]
    return valid


def comparison_spec(config, options):
    """Freeze a declared union of required inputs; never infer it from one condition."""
    name, explicit = options.get("comparison"), options.get("mask_groups")
    if bool(name) == bool(explicit):
        raise ValueError("Declare either --comparison or --mask-groups for the entire contrast.")
    if name:
        profiles = config["encoding"]["comparisons"]
        if name not in profiles:
            raise ValueError("Unknown encoding comparison: " + name)
        required = list(profiles[name])
    else:
        required = list(explicit)
    if not required or len(set(required)) != len(required) or set(required) - ENCODING_GROUPS:
        raise ValueError("Invalid comparison feature groups.")
    models = []
    for value in options.get("mask_models") or []:
        model, separator, layer = value.rpartition(":")
        if not separator or not model or not layer.isdigit():
            raise ValueError("Each --mask-model must be MODEL_ID:LAYER.")
        models.append({"model": model, "layer": int(layer)})
    if len({(m["model"], m["layer"]) for m in models}) != len(models):
        raise ValueError("Duplicate comparison model/layer.")
    if models and "model" not in required:
        required.append("model")
    if "model" in required and not models:
        raise ValueError("A model contrast requires explicit --mask-model for every compared model/layer, including baseline runs.")
    policy = options.get("mask_policy", "comparison")
    if policy not in {"comparison", "all-groups"}:
        raise ValueError("Unknown encoding mask policy.")
    spec = {"format_version": 1, "comparison": name or "explicit", "policy": policy,
            "required_groups": sorted(required),
            "semantic_groups": sorted(SEMANTIC_GROUPS if policy == "all-groups" else set(required) & set(SEMANTIC_GROUPS)),
            "models": sorted(models, key=lambda m: (m["model"], m["layer"])),
            "model_kind": config["encoding"]["model_kind"],
            "timing_policy": "unit interpretation availability AND word exposure; all original-timeline FIR lags"}
    validate_condition(spec, options["groups"], options.get("model"), options.get("layer"))
    return spec


def validate_condition(spec, groups, model=None, layer=None):
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(spec["required_groups"]):
        raise ValueError("Condition features must be a nonempty subset of the declared comparison inputs.")
    if "model" in groups:
        if {"model": model, "layer": layer} not in spec["models"]:
            raise ValueError("The condition's model/layer is absent from the shared comparison support.")
    elif model is not None or layer is not None:
        raise ValueError("Use --mask-model for a baseline's comparison support; --model/--layer require a fitted model group.")


def model_support(reader, story, layer, kind, rows, delays):
    """Read real extraction metadata only, including for a non-model baseline."""
    if kind not in {"words", "units"}:
        raise ValueError("Unknown representation kind.")
    with h5py.File(reader.source / "aligned" / (story + ".h5"), "r") as file:
        if not file.attrs["complete"] or file.attrs["run_hash"] != object_hash(reader.aligned_run):
            raise ValueError("Incomplete or mismatched model support.")
        shape = file[kind]["features"].shape
        known = file[kind]["known_raw_rows"][()]
        if (not 0 <= layer < shape[0] or shape[1] != len(known) or
                file["response_raw_indices"][()].tolist() != rows or file["fir_delays_trs"][()].tolist() != delays):
            raise ValueError("Compared model layer/timeline does not match semantic FIR coordinates.")
    return delayed_support(known, rows, delays)


def source_coverage(data, story, mask):
    """Descriptive coverage by real phenomenon, not a target for mask optimization."""
    sources = {s["id"]: s for s in data.semantics.records(story, "sources")}
    categories = defaultdict(set)
    categories["all_units"].update(sources)
    for feature in data.semantics.records(story, "features"):
        for group in feature.get("uncertain_groups", []):
            categories["uncertain_group:" + group].add(feature["source_id"])
    for record in data.semantics.records(story, "occurrences"):
        categories["occurrence:" + record["kind"]].add(record["source_id"])
    result = {}
    for category, ids in sorted(categories.items()):
        eligible = [sources[i] for i in sorted(ids) if sources[i]["decoder_timing_eligible"]]
        full, partial = 0, 0
        for source in eligible:
            rows = source["decoder_trimmed_response_rows"]
            observed = mask[rows]
            full += int(observed.all())
            partial += int(observed.any() and not observed.all())
        result[category] = {"units": len(ids), "timed_delayed_windows": len(eligible),
                            "full_window_retained": full, "partial_window_retained": partial,
                            "no_window_rows_retained": len(eligible) - full - partial}
    return result


class EncodingSupport:
    """No learned parameters, response values, feature vocabulary or fold dependence."""
    def __init__(self, data, spec):
        self.data, self.spec = data, spec
        self._masks, self.reports = {}, {}

    def story(self, story):
        if story in self._masks:
            return self._masks[story].copy()
        data = self.data
        timing = read_json(data.semantics.build / "stories" / story / "timing.json")
        rows, delays = timing["response_raw_indices"], timing["fir_delays_trs"]
        alignment = checked_alignment(data.root / "data/processed/alignment", story, data.index["content_hash"])
        if (alignment["response_raw_indices"] != rows or alignment["fir_delays_trs"] != delays or
                alignment["raw_response_rows"] != timing["raw_response_rows"]):
            raise ValueError("Word exposure and interpretation clocks have different FIR coordinates.")
        checks = {"interpretation_timing": delayed_support(timing["known_raw_rows"], rows, delays),
                  "word_exposure_timing": delayed_support(alignment["words"]["known_raw_rows"], rows, delays)}
        valid = checks["interpretation_timing"] & checks["word_exposure_timing"]
        timing_count = int(valid.sum())
        for group in self.spec["semantic_groups"]:
            if group not in timing.get("group_known_raw_rows", {}):
                raise ValueError("Comparison-specific group availability is missing; use a reviewed compiled build.")
            checks["semantic:" + group] = delayed_support(timing["group_known_raw_rows"][group], rows, delays)
            valid &= checks["semantic:" + group]
        semantic_count = int(valid.sum())
        for source in self.spec["models"]:
            key = "model:" + source["model"] + ":" + str(source["layer"])
            checks[key] = model_support(data.model(source["model"]), story, source["layer"],
                                        self.spec["model_kind"], rows, delays)
            valid &= checks[key]
        report = {"response_rows": len(rows), "timing_valid": timing_count,
                  "timing_excluded": len(rows) - timing_count,
                  "semantic_additional_excluded": timing_count - semantic_count,
                  "model_additional_excluded": semantic_count - int(valid.sum()),
                  "retained": int(valid.sum()), "response_row_indices": np.flatnonzero(valid).tolist(),
                  "raw_response_indices": np.asarray(rows)[valid].tolist(), "fir_delays_trs": delays,
                  "failures_by_reason": {key: np.flatnonzero(~mask).tolist() for key, mask in checks.items()},
                  "source_coverage": source_coverage(data, story, valid)}
        self._masks[story], self.reports[story] = valid.copy(), report
        return valid.copy()

    def report(self, stories):
        for story in stories:
            self.story(story)
        records = {s: self.reports[s] for s in sorted(stories)}
        result = {"semantic_build_hash": self.data.semantics.build_hash, "spec": self.spec, "stories": records,
                  "model_alignments": {m["model"]: self.data.model(m["model"]).aligned_run for m in self.spec["models"]},
                  "outcome_independent": True, "learned_parameters": False,
                  "coverage_unit": "stimulus timepoints and annotation units, not participant/voxel/repeat counts"}
        return {**result, "content_hash": object_hash(result)}


def assert_matched_support(first, second):
    """Reject a paired effect if any training/validation/test support differs."""
    for report in (first, second):
        if report.get("content_hash") != object_hash({k: v for k, v in report.items() if k != "content_hash"}):
            raise ValueError("Encoding support receipt changed.")
    if first["content_hash"] != second["content_hash"]:
        raise ValueError("Encoding comparisons require identical declared inputs and all-story support, including training rows.")
