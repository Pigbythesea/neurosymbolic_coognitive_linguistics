"""Verify comparison-specific encoding support on every real corpus story."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from neurosym.analysis_data import AnalysisData
from neurosym.encoding import EncodingFeatures
from neurosym.encoding_support import EncodingSupport, SEMANTIC_GROUPS, assert_matched_support, comparison_spec
from neurosym.io import object_hash, read_json, save_json


def require(value, message):
    if not value:
        raise AssertionError(message)


def verify(data=None):
    data = data or AnalysisData(ROOT)
    stories = data.semantics.story_ids
    profiles = data.config["encoding"]["comparisons"]
    supports, reports = {}, {}
    for name in profiles:
        options = {"comparison": name, "groups": ["presentation"]}
        supports[name] = EncodingSupport(data, comparison_spec(data.config, options))
    strict = EncodingSupport(data, comparison_spec(data.config, {
        "comparison": "joint", "groups": ["presentation"], "mask_policy": "all-groups"}))
    # Fit only one real development story: a small vocabulary deliberately leaves
    # unseen held-out features. Eligibility must still reflect actual uncertainty.
    factories = {name: EncodingFeatures(data, ["story_01"], groups, support=supports[name])
                 for name, groups in profiles.items()}
    baselines = {name: EncodingFeatures(data, ["story_02"], ["presentation"], support=supports[name])
                 for name in profiles}
    expected_totals, design_checks, granularity = Counter(), 0, {g: Counter() for g in SEMANTIC_GROUPS}
    saved_rows, marginal_checks = {}, 0
    for story in stories:
        timing = read_json(data.semantics.build / "stories" / story / "timing.json")
        alignment = read_json(data.root / "data/processed/alignment/stories" / (story + ".json"))
        sources = {s["id"]: s for s in data.semantics.records(story, "sources")}
        # Independent reconstruction from real unit uncertainty, rather than
        # repeating the compiled masks or using generated test observations.
        raw = {g: list(timing["known_raw_rows"]) for g in SEMANTIC_GROUPS}
        for feature in data.semantics.records(story, "features"):
            # Independently project the structural columns onto three unbound
            # marginals. Every repeated incidence contributes, including an old
            # referent absent from introduction-based C.
            expected_bag, actual_bag = Counter(), Counter()
            for key, count in feature["terms"].items():
                definition = data.semantics.definitions[key]
                group, parts = definition["group"], definition["parts"]
                if group in {"PB", "PBR"}:
                    _, predicate, role, filler = parts
                    reference = group == "PBR"
                    for atom in (["predicate", predicate, reference], ["role", role, reference],
                                 ["filler", filler, reference]):
                        expected_bag[object_hash(atom)] += count
                elif group == "BC":
                    actual_bag[object_hash(parts)] += count
            require(actual_bag == expected_bag, "Unbound content differs from the binding constituents: " + feature["source_id"])
            uncertain = set(feature["uncertain_groups"])
            require(("BC" in uncertain) == bool(uncertain & {"PB", "PBR"}), "Binding/control uncertainty differs")
            marginal_checks += 1
            slot = sources[feature["source_id"]]["raw_feature_bin"]
            present = {data.semantics.definitions[k]["group"] for k in feature["terms"]}
            for group in feature["uncertain_groups"]:
                granularity[group]["uncertain_units"] += 1
                granularity[group]["uncertain_units_with_known_contributions"] += int(group in present)
                if slot >= 0:
                    raw[group][slot] = False
        for group in SEMANTIC_GROUPS:
            require(raw[group] == timing["group_known_raw_rows"][group], "Compiler group availability diverged: " + group)
        expected = {}
        for name, groups in profiles.items():
            expected[name] = np.array([
                all(t - d >= 0 and timing["known_raw_rows"][t - d] and alignment["words"]["known_raw_rows"][t - d]
                    and all(raw[g][t - d] for g in groups if g in raw) for d in timing["fir_delays_trs"])
                for t in timing["response_raw_indices"]], dtype=bool)
            _, baseline = baselines[name].story(story)
            arrays, augmented = factories[name].story(story)
            require(np.array_equal(baseline, expected[name]) and np.array_equal(augmented, expected[name]),
                    "Baseline/augmented support or independent FIR reconstruction disagree: " + story + "/" + name)
            require(all(np.isfinite(x[augmented]).all() for x in arrays.values()), "Nonfinite real encoding design")
            # Support never depends on which training-story vocabulary is used.
            require(all(v.identity["train_stories"] == ["story_01"] for v in factories[name].vectorizers.values()),
                    "Unexpected vocabulary training stories")
            expected_totals[name] += int(augmented.sum())
            design_checks += 1
        old = np.array([all(t - d >= 0 and timing["comparison_known_raw_rows"][t - d]
                           and alignment["words"]["known_raw_rows"][t - d] for d in timing["fir_delays_trs"])
                        for t in timing["response_raw_indices"]])
        require(np.array_equal(strict.story(story), old), "All-group robustness condition changed the earlier support")
        require(np.array_equal(expected["joint"], old), "Joint profile and all-group robustness disagree")
        require(np.all(old <= expected["concepts"]), "Concept-only support lost globally retained observations")
        saved_rows[story] = len(old)
        print("ENCODING SUPPORT VERIFIED", story, flush=True)
    # Production factories do not allow an undeclared added group or model to
    # silently narrow a baseline's support.
    rejection_checks = [
        {"groups": ["presentation", "C"]},
        {"comparison": "concepts", "groups": ["presentation", "C", "R"]},
        {"comparison": "concepts", "groups": ["model"], "model": "undeclared", "layer": 0},
    ]
    for options in rejection_checks:
        try:
            comparison_spec(data.config, options)
        except ValueError:
            pass
        else:
            raise AssertionError("Undeclared comparison input was accepted")
    for name, support in supports.items():
        reports[name] = support.report(stories)
        assert_matched_support(reports[name], deepcopy(reports[name]))
    changed = deepcopy(reports["reference"])
    changed["stories"]["story_01"]["response_row_indices"].pop()
    changed["content_hash"] = object_hash({k: v for k, v in changed.items() if k != "content_hash"})
    try:
        assert_matched_support(reports["reference"], changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched actual comparison rows were accepted")
    require(data.semantics.splits["heldout"] == ["story_11"], "Final holdout changed")
    try:
        EncodingFeatures(data, ["story_11"], ["C"], support=supports["concepts"])
    except ValueError:
        pass
    else:
        raise AssertionError("Heldout feature fitting was allowed")
    # Full accepted-corpus regression, not a target used to optimize masks.
    require(sum(saved_rows.values()) == 4028, "Response timeline changed")
    require(expected_totals["concepts"] == 3539 and expected_totals["reference"] == 2979 and
            expected_totals["discourse"] == 3365 and expected_totals["binding"] == 2036 and
            expected_totals["joint"] == 1993, "Accepted-corpus support regression")
    return {"status": "verified", "semantic_build_hash": data.semantics.build_hash,
            "config_hash": object_hash(data.config), "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stories": len(stories), "response_rows": sum(saved_rows.values()), "paired_real_design_checks": design_checks,
            "matched_binding_marginal_checks": marginal_checks,
            "retained_by_comparison": dict(expected_totals), "comparisons": reports,
            "all_group_robustness": strict.report(stories),
            "within_group_uncertainty_audit": {g: dict(v) for g, v in granularity.items()},
            "within_group_policy": "Known contributions remain archived. A partially observed group stays unavailable for complete-vector encoding; unrelated groups remain usable. No unknown indicator or imputation is invented.",
            "checks": ["all 1217 units and 11 real stories", "independent uncertainty-to-FIR reconstruction",
                       "matched baseline/augmented designs", "fold-vocabulary-independent support", "all-group reproduction",
                       "undeclared-input rejection", "paired-support mismatch rejection", "story 11 fitting rejection"],
            "scientific_fits_executed": False, "modern_model_metadata_executed": False,
            "modern_model_metadata_note": "Aligned hidden-state files are cluster-side. Their actual support is required at execution, including baseline conditions; missing files stop the run."}


if __name__ == "__main__":
    report = verify()
    destination = ROOT / "artifacts/encoding-support-verification.json"
    save_json(destination, report)
    print("VERIFIED:", report["retained_by_comparison"])
    print("REPORT:", destination)
