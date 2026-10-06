"""Numerical/contract verification using released features and compiled real queries.

Released textual features are used as numerical inputs, explicitly not as fMRI
or modern-model substitutes. This does not train or validate a scientific decoder.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import h5py
from scipy import linalg

from neurosym.analysis_data import AnalysisData, SiteProjector, latest
from neurosym.analysis_runs import decoder_metrics, write_report
from neurosym.decoders import SemanticDecoder, acceptable_loss, detached_trace, query_vocabulary
from neurosym.decoder_fit import evaluate, mismatch_map, write_trace
from neurosym.encoding import EncodingFeatures, GroupRidge
from neurosym.encoding_support import EncodingSupport, comparison_spec
from neurosym.geometry import cosine_rdm, rdm_comparison, trace_signature
from neurosym.io import object_hash
from neurosym.io import read_json


def verify(build):
    torch.set_num_threads(1)
    data = AnalysisData(ROOT, build=build, allow_partial=True)
    train_stories = ["story_01", "story_02"]
    arrays = [data.reader.features(s, ["english1000", "letters", "numwords", "numletters"], trim=False) for s in train_stories]
    x = {"letters": arrays[0]["letters"], "semantics": arrays[0]["english1000"][:, :24]}
    z = {"letters": arrays[1]["letters"], "semantics": arrays[1]["english1000"][:, :24]}
    y = np.concatenate([arrays[0]["numwords"], arrays[0]["numletters"]], axis=1).astype(np.float64)
    ridge = GroupRidge(1.7, {"letters": .4, "semantics": .6}).fit_design(x)
    dual, mean = ridge.coefficients(y)
    tx, vz = ridge.training, ridge.scaler.transform(z)
    train_matrix = np.concatenate([np.sqrt(ridge.weights[g]) * tx[g] for g in tx], axis=1)
    test_matrix = np.concatenate([np.sqrt(ridge.weights[g]) * vz[g] for g in tx], axis=1)
    primal = linalg.solve(train_matrix.T @ train_matrix + 1.7 * np.eye(train_matrix.shape[1]), train_matrix.T @ (y - mean), assume_a="pos")
    first = test_matrix @ primal + mean
    second = ridge.kernel(vz, tx) @ dual + mean
    if not np.allclose(first, second, atol=1e-8, rtol=1e-8):
        raise AssertionError("Group ridge primal/dual disagreement on actual released features.")
    operator = ridge.prediction_operator(z)
    target = np.concatenate([arrays[1]["numwords"], arrays[1]["numletters"]], axis=1).astype(np.float64)
    ym, ys = y.mean(0), y.std(0)
    yc, zc = (y - ym) / ys, (target - ym) / ys
    exact = np.mean((operator @ yc - zc) ** 2)
    gram, cross = yc @ yc.T / y.shape[1], yc @ zc.T / y.shape[1]
    moments = (np.sum((operator @ gram) * operator) - 2 * np.sum(operator * cross.T) + np.sum(zc * zc) / y.shape[1]) / len(zc)
    if not np.isclose(exact, moments, atol=1e-9):
        raise AssertionError("All-target tuning sufficient statistic disagrees with direct MSE.")
    values = arrays[0]["english1000"]
    assignment = np.arange(values.shape[1]) % 8
    projector = SiteProjector(assignment, 3).fit(values, ["story_01"])
    transformed = projector.transform(values)
    # Verify local PCA uses no coordinates outside the registered site.
    positions, center, scale, directions = projector.parameters[0]
    direct = ((values[:, positions] - center) / scale) @ directions
    if not np.allclose(transformed[:, 0, :directions.shape[1]], direct, atol=1e-5):
        raise AssertionError("Local projector coordinate identity failed.")
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        projector.save(temporary)
        restored = SiteProjector.load(temporary)
        if not np.array_equal(restored.transform(values), transformed):
            raise AssertionError("Projector serialization changed real-data output.")
    examples, sources, selected = [], {}, {}
    expected_ops = {"concept", "role", "reference", "status", "polarity", "relation", "binding", "compose"}
    compilation = read_json(data.semantics.build / "report.json")
    reviewed = data.semantics.identity.get("semantic_interface") == "reviewed-scoped-expressions-v1"
    if reviewed:
        from neurosym.reviewed_queries import TASKS
        expected_ops = TASKS - {"compose"} | {"compose/argument_event", "compose/scope_parent", "compose/event_relation"}
    if compilation.get("queries", {}).get("identity_update", {}).get("annotation_supported", 0):
        expected_ops.add("identity")
    for story in data.semantics.splits["development"]:
        sources.update({s["id"]: s for s in data.semantics.records(story, "sources")})
        answers = {r["query_id"]: r for r in data.semantics.records(story, "answers")}
        for query in data.semantics.records(story, "queries"):
            answer = answers[query["id"]]
            source = sources[query["source_id"]]
            slot = source["raw_feature_bin"]
            if not answer["acceptable_indices"] or slot is None or slot < 4:
                continue
            example = {"query_id": query["id"], "source_id": source["id"], "inputs": data.semantics.decoder_inputs(query),
                       "acceptable_indices": answer["acceptable_indices"], "story_id": story,
                       "family": query["family"], "weight": 1.0, "annotation_review_status": answer["annotation_review_status"]}
            ast_query = example["inputs"]["ast"]
            operator_key = ast_query.get("task", ast_query["op"])
            if reviewed and operator_key == "compose":
                operator_key += "/" + ast_query["steps"][0]["op"]
            selected.setdefault(operator_key, example)
        if expected_ops <= set(selected):
            break
    if not expected_ops <= set(selected):
        raise ValueError("Real compiled query coverage is insufficient for operator verification: " + str(expected_ops - set(selected)))
    vocabulary = query_vocabulary(list(selected.values()))
    coverage = {}
    grounding_checks = {}
    numerical_windows = {}
    cached_features = {"story_01": transformed}
    for family in ("prior", "linear", "mlp", "structured"):
        torch.manual_seed(11)
        model = SemanticDecoder(family, (8, 4, 3), vocabulary, hidden=16, site_mask=projector.site_mask)
        for op, example in selected.items():
            story = example["story_id"]
            if story not in cached_features:
                cached_features[story] = projector.transform(data.reader.features(story, ["english1000"], trim=False)["english1000"])
            slot = sources[example["source_id"]]["raw_feature_bin"]
            observed = torch.as_tensor(cached_features[story][slot - 4:slot].transpose(1, 0, 2))
            numerical_windows[example["source_id"]] = observed.numpy()[None]
            model.zero_grad(set_to_none=True)
            logits, trace = model(observed, example["inputs"], capture=True)
            loss = acceptable_loss(logits, example["acceptable_indices"])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError(f"Nonfinite real-query forward/backward: {family}/{op}")
            if family == "structured":
                if not trace["primitive"] or not torch.isfinite(trace["latent"]).all():
                    raise AssertionError("Structured execution omitted its spatial grounding trace.")
                if op != "concept" and not trace["relations"]:
                    raise AssertionError("Relational query omitted ordered-pair support.")
                if reviewed and example["inputs"]["ast"]["task"] in {"concept", "role", "reference", "scope", "polarity", "qualifier", "property"}:
                    query_ast = example["inputs"]["ast"]
                    primitives = {object_hash(p["description"]): p["scores"] for p in trace["primitive"]}
                    left = primitives[object_hash(query_ast["anchor"])]
                    rel = next(r for r in trace["relations"] if r["operation"] == query_ast["operation"])
                    pair = rel["left_factor"] @ rel["right_factor"].T / rel["scale"]
                    pair = pair.masked_fill(~(model.site_mask[:, None] & model.site_mask[None]), -1e4)
                    direct = []
                    for candidate in example["inputs"]["candidates"]:
                        right = primitives[object_hash(candidate)]
                        la, ra = left.softmax(0), right.softmax(0)
                        direct.append(la @ pair @ ra + .5 * (la @ left + ra @ right))
                    if not torch.allclose(logits, torch.stack(direct), atol=1e-5, rtol=1e-5):
                        raise AssertionError("Vectorized grounding changed the real candidate-choice scores")
            coverage[family + "/" + op] = {"query_id": example["query_id"], "candidate_count": len(logits), "finite_gradient": True}
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
            with h5py.File(Path(temporary) / "traces.h5", "w") as file:
                predicted = evaluate(model, list(selected.values()), numerical_windows, trace_file=file)
                if len(predicted) != len(selected) or decoder_metrics(predicted)["n_queries"] != len(selected):
                    raise AssertionError("Real query evaluation/repetition aggregation changed the query inventory.")
                if family == "structured" and any("site_mask" not in group["0"] for group in file.values()):
                    raise AssertionError("Serialized spatial trace lost its observed-site mask.")
                if family == "structured" and reviewed:
                    kinds = {"role": "role", "reference": "reference", "relation": "discourse", "identity": "identity",
                             "qualifier": "qualification", "property": "state_update"}
                    for task, kind in kinds.items():
                        example = selected[task]
                        records = [r for r in data.semantics.records(example["story_id"], "occurrences")
                                   if r["kind"] == kind and r["source_id"] == example["source_id"]]
                        signatures = [trace_signature(file[example["query_id"]]["0"], r, "grounding") for r in records]
                        signatures = [s for s in signatures if s is not None]
                        if not signatures or any(len(s) != int(projector.site_mask.sum()) ** 2 or not np.isfinite(s).all() for s in signatures):
                            raise AssertionError("Real ordered grounding profile did not survive serialization: " + task)
                        grounding_checks[task] = len(signatures)
        try:
            model.answer(None, {**next(iter(selected.values()))["inputs"], "answer": "forbidden"})
        except ValueError:
            pass
        else:
            raise AssertionError("Decoder accepted a private answer input.")
    if reviewed:
        candidates = next(iter(selected.values()))["inputs"]["candidates"]
        before = object_hash(candidates)
        try:
            candidates[0].clear()
        except TypeError:
            pass
        else:
            raise AssertionError("Shared public candidate storage was mutable")
        if object_hash(candidates) != before:
            raise AssertionError("Candidate descriptor changed during execution")
    support = EncodingSupport(data, comparison_spec(data.config, {"comparison": "concepts", "groups": ["presentation", "C"]}))
    factory = EncodingFeatures(data, ["story_01"], ["presentation", "C"], support=support)
    design, mask = factory.story("story_01")
    if len(mask) != data.reader.contract["subjects"]["subject01"]["story_01"]["timepoints"] - 20 or not mask.any():
        raise AssertionError("Actual response/feature timing contract failed.")
    rdm, valid, _ = cosine_rdm(values[20:30])
    observed = rdm[np.ix_(valid, valid)]
    comparison = rdm_comparison(observed, observed, permutations=31)
    if not np.isclose(comparison["spearman_r"], 1):
        raise AssertionError("Identical observed feature geometries disagree.")
    spatial = {}
    for subject in data.reader.contract["subjects"]:
        assignment = data.spatial.assignments(subject)
        if len(assignment) != data.reader.contract["subjects"][subject]["story_01"]["voxels"]:
            raise AssertionError("Native voxel/parcel assignment shape mismatch.")
        spatial[subject] = {"assigned": int((assignment >= 0).sum()), "total": len(assignment),
                            "observed_parcels": len(np.unique(assignment[assignment >= 0]))}
    for relative in ["analysis_data.py", "analysis_runs.py", "decoders.py", "decoder_fit.py", "encoding.py", "encoding_support.py", "geometry.py", "spatial.py"]:
        ast.parse((ROOT / "neurosym" / relative).read_text(encoding="utf-8"), feature_version=(3, 11))
    from package_analysis import MODULES
    verified_files = [f"neurosym/{name}.py" for name in MODULES] + ["scripts/run_analysis.py", "scripts/verify_analysis.py", "scripts/verify_encoding_support.py"]
    return {"status": "verified", "semantic_build_hash": data.semantics.build_hash,
            "config_hash": object_hash(data.config),
            "verified_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in verified_files},
            "annotation_protocols": sorted({s["annotation_protocol"] for s in sources.values()}),
            "partial_annotations": data.partial, "real_query_execution": coverage,
            "real_grounding_pair_profiles": grounding_checks,
            "group_ridge_max_primal_dual_error": float(np.max(np.abs(first - second))),
            "mse_sufficient_statistic_error": float(abs(exact - moments)), "spatial": spatial,
            "encoding_valid_rows_story01": int(mask.sum()),
            "checks": ["train-only local projector and serialization", "all observed required query operators across all 4 readouts",
                       "private-answer rejection", "group ridge primal/dual identity", "exact all-voxel tuning algebra",
                       "released feature/annotation time alignment", "real-feature geometry identity", "all 9 subject atlas contracts", "Python 3.11 syntax"],
            "scientific_fits_executed": False, "modern_frozen_states_executed": False,
            "scope": "Numerical and interface verification using actual released textual features and real compiled queries. These inputs are not represented as measured fMRI or modern model states."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, help="Defaults to the currently configured compiled snapshot")
    args = parser.parse_args()
    build = args.build or latest(ROOT / read_json(ROOT / "configs/analysis.json")["semantics"])
    result = verify(build)
    write_report(ROOT / "artifacts/analysis-verification.json", result)
    print(json.dumps(result, indent=2))
