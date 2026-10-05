"""Separate measured, grounding, latent and encoding-implied semantic geometries.

No coordinate averaging across independently fitted latent spaces. Missing
signatures and zero norms are unavailable measurements, never zero distances.
"""
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr

from .analysis_runs import partition, run_directory, write_report
from .extraction import exclusive_run
from .io import object_hash, read_json


def cosine_rdm(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite signature matrix required.")
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-12
    unit = np.divide(values, norms[:, None], out=np.zeros_like(values), where=valid[:, None])
    result = np.clip(1 - unit @ unit.T, 0, 2)
    result[~(valid[:, None] & valid[None])] = np.nan
    return result, valid, norms


def rdm_comparison(first, second, *, permutations=10000, seed=11):
    """Concept-label permutation; shared RDM cells are not independent trials."""
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 3:
        raise ValueError("At least three matched semantic items in square RDMs required.")
    upper = np.triu_indices(len(a), 1)
    mask = np.isfinite(a[upper]) & np.isfinite(b[upper])
    if mask.sum() < 3:
        raise ValueError("Insufficient observed, nonzero signatures for RDM comparison.")
    if not np.isfinite(a[upper]).all() or not np.isfinite(b[upper]).all():
        raise ValueError("Remove unavailable semantic items before label permutation.")
    observed = float(spearmanr(a[upper], b[upper]).statistic)
    if not np.isfinite(observed):
        return {"available": False, "reason": "constant pairwise dissimilarities"}
    rng, exceed = np.random.default_rng(seed), 0
    for _ in range(permutations):
        order = rng.permutation(len(a))
        value = float(spearmanr(a[upper], b[np.ix_(order, order)][upper]).statistic)
        exceed += abs(value) >= abs(observed)
    return {"available": True, "spearman_r": observed, "two_sided_label_permutation_p": (1 + exceed) / (1 + permutations),
            "permutations": permutations, "seed": seed, "items": len(a), "pairs": int(mask.sum()),
            "scope": "Conditional association for these semantic items and fitted views; not population-level brain inference."}


def crossed_interval(records, *, samples=10000, seed=11):
    """Paired effect intervals resampling participant and story clusters separately."""
    cells = defaultdict(list)
    for record in records:
        value = float(record["difference"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite paired contrast.")
        cells[(record["subject"], record["story"])].append(value)
    subjects = sorted({k[0] for k in cells})
    stories = sorted({k[1] for k in cells})
    if len(subjects) < 2 or len(stories) < 2:
        raise ValueError("Crossed interval requires multiple participants AND independent stories.")
    if any((s, t) not in cells for s in subjects for t in stories):
        raise ValueError("A complete paired subject-story panel is required; do not silently impute missing cells.")
    matrix = np.array([[np.mean(cells[(s, t)]) for t in stories] for s in subjects])
    rng = np.random.default_rng(seed)
    draws = [matrix[np.ix_(rng.integers(len(subjects), size=len(subjects)),
                           rng.integers(len(stories), size=len(stories)))].mean() for _ in range(samples)]
    return {"mean_difference": float(matrix.mean()), "percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "participants": len(subjects), "stories": len(stories), "samples": samples, "seed": seed,
            "unit": "crossed participant and story clusters; repetitions/seeds averaged within cells"}


def fdr_bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite p values in [0,1] required.")
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def semantic_occurrences(data, stories, kind, *, reviewed_only=False):
    records, seen = [], set()
    for story in stories:
        for record in data.semantics.records(story, "occurrences"):
            if record["kind"] != kind or record.get("grounding_resolved") is False:
                continue
            if reviewed_only and record["annotation_review_status"] not in {"reviewed", "adjudicated", "human_reviewed"}:
                continue
            key = record["source_id"], object_hash({"kind": kind, "label": record["label"]})
            if key not in seen:
                records.append({**record, "item_id": key[1]})
                seen.add(key)
    return records


def summarize_geometry(records, vectors, directory, *, min_stories=2):
    """One occurrence per source/item, then equal story means, never question counts."""
    if min_stories < 1:
        raise ValueError("Positive minimum context support required.")
    cells, labels, counts = defaultdict(list), {}, defaultdict(int)
    for record in records:
        key = (record["source_id"], record["item_id"])
        if key not in vectors:
            continue
        vector = np.asarray(vectors[key], dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Signature coordinates must be finite and one-dimensional.")
        cells[(record["item_id"], record["story_id"])].append(vector)
        labels[record["item_id"]] = record["label"]
        counts[record["item_id"]] += 1
    story_means = {key: np.mean(values, axis=0) for key, values in cells.items()}
    contexts = {item: sorted(story for key, story in cells if key == item) for item in labels}
    items = sorted(item for item in labels if len(contexts[item]) >= min_stories)
    if not items:
        raise ValueError("No real semantic items satisfy the requested independent-story support.")
    signatures = np.stack([np.mean([story_means[(item, s)] for s in contexts[item]], axis=0) for item in items])
    rdm, valid, norms = cosine_rdm(signatures)
    reliability = {"available": False, "reason": "Too few items shared across independent story halves."}
    stories = sorted({story for _, story in cells})
    halves = [set(stories[::2]), set(stories[1::2])]
    shared = [item for item in items if all(set(contexts[item]) & half for half in halves)]
    if len(shared) >= 3:
        half_vectors = [np.stack([np.mean([story_means[(item, s)] for s in contexts[item] if s in half], 0)
                                 for item in shared]) for half in halves]
        first, a, _ = cosine_rdm(half_vectors[0])
        second, b, _ = cosine_rdm(half_vectors[1])
        common = np.flatnonzero(a & b)
        if len(common) >= 3:
            upper = np.triu_indices(len(common), 1)
            rho = float(spearmanr(first[np.ix_(common, common)][upper], second[np.ix_(common, common)][upper]).statistic)
            reliability = {"available": bool(np.isfinite(rho)), "split_story_rdm_spearman": rho,
                           "items": len(common), "halves": [sorted(h) for h in halves],
                           "scope": "Context stability, not a noise ceiling; one fixed split, no disattenuation."}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory / "geometry.h5", "w") as file:
        file.attrs["complete"] = False
        file.create_dataset("signatures", data=signatures.astype(np.float32), compression="lzf")
        file.create_dataset("rdm", data=rdm)
        file.create_dataset("valid", data=valid)
        file.create_dataset("norms", data=norms)
        for item, story in sorted(story_means):
            file.create_dataset("story_means/" + item + "/" + story, data=story_means[(item, story)].astype(np.float32), compression="lzf")
        file.attrs["complete"] = True
    report = {"items": items, "labels": {i: labels[i] for i in items},
              "source_counts": {i: counts[i] for i in items}, "contexts": {i: contexts[i] for i in items},
              "minimum_stories": min_stories, "zero_norm_items": [i for i, ok in zip(items, valid, strict=True) if not ok],
              "available_source_item_pairs": len(vectors), "requested_source_item_pairs": len(records),
              "signature_dimensions": signatures.shape[1], "reliability": reliability,
              "aggregation": "deduplicate source/item; equal weight per story; exact labels/senses retained"}
    write_report(directory / "geometry.json", report)
    return report


def native_source_vectors(data, options, split):
    modality = options["modality"]
    if modality == "brain":
        # Standardize each measured native voxel on training recordings only.
        arrays = [data.brain(options["subject"], s)[0] for s in split["train"]]
    elif modality == "model":
        arrays = [data.states(options["model"], options["layer"], s) for s in split["train"]]
    else:
        raise ValueError("Unknown native modality.")
    n = sum(len(a) for a in arrays)
    mean = sum(a.astype(np.float64).sum(0) for a in arrays) / n
    variance = sum(((a.astype(np.float64) - mean) ** 2).sum(0) for a in arrays) / n
    active = variance > 1e-16
    scale = np.sqrt(variance[active])
    result = {}
    selected_stories = split["train"] + split["test"] if options.get("residualize_presentation") else split["test"]
    for story in selected_stories:
        observations = data.brain(options["subject"], story) if modality == "brain" else data.states(options["model"], options["layer"], story)
        for source in data.semantics.records(story, "sources"):
            if not source["decoder_timing_eligible"]:
                continue
            value = (observations[:, source["decoder_trimmed_response_rows"]].mean(axis=(0, 1)) if modality == "brain"
                     else observations[source["model_unit_row"]])
            result[source["id"]] = ((value[active] - mean[active]) / scale).astype(np.float32)
    parameters = {"mean": mean, "active": active, "scale": scale}
    if options.get("residualize_presentation"):
        nuisance, train_ids, test_ids = {}, [], []
        for story in selected_stories:
            text = data.story(story)
            for source in data.semantics.records(story, "sources"):
                if source["id"] in result:
                    start, end = source["local_word_span"]
                    words = [w["text"] for w in text["words"][start:end]]
                    letters = "".join(words).casefold()
                    lengths = [len(w) for w in words]
                    # Prefix-only control information; no post-endpoint words
                    # may enter the residualized frozen-model representation.
                    nuisance[source["id"]] = np.array([end, len(words), len(letters), np.std(lengths), source["available_at_seconds"],
                        *[letters.count(chr(97 + i)) for i in range(26)]], dtype=np.float64)
                    (train_ids if story in split["train"] else test_ids).append(source["id"])
        if len(train_ids) < 2 or not test_ids:
            raise ValueError("Insufficient observed presentation controls for training/test source windows.")
        x, y = np.stack([nuisance[s] for s in train_ids]).astype(np.float64), np.stack([result[s] for s in train_ids])
        center, scaling = x.mean(0), x.std(0)
        varying = scaling > 1e-8
        x = (x[:, varying] - center[varying]) / scaling[varying]
        target_mean = y.mean(0)
        # Fixed shrinkage specified before seeing test geometry; no test-fitted residualization.
        coefficients = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (y - target_mean))
        result = {s: result[s] - target_mean - ((nuisance[s][varying] - center[varying]) / scaling[varying]) @ coefficients
                  for s in test_ids}
        parameters.update(nuisance_mean=center, nuisance_scale=scaling, nuisance_active=varying,
                          nuisance_target_mean=target_mean, nuisance_coefficients=coefficients)
    return result, parameters


def matches(description, label):
    """Do not conflate distinct non-null senses or fillers absent in a descriptor."""
    return all(value is None or description.get(key) == value for key, value in label.items())


def trace_signature(group, record, view):
    if view == "latent":
        return group["latent"][()] if "latent" in group else None
    kind, label = record["kind"], record["label"]
    if "site_mask" not in group:
        return None
    mask = group["site_mask"][()].astype(bool)
    primitives = [(json.loads(child.attrs["description"]), child["scores"][()]) for child in group["primitive"].values()]
    if kind in {"concept", "predicate"}:
        values = [scores for desc, scores in primitives if matches(desc, label)]
        return np.mean(values, axis=0)[mask] if values else None
    if kind not in {"role", "discourse"}:
        raise ValueError("Grounding profiles support concept, predicate, role and discourse; use native/latent/encoding-implied for other labels.")
    if kind == "role":
        anchor, filler = label["predicate"], label["filler"]
        if "event" in filler:
            filler = filler["event"]
        if "literal" in filler:
            filler = filler["literal"]
        relation_label = {"op": "role", "role": label["role"]}
    else:
        anchor, filler = label["source"], label["target"]
        relation_label = {"op": "event_relation", "relation": label["relation"], "direction": "out",
                          "contexts": record.get("context_selectors", [])}
    left = [scores for desc, scores in primitives if matches(desc, anchor)]
    right = [scores for desc, scores in primitives if matches(desc, filler)]
    if not left or not right:
        return None
    for child in group["relations"].values():
        if json.loads(child.attrs["operation"]) == relation_label:
            pair = child["left_factor"][()][mask] @ child["right_factor"][()][mask].T / json.loads(child.attrs["scale"])
            la, ra = softmax(np.mean(left, 0)[mask]), softmax(np.mean(right, 0)[mask])
            return (la[:, None] * pair * ra[None]).reshape(-1)
    return None


def trace_vectors(path, records, view):
    by_source, collected = defaultdict(list), defaultdict(dict)
    for record in records:
        by_source[record["source_id"]].append(record)
    with h5py.File(Path(path) / "traces.h5", "r") as file:
        if not file.attrs["complete"]:
            raise ValueError("Incomplete decoder evidence.")
        for repeats in file.values():
            for group in repeats.values():
                source = group.attrs["source_id"]
                for record in by_source[source]:
                    vector = trace_signature(group, record, view)
                    if vector is None:
                        continue
                    # Repeating the same query/candidate profile does not increase support.
                    digest = object_hash(np.asarray(vector).tolist())
                    collected[(source, record["item_id"])][digest] = vector
    return {key: np.mean(list(values.values()), axis=0) for key, values in collected.items()}


def implied_vectors(path, data, stories, component=None):
    result = {}
    with h5py.File(Path(path) / "encoding.h5", "r") as file:
        if not file.attrs["complete"]:
            raise ValueError("Incomplete fitted encoding output.")
        for story in stories:
            group = file[story]
            row_map = {int(row): index for index, row in enumerate(group["response_rows"][()])}
            matrix = group["contributions/" + component] if component else group["prediction"]
            for source in data.semantics.records(story, "sources"):
                rows = source["decoder_trimmed_response_rows"]
                if source["decoder_timing_eligible"] and all(r in row_map for r in rows):
                    result[source["id"]] = matrix[sorted(row_map[r] for r in rows)].mean(0)
    return result


def run_geometry(data, options):
    split = partition(data.semantics, options["fold"])
    data.validate_partition(split["train"], test=split["test"])
    records = semantic_occurrences(data, split["test"], options["kind"], reviewed_only=options.get("reviewed_only", False))
    if options["view"] != "native":
        parent = Path(options["from_run"])
        receipt = read_json(parent / "complete.json")
        if (receipt["identity"]["semantic_build_hash"] != data.semantics.build_hash or
                receipt["partition"]["test"] != split["test"] or
                not set(receipt["partition"]["train"]) <= set(split["train"])):
            raise ValueError("Geometry must use the identical semantic build and heldout partition of its parent fit.")
        expected = "encoding" if options["view"] == "encoding-implied" else "decoder"
        if receipt["identity"]["kind"] != expected:
            raise ValueError("Wrong parent fit family for this geometry.")
        parent_options = receipt["identity"]["options"]
        for field in ("subject", "model", "layer"):
            if options.get(field) != parent_options.get(field):
                raise ValueError("Geometry observation identity differs from parent fit: " + field)
        if options["view"] == "grounding" and parent_options.get("family") != "structured":
            raise ValueError("Only the structured decoder supplies parcel-local grounding traces.")
        split = receipt["partition"]
        options = {**options, "parent_identity_hash": object_hash(receipt["identity"])}
    directory, identity = run_directory(data, "geometry", options)
    with exclusive_run(directory / "RUNNING.lock"):
        if (directory / "complete.json").exists():
            return directory
        if options["view"] == "native":
            sources, scaling = native_source_vectors(data, options, split)
            np.savez_compressed(directory / "native_scaler.npz", **scaling)
            vectors = {(r["source_id"], r["item_id"]): sources[r["source_id"]] for r in records if r["source_id"] in sources}
        elif options["view"] in {"grounding", "latent"}:
            vectors = trace_vectors(options["from_run"], records, options["view"])
            decoder = read_json(Path(options["from_run"]) / "decoder.json")
            mask = np.asarray(decoder["site_mask"], dtype=bool)
            sites = ([name for name, ok in zip(data.spatial.names, mask, strict=True) if ok] if options["modality"] == "brain"
                     else [f"coordinate_group_{i}" for i in np.flatnonzero(mask)])
            write_report(directory / "coordinates.json", {"sites": sites, "parent_fit": options["parent_identity_hash"],
                         "layout": "ordered source-site x target-site, row-major" if options["kind"] in {"role", "discourse"} and options["view"] == "grounding"
                         else "local site scores" if options["view"] == "grounding" else "query-conditioned latent coordinates; only this fitted basis"})
        elif options["view"] == "encoding-implied":
            sources = implied_vectors(options["from_run"], data, split["test"], options.get("component"))
            vectors = {(r["source_id"], r["item_id"]): sources[r["source_id"]] for r in records if r["source_id"] in sources}
        else:
            raise ValueError("Unknown geometry family.")
        report = summarize_geometry(records, vectors, directory, min_stories=options["min_stories"])
        write_report(directory / "complete.json", {"identity": identity, "partition": split, "geometry": report,
                     "interpretation": "Geometry of labelled contexts, not isolated causal concept responses. Co-occurring meaning and lexical content remain possible explanations."})
    return directory


def compare_geometry(first, second, *, permutations=10000, seed=11):
    first, second = Path(first), Path(second)
    a, b = read_json(first / "complete.json"), read_json(second / "complete.json")
    if a["identity"]["semantic_build_hash"] != b["identity"]["semantic_build_hash"]:
        raise ValueError("Cannot match geometry from different annotation snapshots.")
    if a["partition"]["test"] != b["partition"]["test"]:
        raise ValueError("Compare the same observed heldout stories.")
    am, bm = read_json(first / "geometry.json"), read_json(second / "geometry.json")
    with h5py.File(first / "geometry.h5", "r") as af, h5py.File(second / "geometry.h5", "r") as bf:
        common = sorted(set(i for i, ok in zip(am["items"], af["valid"][()], strict=True) if ok) &
                        set(i for i, ok in zip(bm["items"], bf["valid"][()], strict=True) if ok))
        ai, bi = [am["items"].index(i) for i in common], [bm["items"].index(i) for i in common]
        result = rdm_comparison(af["rdm"][()][np.ix_(ai, ai)], bf["rdm"][()][np.ix_(bi, bi)], permutations=permutations, seed=seed)
    return {"first": object_hash(a["identity"]), "second": object_hash(b["identity"]), "common_items": common,
            "first_only": len(am["items"]) - len(common), "second_only": len(bm["items"]) - len(common), **result}


def grounding_stability(first, second, *, permutations=10000, seed=11):
    """Common-atlas map agreement, with semantic-label rather than site shuffling."""
    first, second = Path(first), Path(second)
    receipts = [read_json(p / "complete.json") for p in (first, second)]
    for receipt in receipts:
        options = receipt["identity"]["options"]
        if options["view"] != "grounding" or options["modality"] != "brain":
            raise ValueError("Anatomical map stability requires two brain grounding geometries.")
    if receipts[0]["identity"]["semantic_build_hash"] != receipts[1]["identity"]["semantic_build_hash"]:
        raise ValueError("Map labels belong to different semantic snapshots.")
    metadata = [read_json(p / "geometry.json") for p in (first, second)]
    coordinates = [read_json(p / "coordinates.json") for p in (first, second)]
    if coordinates[0]["layout"] != coordinates[1]["layout"]:
        raise ValueError("Cannot compare unary maps with ordered pair maps.")
    sites = sorted(set(coordinates[0]["sites"]) & set(coordinates[1]["sites"]))
    items = sorted(set(metadata[0]["items"]) & set(metadata[1]["items"]))
    if len(sites) < 3 or len(items) < 3:
        raise ValueError("Map stability needs at least three common sites and semantic items.")
    matrices = []
    pairwise = coordinates[0]["layout"].startswith("ordered")
    for p, meta, coords in zip((first, second), metadata, coordinates, strict=True):
        with h5py.File(p / "geometry.h5", "r") as file:
            values = file["signatures"][()][[meta["items"].index(i) for i in items]]
        select = [coords["sites"].index(s) for s in sites]
        if pairwise:
            values = values.reshape(len(items), len(coords["sites"]), len(coords["sites"]))[:, select][:, :, select].reshape(len(items), -1)
        else:
            values = values[:, select]
        values = values - values.mean(1, keepdims=True)
        matrices.append(values)
    norms = [np.linalg.norm(v, axis=1) for v in matrices]
    valid = (norms[0] > 1e-12) & (norms[1] > 1e-12)
    if valid.sum() < 3:
        raise ValueError("Insufficient variable common maps.")
    a, b = [v[valid] / n[valid, None] for v, n in zip(matrices, norms, strict=True)]
    correlations = np.sum(a * b, axis=1)
    observed, exceed, rng = float(correlations.mean()), 0, np.random.default_rng(seed)
    for _ in range(permutations):
        null = float(np.sum(a * b[rng.permutation(len(b))], axis=1).mean())
        exceed += null >= observed
    return {"common_sites": sites, "items": [i for i, ok in zip(items, valid, strict=True) if ok],
            "item_map_correlations": correlations.tolist(), "mean_map_correlation": observed,
            "semantic_label_permutation_p": (exceed + 1) / (permutations + 1), "permutations": permutations,
            "scope": "Conditional reproducibility and label specificity of decoder maps. Parcel autocorrelation is retained; this does not establish anatomical necessity."}
