"""Story-nested, group-regularized encoding of actual native-voxel responses."""
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg

from .analysis_runs import column_correlation, partition, run_directory, write_report
from .extraction import exclusive_run
from .io import read_json, save_json
from .semantic_features import SemanticVectorizer
from .temporal import checked_alignment, fir_design


class GroupScaler:
    """Training-only z scores, with equal total variance per feature group."""
    def fit(self, groups):
        self.parameters = {}
        for name, x in groups.items():
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError("Invalid encoding feature group.")
            mean, scale = x.mean(0), x.std(0)
            active = scale > 1e-8
            if not active.any():
                raise ValueError(f"Feature group {name} has no training variation.")
            self.parameters[name] = (mean, scale, active)
        return self

    def transform(self, groups):
        if set(groups) != set(self.parameters):
            raise ValueError("Encoding feature group inventory changed.")
        result = {}
        for name, x in groups.items():
            mean, scale, active = self.parameters[name]
            if x.shape[1] != len(mean) or not np.isfinite(x).all():
                raise ValueError("Encoding feature coordinates changed.")
            result[name] = ((x[:, active] - mean[active]) / scale[active] / np.sqrt(active.sum())).astype(np.float64)
        return result

    def save(self, path):
        np.savez_compressed(path, **{f"{name}_{key}": value for name, values in self.parameters.items()
                                    for key, value in zip(("mean", "scale", "active"), values, strict=True)})


class GroupRidge:
    """Sum of linear group kernels; penalties are alpha / group_weight."""
    def __init__(self, alpha, weights):
        if alpha <= 0 or not weights or any(v <= 0 for v in weights.values()):
            raise ValueError("Positive group penalties required.")
        self.alpha, self.weights = float(alpha), dict(weights)

    def kernel(self, first, second):
        if set(first) != set(self.weights) or set(second) != set(self.weights):
            raise ValueError("Kernel weights and groups disagree.")
        return sum(self.weights[g] * first[g] @ second[g].T for g in self.weights)

    def fit_design(self, groups):
        self.scaler = GroupScaler().fit(groups)
        self.training = self.scaler.transform(groups)
        kernel = self.kernel(self.training, self.training)
        kernel.flat[::len(kernel) + 1] += self.alpha
        self.factor = linalg.cho_factor(kernel, lower=True, check_finite=False)
        return self

    def prediction_operator(self, groups):
        cross = self.kernel(self.scaler.transform(groups), self.training)
        return linalg.cho_solve(self.factor, cross.T, check_finite=False).T

    def coefficients(self, y):
        mean = y.mean(0)
        return linalg.cho_solve(self.factor, y - mean, check_finite=False), mean


class EncodingFeatures:
    def __init__(self, data, train, groups, *, model=None, layer=None):
        data.validate_partition(train)
        if len(set(groups)) != len(groups) or not groups:
            raise ValueError("Choose a nonempty unique encoding group list.")
        legal = {"presentation", "legacy", "model", "C", "B", "BR", "GB", "GBR", "PB", "PBR", "R", "S", "D", "U", "L"}
        if set(groups) - legal or ("model" in groups and (model is None or layer is None)):
            raise ValueError("Unknown encoding group or unspecified model layer.")
        self.data, self.groups, self.train = data, list(groups), list(train)
        self.model, self.layer = model, layer
        self.vectorizers = {g: SemanticVectorizer.fit(data.semantics, train, groups=[g])
                            for g in groups if g not in {"presentation", "legacy", "model"}}
        self.coverage = {}

    def story(self, story):
        data = self.data
        timing = read_json(data.semantics.build / "stories" / story / "timing.json")
        rows, delays = timing["response_raw_indices"], timing["fir_delays_trs"]
        # The common mask remains identical across semantic/model ablations.
        known = np.asarray(timing["known_raw_rows"], dtype=bool)
        valid = np.ones(len(rows), dtype=bool)
        for delay in delays:
            indices = np.asarray(rows) - delay
            valid &= (indices >= 0) & known[np.maximum(indices, 0)]
        alignment = checked_alignment(data.root / "data/processed/alignment", story, data.index["content_hash"])
        word_known = np.asarray(alignment["words"]["known_raw_rows"], dtype=bool)
        for delay in delays:
            indices = np.asarray(rows) - delay
            valid &= (indices >= 0) & word_known[np.maximum(indices, 0)]
        groups, self.coverage[story] = {}, {}
        for group in self.groups:
            if group in self.vectorizers:
                values, mask, info = self.vectorizers[group].design(data.semantics, story)
                self.coverage[story][group] = info
            elif group == "model":
                values, mask = data.model(self.model).design(story, self.layer, kind=data.config["encoding"]["model_kind"])
            else:
                names = ["english1000"] if group == "legacy" else ["numwords", "numletters", "letters", "word_length_std", "pauses"]
                released = data.reader.features(story, names, trim=False)
                full = np.concatenate([released[n] for n in names], axis=1)
                if len(full) + 5 != timing["raw_response_rows"]:
                    raise ValueError("Released feature/end-silence convention changed.")
                # Five trailing silence rows; leading silence already exists in release.
                full = np.pad(full, ((0, 5), (0, 0)))
                values, mask = fir_design(full, rows, delays)
            groups[group] = values
            valid &= mask
        if not valid.any():
            raise ValueError(f"No jointly observed encoding rows in {story}.")
        return groups, valid

    def assemble(self, stories):
        records = {s: self.story(s) for s in stories}
        groups = {g: np.concatenate([records[s][0][g][records[s][1]] for s in stories]) for g in self.groups}
        return groups, {s: records[s][1] for s in stories}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for group, vectorizer in self.vectorizers.items():
            vectorizer.save(directory / (group + ".json"))
        save_json(directory / "features.json", {"groups": self.groups, "train": self.train,
                  "model": self.model, "layer": self.layer, "coverage": self.coverage})


def response_matrix(data, subject, stories, masks, section):
    """Training stories have one repeat; only explicit prediction metrics average repeats."""
    arrays = []
    for story in stories:
        y = data.reader.response(subject, story, section)
        if y.shape[0] != 1:
            raise ValueError("Do not pool heldout repeated recordings into a training matrix.")
        arrays.append(y[0, masks[story]])
    values = np.concatenate(arrays).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured training responses.")
    return values


def target_moments(data, subject, train, validation, masks, batch):
    """Exact all-voxel normalized-MSE tuning through sufficient statistics.

    Avoid materializing one prediction for every hyperparameter x voxel.
    Means/scales are fitted on training responses only, and constant targets excluded.
    """
    n, m = sum(masks[s].sum() for s in train), sum(masks[s].sum() for s in validation)
    gram, cross, total, count = np.zeros((n, n)), np.zeros((n, m)), 0.0, 0
    voxels = data.reader.contract["subjects"][subject][train[0]]["voxels"]
    for start in range(0, voxels, batch):
        section = slice(start, min(start + batch, voxels))
        y = response_matrix(data, subject, train, masks, section)
        z = response_matrix(data, subject, validation, masks, section)
        mean, scale = y.mean(0), y.std(0)
        use = scale > 1e-8
        y, z = (y[:, use] - mean[use]) / scale[use], (z[:, use] - mean[use]) / scale[use]
        gram += y @ y.T
        cross += y @ z.T
        total += float(np.sum(z * z))
        count += int(use.sum())
    if not count:
        raise ValueError("No variable training voxels.")
    return gram / count, cross / count, total / count


def select_encoding(data, options, split):
    cfg, groups = data.config["encoding"], options["groups"]
    mixtures = [np.ones(len(groups)) / len(groups)]
    if len(groups) > 1:
        mixtures.extend(np.random.default_rng(options["seed"]).dirichlet(np.ones(len(groups)), cfg["group_mixtures"]))
    grid = [{"alpha": float(a), "weights": dict(zip(groups, w.tolist(), strict=True))}
            for w in mixtures for a in cfg["alphas"]]
    scores = [[] for _ in grid]
    for inner_index, fold in enumerate(split["inner"]):
        factory = EncodingFeatures(data, fold["train"], groups, model=options.get("model"), layer=options.get("layer"))
        x, train_masks = factory.assemble(fold["train"])
        z, val_masks = factory.assemble(fold["validation"])
        moments = target_moments(data, options["subject"], fold["train"], fold["validation"],
                                 {**train_masks, **val_masks}, cfg["target_batch"])
        scaler = GroupScaler().fit(x)
        tx, vz = scaler.transform(x), scaler.transform(z)
        for mixture_index, weights in enumerate(mixtures):
            weight_map = dict(zip(groups, weights, strict=True))
            ridge = GroupRidge(1, weight_map)
            eigenvalues, vectors = linalg.eigh(ridge.kernel(tx, tx), check_finite=False)
            cross_vectors = ridge.kernel(vz, tx) @ vectors
            for alpha_index, alpha in enumerate(cfg["alphas"]):
                operator = (cross_vectors / (np.maximum(eigenvalues, 0) + alpha)) @ vectors.T
                loss = (np.sum((operator @ moments[0]) * operator) - 2 * np.sum(operator * moments[1].T) + moments[2]) / len(operator)
                scores[mixture_index * len(cfg["alphas"]) + alpha_index].append(float(loss))
        print(f"ENCODING SELECT inner={inner_index + 1}/{len(split['inner'])}", flush=True)
    means = [float(np.mean(s)) for s in scores]
    best = int(np.argmin(means))
    return grid[best], {"criterion": "story-macro MSE normalized by training-only voxel variance; all variable native voxels",
                        "grid": grid, "inner_scores": scores, "mean_loss": means, "selected": best}


def run_encoding(data, options):
    split = partition(data.semantics, options["fold"])
    data.validate_partition(split["train"], test=split["test"])
    directory, identity = run_directory(data, "encoding", options)
    with exclusive_run(directory / "RUNNING.lock"):
        if (directory / "complete.json").exists():
            return directory
        selected, selection = select_encoding(data, options, split)
        write_report(directory / "selection.json", selection)
        factory = EncodingFeatures(data, split["train"], options["groups"], model=options.get("model"), layer=options.get("layer"))
        x, masks = factory.assemble(split["train"])
        test = {s: factory.story(s) for s in split["test"]}
        ridge = GroupRidge(**selected).fit_design(x)
        factory.save(directory / "features")
        ridge.scaler.save(directory / "scaler.npz")
        kernels = {s: ridge.kernel(ridge.scaler.transform({g: v[mask] for g, v in groups.items()}), ridge.training)
                   for s, (groups, mask) in test.items()}
        voxels = data.reader.contract["subjects"][options["subject"]][split["train"][0]]["voxels"]
        batch, summaries = data.config["encoding"]["target_batch"], {}
        with h5py.File(directory / "encoding.h5", "w") as file:
            file.attrs["complete"] = False
            for group, values in ridge.training.items():
                file.create_dataset("training_features/" + group, data=values.astype(np.float32), compression="lzf")
            dual = file.create_dataset("dual", (len(next(iter(x.values()))), voxels), dtype="f4", compression="lzf")
            means = file.create_dataset("target_mean", (voxels,), dtype="f4")
            for story, (_, mask) in test.items():
                root = file.create_group(story)
                root.create_dataset("response_rows", data=np.flatnonzero(mask))
                root.create_dataset("prediction", (mask.sum(), voxels), dtype="f4", compression="lzf")
                repeats = data.reader.contract["subjects"][options["subject"]][story]["repeats"]
                root.create_dataset("repeat_correlation", (repeats, voxels), dtype="f4")
                root.create_dataset("repeat_mse", (repeats, voxels), dtype="f4")
                root.create_dataset("mean_response_correlation", (voxels,), dtype="f4")
                if repeats == 2:
                    root.create_dataset("repeat_reliability", (voxels,), dtype="f4")
                for group in options["groups"]:
                    root.create_dataset("contributions/" + group, (mask.sum(), voxels), dtype="f4", compression="lzf")
            for start in range(0, voxels, batch):
                section = slice(start, min(start + batch, voxels))
                y = response_matrix(data, options["subject"], split["train"], masks, section)
                coefficients, mean = ridge.coefficients(y)
                dual[:, section], means[section] = coefficients, mean
                for story, (groups, mask) in test.items():
                    prediction = kernels[story] @ coefficients + mean
                    target = data.reader.response(options["subject"], story, section)[:, mask]
                    root = file[story]
                    root["prediction"][:, section] = prediction
                    for repeat in range(len(target)):
                        root["repeat_correlation"][repeat, section] = column_correlation(prediction, target[repeat])
                        root["repeat_mse"][repeat, section] = ((prediction - target[repeat]) ** 2).mean(0)
                    root["mean_response_correlation"][section] = column_correlation(prediction, target.mean(0))
                    if len(target) == 2:
                        root["repeat_reliability"][section] = column_correlation(target[0], target[1])
                    transformed = ridge.scaler.transform({g: v[mask] for g, v in groups.items()})
                    for group in options["groups"]:
                        root["contributions/" + group][:, section] = selected["weights"][group] * transformed[group] @ ridge.training[group].T @ coefficients
                print(f"ENCODING FIT voxels={section.stop}/{voxels}", flush=True)
            for story in test:
                summaries[story] = {"mean_voxel_r": float(np.nanmean(file[story]["mean_response_correlation"][()])),
                                    "valid_rows": int(test[story][1].sum()), "voxels": voxels}
            file.attrs["complete"] = True
        write_report(directory / "complete.json", {"identity": identity, "partition": split,
                     "selected": selected, "stories": summaries,
                     "interpretation": "Heldout prediction of measured native voxels. Per-group contributions are model-dependent, not causal effects."})
    return directory
