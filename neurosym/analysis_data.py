"""Shared real-data observations and training-fitted local representation adapters."""
from pathlib import Path

import numpy as np
from scipy import linalg

from .dataset import DenizReader
from .compute import FoldCache, source_identity
from .storage import ResidentCache
from .extraction_inputs import checked_index, checked_story
from .io import object_hash, read_json
from .model_features import FrozenFeatureReader
from .semantic_features import SemanticDataset, freeze_descriptor
from .spatial import SpatialView
from .pca import fit_statistics, story_statistics, transform_gpu
from .runtime import deadline


def latest(directory):
    directory = Path(directory)
    return directory / read_json(directory / "latest.json")["path"]


class SiteProjector:
    """Independent local PCA. No cross-site mixing; training-only means/ranks."""
    def __init__(self, assignment, components):
        self.assignment = np.asarray(assignment, dtype=np.int32)
        if self.assignment.ndim != 1 or components < 1 or not np.any(self.assignment >= 0):
            raise ValueError("Invalid site partition or component count.")
        self.components = int(components)
        self.sites = int(self.assignment.max()) + 1
        self.parameters = []

    @property
    def nbytes(self):
        return self.assignment.nbytes + sum(v.nbytes for group in self.parameters for v in group)

    def fit(self, observations, training_stories):
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.assignment) or len(values) < 2:
            raise ValueError("Local projector observations/partition disagree.")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite observations require an explicit upstream mask.")
        self.training_stories = list(training_stories)
        self.parameters = []
        for site in range(self.sites):
            positions = np.flatnonzero(self.assignment == site)
            if not len(positions):
                self.parameters.append((positions, np.empty(0), np.empty(0), np.empty((0, 0))))
                continue
            local = values[:, positions]
            mean, scale = local.mean(0), local.std(0)
            use = scale > 1e-8
            positions, mean, scale = positions[use], mean[use], scale[use]
            local = (local[:, use] - mean) / scale
            count = min(self.components, len(local) - 1, len(positions))
            if count:
                covariance = local.T @ local / (len(local) - 1)
                eigenvalues, vectors = linalg.eigh(covariance, subset_by_index=(len(positions) - count, len(positions) - 1))
                vectors = vectors[:, eigenvalues > 1e-8][:, ::-1]
                # Fixed signs make serialization deterministic for distinct eigenvalues.
                if vectors.shape[1]:
                    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
                    vectors *= signs
            else:
                vectors = np.empty((len(positions), 0))
            self.parameters.append((positions, mean, scale, vectors))
        return self

    @property
    def site_mask(self):
        if len(self.parameters) != self.sites:
            raise ValueError("Projector has not been fitted.")
        return np.array([p[3].shape[1] > 0 for p in self.parameters])

    def transform(self, values, *, device='cpu', batch_sites=16):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != len(self.assignment) or not np.isfinite(values).all():
            raise ValueError("Projector input is missing, nonfinite or has another coordinate system.")
        if len(self.parameters) != self.sites:
            raise ValueError("Projector has not been fitted.")
        if str(device).startswith('cuda'):
            return transform_gpu(self, values, device=device, batch_sites=batch_sites)
        result = np.zeros((len(values), self.sites, self.components), dtype=np.float32)
        for site, (positions, mean, scale, vectors) in enumerate(self.parameters):
            result[:, site, :vectors.shape[1]] = ((values[:, positions] - mean) / scale) @ vectors
        return result

    def save(self, directory):
        from .io import save_json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"assignment": self.assignment}
        for i, values in enumerate(self.parameters):
            for key, value in zip(("positions", "mean", "scale", "vectors"), values, strict=True):
                arrays[f"{i}_{key}"] = value
        np.savez_compressed(directory / "projector.npz", **arrays)
        save_json(directory / "projector.json", {"training_stories": self.training_stories,
                  "components": self.components, "sites": self.sites,
                  "ranks": [p[3].shape[1] for p in self.parameters]})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = read_json(directory / "projector.json")
        with np.load(directory / "projector.npz", allow_pickle=False) as file:
            result = cls(file["assignment"], record["components"])
            result.training_stories = record["training_stories"]
            result.parameters = [tuple(file[f"{i}_{key}"].copy() for key in ("positions", "mean", "scale", "vectors"))
                                 for i in range(record["sites"])]
        return result


class AnalysisData:
    def __init__(self, root, config_path=None, build=None, *, allow_partial=False, require_prepared=False):
        self.root = Path(root).resolve()
        self.config = read_json(Path(config_path) if config_path else self.root / "configs/analysis.json")
        self.semantics = SemanticDataset(Path(build) if build else latest(self.root / self.config["semantics"]))
        if (self.config["require_complete_annotations"] and not allow_partial and
                not all(s["coverage_complete"] for s in self.semantics.identity["coverage"])):
            raise ValueError("Final analyses require complete annotations. Explicit --allow-partial labels a development run.")
        self.partial = not all(s["coverage_complete"] for s in self.semantics.identity["coverage"])
        self.reader = DenizReader(self.root / self.config["raw"], self.root / self.config["contract"])
        self.spatial = SpatialView(latest(self.root / self.config["spatial"]))
        if self.spatial.identity["contract_hash"] != object_hash(self.reader.contract):
            raise ValueError("Spatial assignments and neural data use different release contracts.")
        self.index = checked_index(self.root / self.semantics.identity["config"]["corpus"])
        if self.index["content_hash"] != self.semantics.identity["corpus_hash"]:
            raise ValueError("Analysis corpus differs from semantic build.")
        self._brain, self._models, self._states = {}, {}, {}
        self.compute_config = read_json(self.root / 'configs/compute.json')
        policy = self.compute_config['storage']
        if policy['encoding_output'] != 'response-operators-fp64':
            raise ValueError('Unsupported encoding storage representation.')
        self.cache = FoldCache(self.root / 'data/processed/analysis-cache-v2', require=require_prepared, policy=policy)
        self.host_cache = ResidentCache(policy['host_reuse_gib'] * 2**30)
        self.raw_cache = ResidentCache(policy['raw_reuse_gib'] * 2**30)
        self.device_caches = {}
        self._cache_identities = {}

    def resident(self, device='cpu'):
        if device == 'cpu':
            return self.host_cache
        if device not in self.device_caches:
            self.device_caches[device] = ResidentCache(self.compute_config['storage']['device_reuse_gib'] * 2**30)
        return self.device_caches[device]

    def cache_identity(self, model=None):
        if model not in self._cache_identities:
            self._cache_identities[model] = source_identity(self, model)
        return self._cache_identities[model]

    def validate_partition(self, train, validation=(), test=()):
        sets = [set(train), set(validation), set(test)]
        if not sets[0] or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Training/validation/test stories must be disjoint.")
        if not (sets[0] | sets[1]) <= set(self.semantics.splits["development"]):
            raise ValueError("Heldout stories cannot fit or select an analysis.")
        if not set.union(*sets) <= set(self.semantics.story_ids):
            raise ValueError("Unknown analysis story.")

    def brain(self, subject, story):
        key = subject, story
        def read():
            spec = self.reader.contract["subjects"][subject][story]
            relative = f"responses/{subject}_reading_fmri_data_{spec['split']}.hdf"
            path = self.reader.root / relative
            expected = next(s for s in self.reader.contract["source_files"] if s["path"] == relative)
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise FileNotFoundError("Real completed response file required: " + str(path))
            values = self.reader.response(subject, story)
            if not np.isfinite(values).all():
                raise ValueError("Actual responses include nonfinite measurements.")
            return values
        return self.raw_cache.get(('brain', *key), read)

    def model(self, model):
        if model not in self._models:
            reader = FrozenFeatureReader(self.root, model)
            alignment = read_json(self.root / reader.config["alignment"] / "index.json")
            source = read_json(reader.source / "run.json")
            if (source["corpus_hash"] != self.index["content_hash"] or
                    reader.aligned_run["alignment_hash"] != alignment["content_hash"] or
                    reader.aligned_run["source_run_hash"] != object_hash(source)):
                raise ValueError("Model extraction, timing and semantic corpus identities disagree.")
            self._models[model] = reader
        return self._models[model]

    def states(self, model, layer, story):
        key = model, layer, story
        return self.raw_cache.get(('model', *key), lambda: self.model(model).events(story, layer, kind='units'))

    def response(self, subject, story, section=slice(None)):
        """Share bounded raw recordings across voxel blocks and related fits."""
        return self.brain(subject, story)[:, :, section]

    def projector(self, modality, train, *, subject=None, model=None, layer=None):
        self.validate_partition(train)
        identity = {'inputs': self.cache_identity(model), 'modality': modality, 'subject': subject,
                    'model': model, 'layer': layer, 'train': list(train),
                    'components': self.config['components_per_site'],
                    'coordinate_groups': self.config['model_coordinate_groups'],
                    'coordinate_seed': self.config['coordinate_partition_seed'],
                    'preparation': self.compute_config['preparation']}
        def build(path):
            self._fit_projector(modality, train, subject=subject, model=model, layer=layer).save(path)
        def load():
            path = self.cache.entry('projector', identity, build)
            result = SiteProjector.load(path)
            result.cache_identity = identity
            return result
        return self.host_cache.get(('projector', object_hash(identity)), load)

    def _fit_projector(self, modality, train, *, subject=None, model=None, layer=None):
        if modality == "brain":
            assignment = self.spatial.assignments(subject)
        elif modality == "model":
            width = self.states(model, layer, train[0]).shape[1]
            groups = min(self.config["model_coordinate_groups"], width)
            order = np.random.default_rng(self.config["coordinate_partition_seed"]).permutation(width)
            assignment = np.empty(width, dtype=np.int32)
            assignment[order] = np.arange(width) % groups
        else:
            raise ValueError("Unknown observed representation modality.")
        prep = self.compute_config['preparation']
        def statistics():
            for story in train:
                deadline.check()
                identity = {'inputs': self.cache_identity(model), 'modality': modality, 'subject': subject,
                    'model': model, 'layer': layer, 'story': story, 'partition': object_hash(assignment.tolist()),
                    'preparation': prep}
                def build():
                    values = (self.brain(subject, story).reshape(-1, len(assignment)) if modality == 'brain' else
                              self.states(model, layer, story)[:len(self.semantics.records(story, 'sources'))])
                    return story_statistics(values, assignment, device=prep['device'], batch_sites=prep['batch_sites'])
                yield self.host_cache.get(('pca-story-statistics', object_hash(identity)),
                    lambda: self.cache.arrays('pca-story-statistics', identity, build))
        return fit_statistics(SiteProjector(assignment, self.config['components_per_site']),
            statistics(), train, device=prep['device'], eigensolver=prep['eigensolver'])

    def windows(self, modality, story, projector, *, subject=None, model=None, layer=None):
        """Source -> repetitions x sites x ordered samples x local components."""
        if not hasattr(projector, 'cache_identity'):
            return self._windows(modality, story, projector, subject=subject, model=model, layer=layer)
        identity = {'projector': projector.cache_identity, 'story': story,
                    'modality': modality, 'subject': subject, 'model': model, 'layer': layer}
        def build(path):
            from .io import save_json
            windows = self._windows(modality, story, projector, subject=subject, model=model, layer=layer)
            if not windows:
                raise ValueError('No actual eligible observations in ' + story)
            save_json(path / 'sources.json', list(windows))
            np.save(path / 'windows.npy', np.stack(list(windows.values())), allow_pickle=False)
        def load():
            path = self.cache.entry('windows', identity, build)
            keys = read_json(path / 'sources.json')
            values = np.load(path / 'windows.npy', allow_pickle=False)
            return dict(zip(keys, values, strict=True))
        return self.host_cache.get(('windows', object_hash(identity)), load)

    def _windows(self, modality, story, projector, *, subject=None, model=None, layer=None):
        sources = self.semantics.records(story, "sources")
        result = {}
        if modality == "brain":
            response = self.brain(subject, story)
            prep = self.compute_config['preparation']
            transformed = np.stack([projector.transform(r, device=prep['device'], batch_sites=prep['batch_sites']) for r in response])
            for source in sources:
                if source["decoder_timing_eligible"]:
                    rows = source["decoder_trimmed_response_rows"]
                    result[source["id"]] = transformed[:, rows].transpose(0, 2, 1, 3)
        elif modality == "model":
            prep = self.compute_config['preparation']
            transformed = projector.transform(self.states(model, layer, story), device=prep['device'], batch_sites=prep['batch_sites'])
            for source in sources:
                if source["decoder_timing_eligible"]:
                    result[source["id"]] = transformed[source["model_unit_row"]][None, :, None, :]
        else:
            raise ValueError("Unknown modality.")
        return result

    def examples(self, story, *, reviewed_only=False, families=None):
        queries = {q["id"]: q for q in self.semantics.records(story, "queries")}
        for example in self.semantics.decoder_examples(story, reviewed_only=reviewed_only, families=families):
            query = queries[example["query_id"]]
            # Validated catalog objects are shared read-only by this analysis.
            # Do not retain a separate large prefix catalog for every question.
            example["inputs"]["candidates"] = self.semantics.candidate_sets[query["input"]["candidate_set"]]
            example['inputs'] = freeze_descriptor(example['inputs'])
            yield {**example, "story_id": story, "family": query["family"]}

    def story(self, story):
        entry = next(e for e in self.index["stories"] if e["id"] == story)
        return checked_story(self.root / self.semantics.identity["config"]["corpus"], entry)
