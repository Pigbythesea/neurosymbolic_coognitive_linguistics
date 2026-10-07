"""Story-nested, group-regularized encoding of actual native-voxel responses."""
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg

from .analysis_runs import column_correlation, partition, run_directory, write_report
from .compute import FP64, Timings, file_hash
from .runtime import analysis_lock as exclusive_run, deadline, process_lock
from .encoding_support import EncodingSupport, assert_matched_support, comparison_spec, validate_condition
from .io import object_hash, read_json, save_json
from .semantic_features import SemanticVectorizer
from .temporal import fir_design


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
    def __init__(self, alpha, weights, *, device='cpu', solver='auto'):
        if alpha <= 0 or not weights or any(v <= 0 for v in weights.values()):
            raise ValueError("Positive group penalties required.")
        self.alpha, self.weights = float(alpha), dict(weights)
        self.math = FP64(device)
        if solver not in {'auto', 'primal', 'dual'}:
            raise ValueError('Unknown ridge solver.')
        self.solver = solver

    def kernel(self, first, second):
        if set(first) != set(self.weights) or set(second) != set(self.weights):
            raise ValueError("Kernel weights and groups disagree.")
        return sum(self.weights[g] * self.math.product(first[g], second[g].T) for g in self.weights)

    def fit_design(self, groups):
        self.scaler = GroupScaler().fit(groups)
        self.training = self.scaler.transform(groups)
        n = len(next(iter(self.training.values())))
        p = sum(v.shape[1] for v in self.training.values())
        self.mode = ('primal' if p < n else 'dual') if self.solver == 'auto' else self.solver
        if self.mode == 'primal':
            self.design = self.math.concatenate([np.sqrt(self.weights[g]) * self.training[g] for g in self.weights])
            kernel = self.design.T @ self.design
        else:
            kernel = self.kernel(self.training, self.training)
        self.factor = self.math.factor(kernel, self.alpha)
        return self

    def prediction_operator(self, groups):
        return self.math.numpy(sum(self.group_operators(groups).values()))

    def group_operators(self, groups):
        """Map centered training responses to each heldout group contribution.

        Operators are FP64 and independent of target voxels. They are a compact
        exact readout on the heldout design, with targets referenced once from
        the immutable release rather than duplicated in every fitted artifact.
        """
        transformed = self.scaler.transform(groups)
        if self.mode == 'dual':
            return {g: self.math.solve(self.factor, (self.weights[g] * self.math.product(transformed[g], self.training[g].T)).T).T
                    for g in self.weights}
        basis = self.math.solve(self.factor, self.design.T)
        result, offset = {}, 0
        for g in self.weights:
            width = self.training[g].shape[1]
            result[g] = self.math.product(np.sqrt(self.weights[g]) * transformed[g], basis[offset:offset + width])
            offset += width
        return result

    def coefficients(self, y):
        mean = y.mean(0)
        centered = self.math.array(y - mean)
        if self.mode == 'primal':
            beta = self.math.solve(self.factor, self.design.T @ centered)
            dual = (centered - self.design @ beta) / self.alpha
        else:
            dual = self.math.solve(self.factor, centered)
        return dual, self.math.array(mean)


class EncodingFeatures:
    def __init__(self, data, train, groups, *, support, model=None, layer=None):
        data.validate_partition(train)
        validate_condition(support.spec, groups, model, layer)
        if support.data is not data:
            raise ValueError("Encoding support belongs to another dataset.")
        self.support = support
        self.data, self.groups, self.train = data, list(groups), list(train)
        self.model, self.layer = model, layer
        self.vectorizers = {g: SemanticVectorizer.fit(data.semantics, train, groups=[g])
                            for g in groups if g not in {"presentation", "legacy", "model"}}
        self.coverage = {}

    def story(self, story):
        data = self.data
        timing = read_json(data.semantics.build / "stories" / story / "timing.json")
        rows, delays = timing["response_raw_indices"], timing["fir_delays_trs"]
        # All conditions and every inner/outer fold use the declared contrast support.
        valid = self.support.story(story)
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
            if values.shape[0] != len(valid) or mask.shape != valid.shape or np.any(valid & ~mask):
                raise ValueError("A condition would silently narrow the shared comparison support: " + group)
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
                  "model": self.model, "layer": self.layer, "coverage": self.coverage,
                  "comparison_support": self.support.spec})


def response_matrix(data, subject, stories, masks, section):
    """Training stories have one repeat; only explicit prediction metrics average repeats."""
    arrays = []
    for story in stories:
        y = data.response(subject, story, section)
        if y.shape[0] != 1:
            raise ValueError("Do not pool heldout repeated recordings into a training matrix.")
        arrays.append(y[0, masks[story]])
    values = np.concatenate(arrays).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured training responses.")
    return values


def validation_row_scale(masks, stories):
    counts = [int(np.count_nonzero(masks[s])) for s in stories]
    if not counts or min(counts) < 1:
        raise ValueError('Every validation story needs observed response rows.')
    total = sum(counts)
    return np.concatenate([np.full(n, np.sqrt(total / (len(counts) * n))) for n in counts])


def target_moments(data, subject, train, validation, masks, batch, *, device='cpu', return_device=False):
    """Exact all-voxel normalized-MSE tuning through sufficient statistics.

    Avoid materializing one prediction for every hyperparameter x voxel.
    Means/scales are fitted on training responses only, and constant targets excluded.
    """
    n, m = sum(masks[s].sum() for s in train), sum(masks[s].sum() for s in validation)
    math = FP64(device)
    row_scale = math.array(validation_row_scale(masks, validation))[:, None]
    gram, cross, total, count = math.array(np.zeros((n, n))), math.array(np.zeros((n, m))), 0.0, 0
    voxels = data.reader.contract["subjects"][subject][train[0]]["voxels"]
    for start in range(0, voxels, batch):
        section = slice(start, min(start + batch, voxels))
        y = response_matrix(data, subject, train, masks, section)
        z = response_matrix(data, subject, validation, masks, section)
        if math.gpu:
            dy, dz = math.array(y), math.array(z)
            mean, scale = dy.mean(0), dy.std(0, correction=0)
            use = scale > 1e-8
            dy, dz = (dy[:, use] - mean[use]) / scale[use], (dz[:, use] - mean[use]) / scale[use]
            dz = dz * row_scale
            total = total + (dz * dz).sum()
            count += dy.shape[1]
        else:
            mean, scale = y.mean(0), y.std(0)
            use = scale > 1e-8
            dy, dz = (y[:, use] - mean[use]) / scale[use], (z[:, use] - mean[use]) / scale[use]
            dz = dz * row_scale
            total += float(np.sum(dz * dz))
            count += int(use.sum())
        gram += dy @ dy.T
        cross += dy @ dz.T
    if not count:
        raise ValueError("No variable training voxels.")
    if return_device:
        return gram / count, cross / count, math.array(total / count)
    return math.numpy(gram / count), math.numpy(cross / count), float(math.numpy(math.array(total / count)))


def fold_kernels(data, options, fold, support, math):
    """Bounded resident designs/products, shared across a panel of participants."""
    masks = {s: support.story(s) for s in fold['train'] + fold['validation']}
    rows = {s: np.flatnonzero(m).tolist() for s, m in masks.items()}
    designs, identities = {}, {}
    backend = {'device': 'cuda' if math.gpu else 'cpu', 'dtype': 'float64'}
    if math.gpu:
        backend['torch'] = math.torch.__version__
    for group in options['groups']:
        model = options.get('model') if group == 'model' else None
        identity = {'inputs': data.cache_identity(model), 'code': file_hash(Path(__file__)),
                    'fold': fold, 'rows': rows, 'group': group, 'backend': backend,
                    'model': model, 'layer': options.get('layer') if model else None,
                    'model_kind': data.config['encoding']['model_kind'] if model else None}
        def build(group=group):
            factory = EncodingFeatures(data, fold['train'], [group], support=support,
                                       model=options.get('model') if group == 'model' else None,
                                       layer=options.get('layer') if group == 'model' else None)
            x, _ = factory.assemble(fold['train'])
            z, _ = factory.assemble(fold['validation'])
            scaler = GroupScaler().fit(x)
            tx, vz = scaler.transform(x)[group], scaler.transform(z)[group]
            vz = vz * validation_row_scale(masks, fold['validation'])[:, None]
            return {'train': tx, 'validation': vz}
        identities[group] = identity
        designs[group] = data.host_cache.get(('encoding-design', object_hash(identity)), build)
    n = len(next(iter(designs.values()))['train'])
    primal = sum(v['train'].shape[1] for v in designs.values()) < n
    kernels = {}
    if not primal:
        for group, design in designs.items():
            def products(design=design):
                x = math.array(design['train'])
                return {'train': x @ x.T, 'validation': math.array(design['validation']) @ x.T}
            kernels[group] = data.resident(math.device).get(('encoding-kernel', object_hash(identities[group])), products)
    identity = {'inputs': data.cache_identity(), 'code': file_hash(Path(__file__)),
                'subject': options['subject'], 'fold': fold, 'rows': rows, 'backend': backend,
                'target_batch': data.config['encoding']['target_batch']}
    def moments_on_device():
        def build(path):
            gram, cross, total = target_moments(data, options['subject'], fold['train'], fold['validation'], masks,
                                               data.config['encoding']['target_batch'], device=math.device)
            np.savez(path / 'moments.npz', gram=gram, cross=cross, total=total)
            save_json(path / 'producer.json', backend_identity(math))
        path = data.cache.entry('encoding-moments', identity, build)
        with np.load(path / 'moments.npz', allow_pickle=False) as file:
            return tuple(math.array(file[name]) for name in ('gram', 'cross', 'total'))
    moments = data.resident(math.device).get(('encoding-moments', object_hash(identity)), moments_on_device)
    return {'kernels': kernels, 'designs': designs, 'identities': identities, 'primal': primal}, moments


def spectrum(data, products, weights, math):
    key = object_hash({'groups': products['identities'], 'weights': weights})
    def build():
        if products['primal']:
            x = math.concatenate([np.sqrt(weights[g]) * d['train'] for g, d in products['designs'].items()])
            z = math.concatenate([np.sqrt(weights[g]) * d['validation'] for g, d in products['designs'].items()])
            values, vectors = math.eigh(x.T @ x)
            return {'eigenvalues': values, 'cross_vectors': z @ vectors, 'basis': vectors.T @ x.T}
        kernels = products['kernels']
        kernel = sum(weights[g] * kernels[g]['train'] for g in weights)
        cross = sum(weights[g] * kernels[g]['validation'] for g in weights)
        values, vectors = math.eigh(kernel)
        return {'eigenvalues': values, 'cross_vectors': cross @ vectors, 'basis': vectors.T}
    return data.resident(math.device).get(('encoding-spectrum', key), build)


def spectral_losses(spectral, moments, alphas, math, *, primal):
    basis, cross = spectral['basis'], spectral['cross_vectors']
    gram, paired, total = moments
    if primal:
        gram, paired = basis @ gram @ basis.T, basis @ paired
    losses = []
    for alpha in alphas:
        scaled = cross / (math.nonnegative(spectral['eigenvalues']) + alpha)
        operator = scaled if primal else scaled @ basis
        losses.append((((operator @ gram) * operator).sum() - 2 * (operator * paired.T).sum() + total) / len(operator))
    return math.numpy(math.torch.stack(losses)).tolist() if math.gpu else [float(v) for v in losses]


def select_encoding(data, options, split, support):
    cfg, groups = data.config["encoding"], options["groups"]
    math = FP64(options.get('device', 'cpu'))
    from .protocol import encoding_mixtures
    mixtures = encoding_mixtures(cfg, groups)
    grid = [{"alpha": float(a), "weights": dict(zip(groups, w.tolist(), strict=True))}
            for w in mixtures for a in cfg["alphas"]]
    scores = [[] for _ in grid]
    for inner_index, fold in enumerate(split["inner"]):
        deadline.check()
        identity = {'inputs': data.cache_identity(options.get('model')), 'code': file_hash(Path(__file__)),
                    'subject': options['subject'], 'fold': fold, 'groups': groups, 'grid': grid,
                    'model': options.get('model'), 'layer': options.get('layer'),
                    'model_kind': cfg['model_kind'], 'target_batch': cfg['target_batch'],
                    'rows': {s: np.flatnonzero(support.story(s)).tolist() for s in fold['train'] + fold['validation']},
                    'backend': backend_identity(math)}
        def calculate():
            products, moments = fold_kernels(data, options, fold, support, math)
            values = []
            for weights in mixtures:
                weight_map = dict(zip(groups, weights.tolist(), strict=True))
                values.extend(spectral_losses(spectrum(data, products, weight_map, math), moments,
                                              cfg['alphas'], math, primal=products['primal']))
            return {'scores': np.asarray(values)}
        values = data.cache.arrays('encoding-scores', identity, calculate)['scores']
        if values.shape != (len(grid),) or not np.isfinite(values).all():
            raise ValueError('Invalid encoding selection scores.')
        for slot, value in zip(scores, values, strict=True):
            slot.append(float(value))
        print(f"ENCODING SELECT inner={inner_index + 1}/{len(split['inner'])}", flush=True)
    means = [float(np.average(s, weights=[len(f['validation']) for f in split['inner']])) for s in scores]
    best = int(np.argmin(means))
    return grid[best], {"criterion": "story-macro MSE normalized by training-only voxel variance; all variable native voxels",
                        "grid": grid, "inner_scores": scores, "mean_loss": means, "selected": best}


def encoding_context(data, options):
    from .protocol import selection_partition
    split = selection_partition(data, options, decoder=False)
    data.validate_partition(split["train"], test=split["test"])
    spec = comparison_spec(data.config, options)
    options = {**options, "comparison_support": spec}
    support = EncodingSupport(data, spec)
    report = support.report(split["train"] + split["test"])
    if any(s["retained"] == 0 for s in report["stories"].values()):
        raise ValueError("A comparison has no jointly observed rows.")
    directory, identity = run_directory(data, "encoding", options)
    return options, split, support, report, directory, identity


def prepare_encoding_panel(data, options_list):
    """Visit inner folds before participants, keeping their shared spectra hot."""
    contexts = [encoding_context(data, o) for o in options_list]
    pending = [c for c in contexts if not (c[4] / 'complete.json').exists() and not (c[4] / 'selection.json').exists()]
    maximum = max((len(c[1]['inner']) for c in pending), default=0)
    for index in range(maximum):
        for options, split, support, report, directory, identity in pending:
            if index < len(split['inner']):
                select_encoding(data, options, {'inner': [split['inner'][index]]}, support)


def encoding_file(directory):
    directory = Path(directory).resolve()
    if (directory / 'artifact.json').exists():
        reference = read_json(directory / 'artifact.json')
        target = (directory / reference['path']).resolve()
        if not target.is_relative_to(directory.parent.parent / 'encoding-objects'):
            raise ValueError('Encoding artifact reference escaped analysis storage.')
        shared = read_json(target / 'complete.json')
        logical = read_json(directory / 'complete.json')
        digest = object_hash(shared['identity'])
        if digest != reference['numeric_identity_hash'] or digest != logical['numeric_identity_hash'] or target.name != digest:
            raise ValueError('Logical encoding run and shared numerical artifact disagree.')
        if (shared['identity']['subject'] != logical['identity']['options']['subject'] or
                shared['identity']['train'] != logical['partition']['train'] or
                shared['identity']['test'] != logical['partition']['test'] or
                shared['identity']['selected'] != logical['selected']):
            raise ValueError('Shared encoding artifact belongs to another selected fit.')
        with h5py.File(target / 'encoding.h5', 'r') as file:
            if (not file.attrs['complete'] or file.attrs.get('numeric_identity_hash') != digest or
                    file.attrs.get('observation_rows_hash') != object_hash(shared['identity']['rows'])):
                raise ValueError('Shared encoding HDF header differs from its receipt.')
        return target / 'encoding.h5'
    return directory / 'encoding.h5'


def prediction_batches(directory, data, story, *, component=None, device='cpu'):
    """Recover heldout arrays without refitting or materializing all voxels."""
    receipt = read_json(Path(directory) / 'complete.json')
    if (receipt['identity']['semantic_build_hash'] != data.semantics.build_hash or
            receipt['identity']['data_contract_hash'] != object_hash(data.reader.contract) or
            story not in receipt['partition']['test']):
        raise ValueError('Prediction reconstruction requires the pinned dataset and heldout story.')
    subject = receipt['identity']['options']['subject']
    train = receipt['partition']['train']
    math = FP64(device)
    with h5py.File(encoding_file(directory), 'r') as file:
        if not file.attrs['complete']:
            raise ValueError('Incomplete encoding output.')
        voxels, batch = len(file['target_mean']), data.config['encoding']['target_batch']
        if file.attrs.get('format_version', 1) == 1:
            matrix = file[story]['contributions/' + component] if component else file[story]['prediction']
            for start in range(0, voxels, batch):
                section = slice(start, min(start + batch, voxels))
                yield section, matrix[:, section]
            return
        operators = file[story]['operators']
        operator = (math.array(operators[component][()]) if component else
                    sum(math.array(operators[g][()]) for g in operators.attrs['group_order'].split(',')))
        masks = {}
        for s in train:
            masks[s] = np.zeros(data.reader.contract['subjects'][subject][s]['timepoints'] - 20, dtype=bool)
            masks[s][file['training_response_rows/' + s][()]] = True
        for start in range(0, voxels, batch):
            section = slice(start, min(start + batch, voxels))
            y = response_matrix(data, subject, train, masks, section)
            mean = file['target_mean'][section]
            values = operator @ math.array(y - mean)
            if component is None:
                values = values + math.array(mean)
            # Matches the existing downstream float32 prediction interface.
            yield section, math.numpy(values).astype(np.float32)


def response_correlations(math, first, second):
    if not math.gpu:
        return column_correlation(first, second)
    a, b = first - first.mean(0), second - second.mean(0)
    denominator = math.torch.linalg.vector_norm(a, dim=0) * math.torch.linalg.vector_norm(b, dim=0)
    return math.torch.where(denominator > 1e-12, (a * b).sum(0) / denominator,
                            math.torch.full_like(denominator, float('nan')))


def backend_identity(math):
    return {'device': math.device, 'dtype': 'float64',
            **({'torch': str(math.torch.__version__), 'gpu': math.torch.cuda.get_device_name(math.device)} if math.gpu else {})}


def run_encoding(data, options):
    from .storage import ledger_lock
    import os
    timings = Timings(options.get('device', 'cpu'))
    options, split, support, support_report, directory, identity = encoding_context(data, options)
    with exclusive_run(directory / 'RUNNING.lock'):
        if (directory / 'complete.json').exists():
            return directory
        write_report(directory / 'support.json', support_report)
        if (directory / 'selection.json').exists():
            selection = read_json(directory / 'selection.json')
            if selection['support_hash'] != support_report['content_hash']:
                raise ValueError('Resumed selection uses another support.')
            selected = selection['grid'][selection['selected']]
        else:
            with timings.phase('nested_selection'):
                selected, selection = select_encoding(data, options, split, support)
        selection['support_hash'] = support_report['content_hash']
        write_report(directory / 'selection.json', selection)
        rows = {s: support_report['stories'][s]['response_row_indices'] for s in split['train'] + split['test']}
        # Provenance-only names and seed labels cannot force copies of an
        # identical selected numerical fit. All source/row/feature facts remain.
        numeric_identity = {'format_version': 2, 'source': data.cache_identity(options.get('model')),
            'code': file_hash(Path(__file__)), 'subject': options['subject'], 'train': split['train'],
            'test': split['test'], 'rows': rows, 'groups': options['groups'], 'selected': selected,
            'model': options.get('model'), 'layer': options.get('layer'),
            'model_kind': data.config['encoding']['model_kind'], 'backend': backend_identity(FP64(options.get('device', 'cpu')))}
        canonical = data.root / data.config['output'] / 'encoding-objects' / object_hash(numeric_identity)
        canonical.mkdir(parents=True, exist_ok=True)
        with process_lock(canonical / '.writer.oslock'):
            if (canonical / 'complete.json').exists():
                saved = read_json(canonical / 'complete.json')
                if saved['identity'] != numeric_identity:
                    raise ValueError('Shared encoding artifact identity changed.')
                summaries = saved['stories']
            else:
                with timings.phase('refit_design_and_operators'):
                    factory = EncodingFeatures(data, split['train'], options['groups'], support=support,
                                               model=options.get('model'), layer=options.get('layer'))
                    x, masks = factory.assemble(split['train'])
                    test = {s: factory.story(s) for s in split['test']}
                    ridge = GroupRidge(**selected, device=options.get('device', 'cpu')).fit_design(x)
                    math = ridge.math
                    factory.save(canonical / 'features')
                    ridge.scaler.save(canonical / 'scaler.npz')
                    operators = {s: ridge.group_operators({g: v[mask] for g, v in groups.items()})
                                 for s, (groups, mask) in test.items()}
                    totals = {s: sum(parts.values()) for s, parts in operators.items()}
                    n = len(next(iter(ridge.training.values())))
                    m = sum(len(values) for values in totals.values())
                    p = sum(v.shape[1] for v in ridge.training.values())
                    # Save the common output operators, but apply a lower-cost
                    # equivalent association when p is small enough. Include
                    # the triangular-solve work in this FLOP comparison.
                    direct_primal = ridge.mode == 'primal' and p * (n + m + p) < n * m
                    heldout_designs = ({s: math.concatenate([
                        np.sqrt(selected['weights'][g]) * values[g] for g in options['groups']])
                        for s, (groups, mask) in test.items()
                        for values in [ridge.scaler.transform({g: v[mask] for g, v in groups.items()})]}
                        if direct_primal else {})
                voxels = data.reader.contract['subjects'][options['subject']][split['train'][0]]['voxels']
                from .storage import require_free_space, size_bytes
                require_free_space(canonical, size_bytes(operators) + voxels * (8 + 32 * len(test)),
                                   data.compute_config['storage']['minimum_free_gib'] * 2**30)
                batch, summaries = data.config['encoding']['target_batch'], {}
                with h5py.File(canonical / 'encoding.h5', 'a') as file:
                    if file.attrs.get('numeric_identity_hash', object_hash(numeric_identity)) != object_hash(numeric_identity):
                        raise ValueError('Interrupted encoding output belongs to another fit.')
                    initialized = bool(file.attrs.get('operators_ready', False))
                    file.attrs.update(complete=False, format_version=2, solver=ridge.mode,
                        observation_rows_hash=object_hash(rows), subject=options['subject'],
                        numeric_identity_hash=object_hash(numeric_identity),
                        metric_product='primal coefficients' if direct_primal else 'response operators',
                        representation='FP64 group operators on centered pinned training responses')
                    for story, mask in masks.items():
                        name = 'training_response_rows/' + story
                        if name not in file:
                            file.create_dataset(name, data=np.flatnonzero(mask))
                        elif not initialized:
                            file[name][...] = np.flatnonzero(mask)
                    file.require_dataset('target_mean', (voxels,), dtype='f8')
                    for story, (_, mask) in test.items():
                        group = file.require_group(story)
                        if 'response_rows' not in group:
                            group.create_dataset('response_rows', data=np.flatnonzero(mask))
                        elif not initialized:
                            group['response_rows'][...] = np.flatnonzero(mask)
                        opgroup = group.require_group('operators')
                        opgroup.attrs['group_order'] = ','.join(options['groups'])
                        for name, values in operators[story].items():
                            shape = values.shape
                            if name not in opgroup:
                                opgroup.create_dataset(name, data=math.numpy(values), compression='lzf',
                                                       chunks=(min(64, shape[0]), min(512, shape[1])))
                            elif not initialized:
                                opgroup[name][...] = math.numpy(values)
                        repeats = data.reader.contract['subjects'][options['subject']][story]['repeats']
                        group.require_dataset('repeat_correlation', (repeats, voxels), dtype='f4')
                        group.require_dataset('repeat_mse', (repeats, voxels), dtype='f4')
                        group.require_dataset('mean_response_correlation', (voxels,), dtype='f4')
                        if repeats == 2:
                            group.require_dataset('repeat_reliability', (voxels,), dtype='f4')
                    file.attrs['operators_ready'] = True
                    file.flush()
                    with timings.phase('voxel_predictions_and_metrics'):
                        cursor = int(file.attrs.get('completed_voxels', 0))
                        if not 0 <= cursor <= voxels or (cursor != voxels and cursor % batch):
                            raise ValueError('Invalid encoding continuation cursor.')
                        for start in range(cursor, voxels, batch):
                            section = slice(start, min(start + batch, voxels))
                            y = response_matrix(data, options['subject'], split['train'], masks, section)
                            resident = math.array(y)
                            dmean = resident.mean(0)
                            file['target_mean'][section] = math.numpy(dmean)
                            centered = resident - dmean
                            beta = math.solve(ridge.factor, ridge.design.T @ centered) if direct_primal else None
                            for story, (_, mask) in test.items():
                                prediction = (heldout_designs[story] @ beta if direct_primal else totals[story] @ centered) + dmean
                                actual = data.response(options['subject'], story, section)[:, mask]
                                target = math.array(actual)
                                group = file[story]
                                for repeat in range(len(actual)):
                                    group['repeat_correlation'][repeat, section] = math.numpy(response_correlations(math, prediction, target[repeat]))
                                    group['repeat_mse'][repeat, section] = math.numpy(((prediction - target[repeat]) ** 2).mean(0))
                                # Preserve the original float32 repeat-mean convention before FP64 correlation.
                                group['mean_response_correlation'][section] = math.numpy(response_correlations(math, prediction, math.array(actual.mean(0))))
                                if len(actual) == 2:
                                    group['repeat_reliability'][section] = math.numpy(response_correlations(math, target[0], target[1]))
                            print(f'ENCODING FIT voxels={section.stop}/{voxels}', flush=True)
                            file.attrs['completed_voxels'] = section.stop
                            file.flush()
                            deadline.advance()
                            deadline.check()
                    for story in test:
                        summaries[story] = {'mean_voxel_r': float(np.nanmean(file[story]['mean_response_correlation'][()])),
                                            'valid_rows': int(test[story][1].sum()), 'voxels': voxels}
                    file.attrs['complete'] = True
                write_report(canonical / 'complete.json', {'identity': numeric_identity, 'stories': summaries,
                    'solver': ridge.mode, 'prediction_reconstruction': 'No hyperparameter selection or fit required; apply saved operators to pinned centered training responses.'})
        write_report(directory / 'artifact.json', {'path': Path(os.path.relpath(canonical, directory)).as_posix(),
                                                   'numeric_identity_hash': object_hash(numeric_identity)})
        write_report(directory / 'complete.json', {'identity': identity, 'partition': split, 'selected': selected,
            'stories': summaries, 'support_hash': support_report['content_hash'], 'numeric_identity_hash': object_hash(numeric_identity),
            'interpretation': 'Heldout prediction of measured native voxels. Group contributions are model-dependent, not causal effects.'})
        write_report(directory / 'runtime.json', {**timings.report(), 'cache_hits': data.cache.hits,
            'cache_misses': data.cache.misses, 'encoding_dtype': 'float64',
            'host_reuse_bytes': data.host_cache.bytes, 'device_reuse_bytes': data.resident(options.get('device', 'cpu')).bytes})
    return directory


def compare_encoding(first, second):
    """Paired descriptive effects; enforce matching support before subtracting scores."""
    from .io import object_hash
    first, second = Path(first), Path(second)
    a, b = read_json(first / "complete.json"), read_json(second / "complete.json")
    if any(r["identity"]["kind"] != "encoding" for r in (a, b)):
        raise ValueError("Paired encoding comparison requires two completed encoding runs.")
    for key in ("semantic_build_hash", "data_contract_hash", "config", "code", "packages"):
        if a["identity"].get(key) != b["identity"].get(key):
            raise ValueError("Paired encoding run contracts differ: " + key)
    if a["partition"] != b["partition"] or a["identity"]["options"]["subject"] != b["identity"]["options"]["subject"]:
        raise ValueError("Paired encoding effects require the same participant and nested story partition.")
    supports = [read_json(p / "support.json") for p in (first, second)]
    assert_matched_support(*supports)
    if any(r.get("support_hash") != s["content_hash"] for r, s in zip((a, b), supports, strict=True)):
        raise ValueError("Run and comparison support identities differ.")
    stories = {}
    with h5py.File(encoding_file(first), "r") as af, h5py.File(encoding_file(second), "r") as bf:
        for file in (af, bf):
            if not file.attrs["complete"] or (file.attrs.get('format_version', 1) == 1 and file.attrs.get("support_hash") != supports[0]["content_hash"]):
                raise ValueError("Encoding output has missing or mismatched support.")
            for story in a["partition"]["train"]:
                if file["training_response_rows/" + story][()].tolist() != supports[0]["stories"][story]["response_row_indices"]:
                    raise ValueError("Encoding training rows differ from comparison support.")
        for story in a["partition"]["test"]:
            for file in (af, bf):
                if file[story]["response_rows"][()].tolist() != supports[0]["stories"][story]["response_row_indices"]:
                    raise ValueError("Encoding test rows differ from comparison support.")
            left, right = af[story]["mean_response_correlation"][()], bf[story]["mean_response_correlation"][()]
            if left.shape != right.shape:
                raise ValueError("Native voxel coordinates differ.")
            valid = np.isfinite(left) & np.isfinite(right)
            difference = right[valid] - left[valid]
            stories[story] = {"finite_paired_voxels": int(valid.sum()), "total_voxels": len(valid),
                              "mean_voxel_delta_r": float(difference.mean()) if len(difference) else None,
                              "response_rows": supports[0]["stories"][story]["retained"]}
    return {"first": object_hash(a["identity"]), "second": object_hash(b["identity"]),
            "direction": "second minus first", "subject": a["identity"]["options"]["subject"],
            "support_hash": supports[0]["content_hash"], "stories": stories,
            "interpretation": "Paired descriptive prediction differences. Voxels/timepoints are not independent replicates; no population p-value is inferred here."}
