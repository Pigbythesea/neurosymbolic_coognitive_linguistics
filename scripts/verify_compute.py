"""Numerical equivalence on released text features and accepted real queries.

These are numerical checks, not fMRI/model experiments or replacement data.
The CUDA invocation additionally checks the actual installed GPU backend.
"""
import argparse
import copy
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('NEUROSYM_ROOT', CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import h5py
from scipy.stats import spearmanr

from neurosym.analysis_data import AnalysisData, SiteProjector
from neurosym.analysis_runs import write_report, decoder_metrics
from neurosym.compute import FoldCache, FP64, atomic_torch_save, file_hash
from neurosym.decoder_fit import evaluate, train_decoder, fit_identity, examples_and_windows
from neurosym.decoders import SemanticDecoder, acceptable_loss, query_vocabulary
from neurosym.decoder_batch import choice_targets, choice_losses
from neurosym.encoding import GroupRidge, spectral_losses, spectrum, response_correlations
from neurosym.storage import ResidentCache, DiskBudget, size_bytes
from neurosym.execution import code_identity, execution_manifest
from neurosym.geometry import cosine_rdm, rdm_comparison, map_label_permutation
from neurosym.io import object_hash
from neurosym.reviewed_queries import TASKS
from neurosym.pca import story_statistics, fit_statistics
from neurosym.runtime import deadline, YieldRequested, process_lock


def close(first, second, *, atol=1e-8, rtol=1e-8):
    if not np.allclose(first, second, atol=atol, rtol=rtol):
        raise AssertionError('Numerical equivalence failed; maximum error=' + str(np.max(np.abs(first - second))))
    return float(np.max(np.abs(first - second))) if np.size(first) else 0.


def real_inputs(data, *, include_full_source=False):
    arrays = [data.reader.features(s, ['english1000', 'letters', 'numwords', 'numletters'], trim=False)
              for s in ['story_01', 'story_02']]
    values = arrays[0]['english1000']
    projector = SiteProjector(np.arange(values.shape[1]) % 8, 3).fit(values, ['story_01'])
    selected, sources = {}, {}
    expected = TASKS - {'compose'} | {'compose/argument_event', 'compose/scope_parent', 'compose/event_relation'}
    for story in data.semantics.splits['development']:
        sources.update({s['id']: s for s in data.semantics.records(story, 'sources')})
        for example in data.examples(story):
            source = sources[example['source_id']]
            if not source['decoder_timing_eligible'] or source['raw_feature_bin'] < 4:
                continue
            ast = example['inputs']['ast']
            key = ast['task']
            if key == 'compose':
                key += '/' + ast['steps'][0]['op']
            selected.setdefault(key, example)
        if expected <= set(selected):
            break
    if not expected <= set(selected):
        raise ValueError('Accepted real operator coverage is incomplete: ' + str(expected - set(selected)))
    examples, windows, transformed = list(selected.values()), {}, {}
    # Include an entire actual source as well as operator coverage: source
    # batching must handle shared descriptors and unequal candidate counts.
    if include_full_source:
        first_story = examples[0]['story_id']
        source_queries = {}
        for example in data.examples(first_story):
            source = sources[example['source_id']]
            if source['decoder_timing_eligible'] and source['raw_feature_bin'] >= 4:
                source_queries.setdefault(example['source_id'], []).append(example)
        full_source = max(source_queries.values(), key=lambda group: sum(len(e['inputs']['candidates']) for e in group))
        known = {e['query_id'] for e in examples}
        examples.extend(e for e in full_source if e['query_id'] not in known)
    for e in examples:
        story = e['story_id']
        if story not in transformed:
            transformed[story] = projector.transform(data.reader.features(story, ['english1000'], trim=False)['english1000'])
        slot = sources[e['source_id']]['raw_feature_bin']
        windows[e['source_id']] = transformed[story][slot - 4:slot].transpose(1, 0, 2)[None]
    return arrays, projector, examples, windows


def verify_source_batches(examples, windows, projector, *, device):
    from collections import defaultdict
    groups = defaultdict(list)
    for example in examples:
        groups[example['source_id']].append(example)
    vocabulary = query_vocabulary(examples)
    errors = {}
    for family in ('prior', 'linear', 'mlp', 'structured'):
        torch.manual_seed(11)
        reference = SemanticDecoder(family, (8, 4, 3), vocabulary, hidden=64, site_mask=projector.site_mask)
        reference.source_batched = False
        candidate = copy.deepcopy(reference).to(device)
        candidate.source_batched = True
        maximum, gradient_error, loss_error = 0., 0., 0.
        for source, queries in groups.items():
            x = torch.from_numpy(windows[source][0])
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            reference.descriptors.begin_source()
            observed = reference.encode_observation(None if family == 'prior' else x)
            original = [reference.answer(observed, e['inputs'])[0] for e in queries]
            projected = candidate.encode_observation(None if family == 'prior' else x.to(device))
            batched = candidate.answer_many(projected, [e['inputs'] for e in queries])
            for a, b in zip(original, batched, strict=True):
                maximum = max(maximum, close(a.detach().numpy(), b[:len(a)].detach().cpu().numpy(), atol=3e-5, rtol=3e-5))
                if not torch.isneginf(b[len(a):]).all():
                    raise AssertionError('Padded alternatives have nonzero probability.')
            a_loss = sum(e['weight'] * acceptable_loss(a, e['acceptable_indices']) for a, e in zip(original, queries, strict=True))
            targets = choice_targets(queries, device)
            b_loss = (targets['weights'] * choice_losses(batched, targets)).sum()
            loss_error = max(loss_error, close(a_loss.detach().numpy(), b_loss.detach().cpu().numpy(), atol=3e-5, rtol=3e-5))
            a_loss.backward()
            b_loss.backward()
            for (name, a), (other, b) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                if name != other or (a.grad is None) != (b.grad is None):
                    raise AssertionError('Source batching changed gradient support: ' + name)
                if a.grad is not None:
                    gradient_error = max(gradient_error, close(a.grad.numpy(), b.grad.cpu().numpy(), atol=4e-5, rtol=4e-5))
            # Cached constants must not retain an autograd graph or mix order.
            with torch.no_grad():
                observed = candidate.encode_observation(None if family == 'prior' else x.to(device))
                again = candidate.answer_many(observed, [e['inputs'] for e in queries])
                reverse = candidate.answer_many(observed, [e['inputs'] for e in reversed(queries)])
                for i, e in enumerate(queries):
                    count = len(e['inputs']['candidates'])
                    close(again[i, :count].cpu().numpy(), reverse[len(queries) - 1 - i, :count].cpu().numpy(), atol=3e-5, rtol=3e-5)
            reference.descriptors.end_source()
            if candidate._source_program_bytes > candidate.source_program_budget:
                raise AssertionError('Source program cache exceeded its byte budget.')
        errors[family] = {'logit_max_error': maximum, 'gradient_max_error': gradient_error, 'loss_max_error': loss_error}
    return errors


def verify_cross_source_batches(examples, windows, projector, *, device):
    """Compare the declared macro objective with scalar real-query execution."""
    from collections import Counter, defaultdict
    from neurosym.decoder_fit import device_windows
    from neurosym.decoder_minibatch import PackedBatches
    groups = defaultdict(list)
    for e in examples:
        groups[e['source_id']].append(e)
    counts = Counter(q[0]['story_id'] for q in groups.values())
    errors = {}
    for family in ('prior', 'linear', 'mlp', 'structured'):
        torch.manual_seed(11)
        reference = SemanticDecoder(family, (8, 4, 3), query_vocabulary(examples), hidden=16,
                                    site_mask=projector.site_mask).to(device)
        reference.source_batched = False
        candidate = copy.deepcopy(reference)
        candidate.source_batched = True
        staged = None if family == 'prior' else device_windows(windows, device)
        # Size 4 exercises different observations and an incomplete last batch.
        packed = PackedBatches(candidate, examples, staged, size=4)
        expected_logits, reference_loss = {}, 0.
        for source, queries in groups.items():
            reference.descriptors.begin_source()
            observed = reference.encode_observation(None if family == 'prior' else staged[source][0])
            source_loss = 0.
            for e in queries:
                logits = reference.answer(observed, e['inputs'])[0]
                expected_logits[e['query_id']] = logits.detach().cpu().numpy()
                source_loss = source_loss + e['weight'] * acceptable_loss(logits, e['acceptable_indices'])
            source_loss = source_loss / (sum(e['weight'] for e in queries) * counts[queries[0]['story_id']] * len(counts))
            reference_loss += float(source_loss.detach().cpu())
            source_loss.backward()
            reference.descriptors.end_source()
            reference.end_observation()
        maximum, candidate_loss = 0., 0.
        for index in range(len(packed.batches)):
            logits, targets, weights, rows = packed.logits(candidate, index)
            for row, (e, _, _) in zip(logits, rows, strict=True):
                expected = expected_logits[e['query_id']]
                maximum = max(maximum, close(expected, row[:len(expected)].detach().cpu().numpy(), atol=3e-5, rtol=3e-5))
                if not torch.isneginf(row[len(expected):]).all():
                    raise AssertionError('Multi-source padding changes candidate support.')
            loss = (choice_losses(logits, targets) * weights).sum() / len(packed.batches)
            candidate_loss += float(loss.detach().cpu())
            loss.backward()
        loss_error = close(np.asarray(reference_loss), np.asarray(candidate_loss), atol=3e-5, rtol=3e-5)
        gradient_error = 0.
        for (name, a), (other, b) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
            if name != other or (a.grad is None) != (b.grad is None):
                raise AssertionError('Multi-source gradient support changed: ' + name)
            if a.grad is not None:
                gradient_error = max(gradient_error, close(a.grad.cpu().numpy(), b.grad.cpu().numpy(), atol=5e-5, rtol=5e-5))
        close(np.asarray(reference_loss), np.asarray(packed.validation_loss(candidate)), atol=3e-5, rtol=3e-5)
        errors[family] = {'logit_max_error': maximum, 'macro_loss_error': loss_error, 'gradient_max_error': gradient_error,
                          'sources': len(groups), 'batches': len(packed.batches)}
    return errors


def verify_protocol(data, manifest):
    from collections import Counter
    from neurosym.protocol import selection_partition, selection_options, encoding_mixtures
    from neurosym.experiment_plan import experiment_plan
    plan = experiment_plan(data)
    for name, split in plan['partitions'].items():
        validation = [s for fold in split['inner'] for s in fold['validation']]
        if len(split['inner']) != 3 or sorted(validation) != sorted(split['train']) or set(split['train']) & set(split['test']):
            raise AssertionError('Whole-story selection coverage changed: ' + name)
        for fold in split['inner']:
            if set(fold['train']) & set(fold['validation']) or set(fold['train'] + fold['validation']) != set(split['train']):
                raise AssertionError('An inner fold leaks or omits stories.')
    for j in manifest['jobs']:
        if j['kind'] != 'decoder':
            continue
        o = j['options']
        a = selection_options(data, o)
        b = selection_options(data, {**o, 'seed': 47, 'retrain_null': True})
        if a != b:
            raise AssertionError('Evaluation seed/null receives different selection settings.')
        expected = {'kind': 'select-decoder', 'options': {**a}}
        if o.get('require_prepared'):
            expected['options']['require_prepared'] = True
        if object_hash(expected) not in j['dependencies']:
            raise AssertionError('Missing canonical selection dependency.')
        if o.get('model'):
            model = next(m for m in plan['models'] if m['id'] == o['model'])
            if o['layer'] != model['primary_layer'] and (o['family'] != 'linear' or o['seed'] != 11 or o.get('retrain_null')):
                raise AssertionError('Expensive decoder grid expanded into descriptive layers.')
        if o.get('export_traces') and (o['family'] != 'structured' or not o['fold'].startswith('stories:')):
            raise AssertionError('Unused dense traces entered the execution inventory.')
    weights = encoding_mixtures(data.config['encoding'], ['presentation', 'C', 'BC'])
    if len(weights) != 25 or not np.allclose(np.sum(weights, axis=1), 1):
        raise AssertionError('Encoding search union lost candidates or normalization.')
    # Validate a DAG, not just that every dependency name exists.
    available = set()
    pending = {j['id']: set(j['dependencies']) for j in manifest['jobs']}
    while pending:
        ready = {k for k, dependencies in pending.items() if dependencies <= available}
        if not ready:
            raise AssertionError('Execution graph has a cycle or missing parent.')
        available.update(ready)
        for key in ready:
            del pending[key]
    reports = [j for j in manifest['jobs'] if j['kind'] == 'study-report']
    if len(reports) != 1 or any(j['id'] not in reports[0]['dependencies'] for j in manifest['jobs']
                               if j['kind'] in {'decoder', 'encoding', 'compare-geometry-panel', 'semantic-coverage'}):
        raise AssertionError('Study reporting omits declared results.')
    return {'counts_by_kind': dict(Counter(j['kind'] for j in manifest['jobs'])),
            'selection_folds': 3, 'encoding_weight_candidates': len(weights), 'dag_verified': True}


def verify_exports(data, projector, *, device):
    """Recovery checks on two complete real sources and released text arrays."""
    from neurosym.grounding_checks import faithfulness
    sources = {s['id']: s for s in data.semantics.records('story_01', 'sources')
               if s['decoder_timing_eligible'] and s['raw_feature_bin'] >= 4}
    ordered = sorted(sources, key=lambda k: sources[k]['raw_feature_bin'])
    selected = {key: sources[key] for key in (ordered[0], ordered[-1])}
    examples = [e for e in data.examples('story_01') if e['source_id'] in selected]
    values = projector.transform(data.reader.features('story_01', ['english1000'], trim=False)['english1000'])
    windows = {key: values[s['raw_feature_bin']-4:s['raw_feature_bin']].transpose(1, 0, 2)[None] for key, s in selected.items()}
    torch.manual_seed(11)
    model = SemanticDecoder('structured', (8, 4, 3), query_vocabulary(examples), hidden=16,
                            site_mask=projector.site_mask).to(device)
    def interrupt():
        raise YieldRequested('Intentional real-source export recovery check.')
    with tempfile.TemporaryDirectory(dir=data.root / 'artifacts') as temporary:
        path = Path(temporary)
        with h5py.File(path / 'traces.h5', 'a') as file:
            try:
                with patch.object(deadline, 'check', side_effect=interrupt):
                    evaluate(model, examples, windows, trace_file=file)
            except YieldRequested:
                pass
            recovered = evaluate(model, examples, windows, trace_file=file)
            repeated = evaluate(model, examples, windows, trace_file=file)
            if recovered != repeated:
                raise AssertionError('Saved trace predictions differ on recovery.')
        reference = evaluate(model, examples, windows)
        by_id = {(r['query_id'], r['repeat']): r for r in reference}
        for row in recovered:
            close(np.asarray(row['probabilities']), np.asarray(by_id[row['query_id'], row['repeat']]['probabilities']), atol=3e-5, rtol=3e-5)
        original = faithfulness(model, examples, windows, selected, fraction=.1, seed=11)
        checkpoint = path / 'faithfulness.json'
        try:
            with patch.object(deadline, 'due', return_value=True), patch.object(deadline, 'check', side_effect=interrupt):
                faithfulness(model, examples, windows, selected, fraction=.1, seed=11, checkpoint=checkpoint)
        except YieldRequested:
            pass
        resumed = faithfulness(model, examples, windows, selected, fraction=.1, seed=11, checkpoint=checkpoint)
        if original != resumed:
            raise AssertionError('Faithfulness masks/metrics changed after continuation.')
    return {'actual_sources': list(selected), 'queries': len(examples), 'trace_and_faithfulness_recovery': True}


def verify_geometry_storage(data, examples, windows, *, device):
    from collections import defaultdict
    from neurosym.geometry import semantic_occurrences, summarize_geometry
    from neurosym.io import read_json
    records = semantic_occurrences(data, sorted({e['story_id'] for e in examples}), 'concept')
    vectors = {(r['source_id'], r['item_id']): windows[r['source_id']].mean(0).reshape(-1)
               for r in records if r['source_id'] in windows}
    cells = defaultdict(list)
    for r in records:
        key = (r['source_id'], r['item_id'])
        if key in vectors:
            cells[r['item_id'], r['story_id']].append(vectors[key].astype(np.float64))
    with tempfile.TemporaryDirectory(dir=data.root / 'artifacts') as temporary:
        path = Path(temporary)
        report = summarize_geometry(records, vectors, path, min_stories=1, device=device)
        if not report['available']:
            raise AssertionError('Real-source geometry verification has no supported concepts.')
        expected = np.stack([np.mean([np.mean(values, axis=0) for (item, _), values in cells.items() if item == key], axis=0)
                             for key in report['items']])
        with h5py.File(path / 'geometry.h5', 'r') as file:
            close(expected.astype(np.float32), file['signatures'][()], atol=1e-7, rtol=1e-7)
            rdm = file['rdm'][()]
        summarize_geometry(records, vectors, path, min_stories=1, device=device,
                           retain_signatures=False, retain_story_means=False)
        with h5py.File(path / 'geometry.h5', 'r') as file:
            if 'signatures' in file or 'story_means' in file:
                raise AssertionError('Compact geometry retained redundant wide vectors.')
            close(rdm, file['rdm'][()], atol=0, rtol=0)
    return {'items': len(report['items']), 'streaming_means_and_compact_rdm': True}


def verify(data, *, device='cpu'):
    print('COMPUTE CHECK: reading actual features and reviewed query operators', flush=True)
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA equivalence must be checked on the actual compute node; no CPU fallback.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    arrays, projector, examples, windows = real_inputs(data)
    from neurosym.encoding import validation_row_scale
    # Different lengths of two released text-feature series exercise equal-story
    # scaling without inventing fMRI targets or substituting them in a fit.
    actual = [a['english1000'][20:20 + n, :24] for a, n in zip(arrays, [32, 64], strict=True)]
    masks = {str(i): np.isfinite(v).all(1) for i, v in enumerate(actual)}
    scale = validation_row_scale(masks, list(masks))
    close(np.asarray(np.mean([np.square(v).mean() for v in actual])),
          np.asarray(np.square(np.concatenate(actual) * scale[:, None]).mean()))
    print('COMPUTE CHECK: merged story PCA and device projection', flush=True)
    pca_inputs = [a['english1000'] for a in arrays]
    assignment = np.arange(pca_inputs[0].shape[1]) % 8
    pca_ref = SiteProjector(assignment, 3).fit(np.concatenate(pca_inputs), ['story_01', 'story_02'])
    moments = [story_statistics(a, assignment, device=device) for a in pca_inputs]
    pca_errors = {}
    for solver in ('cpu', 'device'):
        merged = fit_statistics(SiteProjector(assignment, 3), moments, ['story_01', 'story_02'], device=device, eigensolver=solver)
        if not np.array_equal(pca_ref.site_mask, merged.site_mask):
            raise AssertionError('Moment merging changed site availability.')
        for first, second in zip(pca_ref.parameters, merged.parameters, strict=True):
            if not np.array_equal(first[0], second[0]) or first[3].shape != second[3].shape:
                raise AssertionError('Moment merging changed PCA positions/rank.')
            close(first[1], second[1])
            close(first[2], second[2])
            close(first[3] @ first[3].T, second[3] @ second[3].T)
        pca_errors[solver] = close(pca_ref.transform(pca_inputs[1]), merged.transform(pca_inputs[1], device=device), atol=3e-5, rtol=3e-5)
    math = FP64(device)
    # Fixed real rows bound the numerical check; production inventories are unchanged.
    x = {'letters': arrays[0]['letters'][20:148], 'semantics': arrays[0]['english1000'][20:148, :24]}
    z = {'letters': arrays[1]['letters'][20:84], 'semantics': arrays[1]['english1000'][20:84, :24]}
    y = np.concatenate([arrays[0][n][20:148] for n in ('numwords', 'numletters')], 1).astype(np.float64)
    target = np.concatenate([arrays[1][n][20:84] for n in ('numwords', 'numletters')], 1).astype(np.float64)
    reference = GroupRidge(1.7, {'letters': .4, 'semantics': .6}, solver='dual').fit_design(x)
    candidate = GroupRidge(1.7, reference.weights, device=device).fit_design(x)
    operator_error = close(reference.prediction_operator(z), candidate.prediction_operator(z))
    rd, rm = reference.coefficients(y)
    cd, cm = candidate.coefficients(y)
    prediction_error = close(reference.kernel(reference.scaler.transform(z), reference.training) @ rd + rm,
                             math.numpy(candidate.kernel(candidate.scaler.transform(z), candidate.training) @ cd + cm))
    if candidate.mode != 'primal':
        raise AssertionError('Real small-feature case did not select the primal solve.')
    contributions = candidate.group_operators(z)
    for group, operator in contributions.items():
        expected = reference.weights[group] * reference.scaler.transform(z)[group] @ reference.training[group].T @ rd
        close(expected, math.numpy(operator @ math.array(y - y.mean(0))))
    predicted = math.numpy(sum(contributions.values()) @ math.array(y - y.mean(0)) + math.array(y.mean(0)))
    close(predicted, reference.prediction_operator(z) @ (y - y.mean(0)) + y.mean(0))
    primal_design = math.concatenate([np.sqrt(candidate.weights[g]) * candidate.scaler.transform(z)[g] for g in candidate.weights])
    beta = math.solve(candidate.factor, candidate.design.T @ math.array(y - y.mean(0)))
    close(predicted, math.numpy(primal_design @ beta + math.array(y.mean(0))))
    from neurosym.analysis_runs import column_correlation
    close(column_correlation(predicted, target), math.numpy(response_correlations(math, math.array(predicted), math.array(target))))
    wide = {'semantics': arrays[0]['english1000'][20:148]}
    wide_test = {'semantics': arrays[1]['english1000'][20:84]}
    auto_wide = GroupRidge(1.7, {'semantics': 1.0}, device=device).fit_design(wide)
    if auto_wide.mode != 'dual':
        raise AssertionError('Real wide-feature case did not select the dual solve.')
    close(GroupRidge(1.7, {'semantics': 1.0}, solver='dual').fit_design(wide).prediction_operator(wide_test),
          auto_wide.prediction_operator(wide_test))
    tx, vz = reference.training, reference.scaler.transform(z)
    yy = (y - y.mean(0)) / y.std(0)
    zz = (target - y.mean(0)) / y.std(0)
    direct_scores, optimized_scores = [], []
    products = {g: (math.product(tx[g], tx[g].T), math.product(vz[g], tx[g].T)) for g in tx}
    moments = (math.product(yy, yy.T) / yy.shape[1], math.product(yy, zz.T) / yy.shape[1], np.sum(zz * zz) / yy.shape[1])
    mixtures = [np.ones(2) / 2, *np.random.default_rng(11).dirichlet(np.ones(2), 8)]
    for weights in mixtures:
        kernel = sum(w * products[g][0] for g, w in zip(tx, weights, strict=True))
        cross = sum(w * products[g][1] for g, w in zip(tx, weights, strict=True))
        vals, vecs = math.eigh(kernel)
        designs = {g: {'train': tx[g], 'validation': vz[g]} for g in tx}
        identities = {g: {'numerical_check': data.semantics.build_hash, 'actual_rows': [20, 148, 84],
                           'group': g, 'device': device} for g in tx}
        primal = spectrum(data, {'primal': True, 'designs': designs, 'identities': identities}, dict(zip(tx, weights, strict=True)), math)
        primal_scores = spectral_losses(primal, tuple(math.array(v) for v in moments), data.config['encoding']['alphas'], math, primal=True)
        for alpha_index, alpha in enumerate(data.config['encoding']['alphas']):
            old = GroupRidge(alpha, dict(zip(tx, weights, strict=True)), solver='dual').fit_design(x).prediction_operator(z)
            op = (cross @ vecs / (math.nonnegative(vals) + alpha)) @ vecs.T
            direct_scores.append(float(np.mean((old @ yy - zz) ** 2)))
            optimized_scores.append(float((((op @ moments[0]) * op).sum() - 2 * (op * moments[1].T).sum() + moments[2]) / len(op)))
            close(np.asarray(direct_scores[-1]), np.asarray(primal_scores[alpha_index]))
    score_error = close(np.array(direct_scores), np.array(optimized_scores))
    if np.argmin(direct_scores) != np.argmin(optimized_scores):
        raise AssertionError('Encoding selection changed on actual released inputs.')

    print('COMPUTE CHECK: float64 ridge passed; comparing decoder logits and gradients', flush=True)
    vocabulary = query_vocabulary(examples)
    errors = {}
    for family in ('prior', 'linear', 'mlp', 'structured'):
        torch.manual_seed(11)
        ref = SemanticDecoder(family, (8, 4, 3), vocabulary, hidden=16, site_mask=projector.site_mask)
        ref.descriptors.batched = False
        ref.factorized_relations = False
        ref.reuse_observation = False
        ref.reuse_query_plans = False
        ref.source_batched = False
        opt = copy.deepcopy(ref).to(device)
        opt.descriptors.batched = True
        opt.factorized_relations = True
        opt.reuse_observation = True
        opt.reuse_query_plans = True
        opt.source_batched = True
        maximum = 0.
        for e in examples:
            observed = torch.from_numpy(windows[e['source_id']][0])
            ref.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            a, at = ref(observed, e['inputs'], capture=True)
            b, bt = opt(observed.to(device), e['inputs'], capture=True)
            maximum = max(maximum, close(a.detach().numpy(), b.detach().cpu().numpy(), atol=2e-5, rtol=2e-5))
            acceptable_loss(a, e['acceptable_indices']).backward()
            acceptable_loss(b, e['acceptable_indices']).backward()
            for (name, pa), (other, pb) in zip(ref.named_parameters(), opt.named_parameters(), strict=True):
                if name != other or (pa.grad is None) != (pb.grad is None):
                    raise AssertionError('Gradient parameter inventory changed.')
                if pa.grad is not None:
                    close(pa.grad.numpy(), pb.grad.cpu().numpy(), atol=3e-5, rtol=3e-5)
            if family == 'structured':
                for aa, bb in zip(at['primitive'], bt['primitive'], strict=True):
                    close(aa['routing'].detach().numpy(), bb['routing'].detach().cpu().numpy(), atol=2e-5, rtol=2e-5)
        detailed = evaluate(opt, examples, windows)
        original = evaluate(ref, examples, windows)
        row_key = lambda r: (r['query_id'], r['repeat'])
        for a, b in zip(sorted(original, key=row_key), sorted(detailed, key=row_key), strict=True):
            close(np.asarray(a['probabilities']), np.asarray(b['probabilities']), atol=2e-5, rtol=2e-5)
        lean = evaluate(opt, examples, windows, probabilities=False)
        if decoder_metrics(detailed) != decoder_metrics(lean):
            raise AssertionError('Validation without probability transfers changed metrics.')
        errors[family] = maximum

    print('COMPUTE CHECK: source-batched logits, weighted losses, gradients and candidate order', flush=True)
    _, batch_projector, batch_examples, batch_windows = real_inputs(data, include_full_source=True)
    batch_errors = verify_source_batches(batch_examples, batch_windows, batch_projector, device=device)
    cross_source_errors = verify_cross_source_batches(batch_examples, batch_windows, batch_projector, device=device)
    export_checks = verify_exports(data, projector, device=device)
    geometry_storage_checks = verify_geometry_storage(data, batch_examples, batch_windows, device=device)

    # Actual prior eligibility and repeats need no fMRI arrays or dummy windows.
    prior_examples, no_windows, _ = examples_and_windows(data, ['story_11'], None,
        {'family': 'prior', 'modality': 'brain', 'subject': 'subject01'})
    if no_windows is not None or {e['observation_repeats'] for e in prior_examples} != {data.reader.contract['subjects']['subject01']['story_11']['repeats']}:
        raise AssertionError('Query-only prior lost actual heldout repetition metadata.')

    print('COMPUTE CHECK: checking cache identity and interrupted-epoch recovery', flush=True)
    with tempfile.TemporaryDirectory(dir=data.root / 'artifacts') as temporary:
        path = Path(temporary)
        # Serialize compact operators on these actual textual numeric inputs.
        # This is not an encoding run and is never represented as brain data.
        with h5py.File(path / 'operators.h5', 'w') as file:
            for name, value in contributions.items():
                file.create_dataset(name, data=math.numpy(value), compression='lzf')
        with h5py.File(path / 'operators.h5', 'r') as file:
            restored_operator = sum(math.array(file[g][()]) for g in candidate.weights)
            close(predicted, math.numpy(restored_operator @ math.array(y - y.mean(0)) + math.array(y.mean(0))))
        memory = ResidentCache(y.nbytes)
        memory.get('training-counts', lambda: y)
        memory.get('heldout-counts', lambda: target)
        if memory.bytes > memory.max_bytes or memory.hits:
            raise AssertionError('Resident byte budget was not enforced.')
        policy = {'cache_gib': 1/1024, 'cache_entry_gib': 1/2048, 'minimum_free_gib': 0}
        bounded = FoldCache(path / 'bounded', policy=policy)
        bounded.arrays('actual-text-features', {'rows': [20, 148]}, lambda: {'values': y})
        state = bounded.budget.state()
        if state['reservations'] or state['published_entries'] != 1 or not 0 < state['published_bytes'] <= bounded.budget.limit:
            raise AssertionError('Cache publication budget/reservation changed.')
        bounded.budget.limit = state['published_bytes']
        try:
            bounded.arrays('actual-text-features', {'rows': [20, 84]}, lambda: {'values': target})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Cache exceeded its configured budget.')
        # Never publish numerical-check fits into the research cache.
        numerical_data = copy.copy(data)
        numerical_data.cache = FoldCache(path / 'cache')
        identity = {'build': data.semantics.build_hash, 'rows': [20, 148], 'split': ['story_01']}
        payload = {'values': y}
        numerical_data.cache.arrays('verification', identity, lambda: payload)
        restored = numerical_data.cache.arrays('verification', identity, lambda: (_ for _ in ()).throw(AssertionError('Cache miss')))
        close(restored['values'], y, atol=0, rtol=0)
        FoldCache(path / 'cache', require=True).arrays('verification', identity, lambda: payload)
        try:
            FoldCache(path / 'cache', require=True).arrays('verification', {**identity, 'split': ['story_02']}, lambda: payload)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Different fold reused a prepared cache entry.')
        numerical_data.config = copy.deepcopy(data.config)
        numerical_data.config['decoder']['source_batch_size'] = 2
        base = SemanticDecoder('linear', (8, 4, 3), vocabulary, hidden=16, site_mask=projector.site_mask).to(device)
        start_rng = torch.get_rng_state()
        start_cuda = torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
        def train(model, checkpoint=None):
            torch.set_rng_state(start_rng)
            if start_cuda is not None:
                torch.cuda.set_rng_state_all(start_cuda)
            return train_decoder(numerical_data, model, examples, windows, learning_rate=.001, epochs=2,
                                 seed=11, validation=(examples, windows), patience=10, checkpoint=checkpoint)
        uninterrupted, interrupted = copy.deepcopy(base), copy.deepcopy(base)
        expected = train(uninterrupted)
        checkpoint = path / 'interrupted.pt'
        def interrupt_after_save(value, destination):
            atomic_torch_save(value, destination)
            raise InterruptedError('Intentional interruption after a complete numerical-check epoch.')
        try:
            with patch('neurosym.decoder_minibatch.atomic_torch_save', interrupt_after_save):
                train(interrupted, checkpoint)
        except InterruptedError:
            pass
        else:
            raise AssertionError('Checkpoint interruption was not exercised.')
        resumed = copy.deepcopy(base)
        result = train(resumed, checkpoint)
        resume_tolerance = 3e-5 if device.startswith('cuda') else 0.
        checkpoint_error = max(close(v.detach().cpu().numpy(), resumed.state_dict()[k].cpu().numpy(),
                                     atol=resume_tolerance, rtol=resume_tolerance)
                               for k, v in uninterrupted.state_dict().items() if v.dtype != torch.bool)
        if result['best_epoch'] != expected['best_epoch'] or not np.isclose(result['best_validation_nll'], expected['best_validation_nll'],
                                                                          atol=resume_tolerance, rtol=resume_tolerance):
            raise AssertionError('Resume changed validation selection.')
        # Interrupt after an actual source update, then restore optimizer/order/RNG.
        mid_checkpoint = path / 'mid-source.pt'
        mid_model = copy.deepcopy(base)
        def yield_now():
            raise YieldRequested('Intentional numerical-check allocation boundary.')
        try:
            with patch.object(deadline, 'due', return_value=True), patch.object(deadline, 'check', side_effect=yield_now):
                train(mid_model, mid_checkpoint)
        except YieldRequested:
            saved = torch.load(mid_checkpoint, map_location='cpu', weights_only=True)
            if saved['completed'] != 0 or saved['pending']['cursor'] != 1:
                raise AssertionError('Mid-epoch source cursor was not checkpointed.')
        else:
            raise AssertionError('Mid-epoch yield was not exercised.')
        continued = copy.deepcopy(base)
        mid_result = train(continued, mid_checkpoint)
        for key, value in uninterrupted.state_dict().items():
            if value.dtype != torch.bool:
                close(value.detach().cpu().numpy(), continued.state_dict()[key].cpu().numpy(), atol=resume_tolerance, rtol=resume_tolerance)
        if mid_result['best_epoch'] != expected['best_epoch']:
            raise AssertionError('Mid-epoch continuation changed selected epoch.')
        # Process-owned locks release on exception; they are never deleted.
        lock_path = path / 'actual-cache.oslock'
        try:
            with process_lock(lock_path):
                raise InterruptedError('Release check')
        except InterruptedError:
            pass
        with process_lock(lock_path, wait=False):
            if not lock_path.exists():
                raise AssertionError('OS lock inode disappeared.')
        # Deduplication keeps per-query paths while storing identical arrays once.
        from neurosym.decoder_fit import write_trace
        with h5py.File(path / 'trace-storage.h5', 'w') as file:
            memo = {}
            write_trace(file.create_group('first'), {'observed': y}, shared_arrays=memo)
            write_trace(file.create_group('second'), {'observed': y}, shared_arrays=memo)
            if file['first/observed'].id != file['second/observed'].id:
                raise AssertionError('Repeated trace arrays were not hard-linked.')
            close(file['second/observed'][()], y, atol=0, rtol=0)
        prior = SemanticDecoder('prior', None, vocabulary, hidden=16).to(device)
        result = train_decoder(numerical_data, prior, examples, None, learning_rate=.001, epochs=1, seed=11)
        duplicate = SemanticDecoder('prior', None, vocabulary, hidden=16).to(device)
        cached = train_decoder(numerical_data, duplicate, examples, None, learning_rate=.001, epochs=1, seed=11)
        if result['cache_hit'] or not cached['cache_hit'] or result['shared_prior_fit'] != cached['shared_prior_fit']:
            raise AssertionError('Identical query-only prior was not reused.')
        for key, value in prior.state_dict().items():
            if not torch.equal(value, duplicate.state_dict()[key]):
                raise AssertionError('Cached prior weights changed.')
        changed = [{**e, 'weight': e['weight'] / 2} for e in examples]
        if fit_identity(numerical_data, prior, examples, None, None) == fit_identity(numerical_data, prior, changed, None, None):
            raise AssertionError('Changed prior weights did not invalidate cache.')

    a, av, _ = cosine_rdm(arrays[0]['english1000'][20:36])
    b, bv, _ = cosine_rdm(arrays[0]['letters'][20:36])
    device_rdm, device_valid, _ = cosine_rdm(arrays[0]['english1000'][20:36], device=device)
    close(a[np.ix_(av, av)], device_rdm[np.ix_(av, av)])
    if not np.array_equal(av, device_valid):
        raise AssertionError('Device geometry changed item availability.')
    valid = av & bv
    a, b = a[np.ix_(valid, valid)], b[np.ix_(valid, valid)]
    # Discrete distances from the same observations exercise tied rank handling.
    for second in (b, np.round(b, 1)):
        upper = np.triu_indices(len(a), 1)
        observed = spearmanr(a[upper], second[upper]).statistic
        rng, exceed = np.random.default_rng(11), 0
        for _ in range(127):
            order = rng.permutation(len(a))
            exceed += abs(spearmanr(a[upper], second[np.ix_(order, order)][upper]).statistic) >= abs(observed)
        comparison = rdm_comparison(a, second, permutations=127, seed=11, device=device)
        if comparison['two_sided_label_permutation_p'] != (1 + exceed) / 128:
            raise AssertionError('Rank reuse changed permutation tail counts.')
    for dtype in (np.float32, np.float64):
        maps = []
        for array in arrays:
            values = array['english1000'][20:36].astype(dtype)
            values -= values.mean(1, keepdims=True)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            maps.append(values)
        aa, bb = maps
        for second in (aa, bb):
            observed = float(np.sum(aa * second, axis=1).mean())
            rng, exceed = np.random.default_rng(11), 0
            for _ in range(127):
                order = rng.permutation(len(aa))
                exceed += float(np.sum(aa * second[order], axis=1).mean()) >= observed
            _, value, pvalue = map_label_permutation(aa, second, permutations=127, seed=11, device=device)
            if value != observed or pvalue != (1 + exceed) / 128:
                raise AssertionError('Map dot-product reuse changed the observed statistic or permutation tail.')
    manifest = execution_manifest(data)
    protocol_checks = verify_protocol(data, manifest)
    from submit_analysis import submission_groups, ready_workers, seconds
    stages = submission_groups(manifest)
    if sorted(i for s in stages for i in s['indices']) != list(range(len(manifest['jobs']))):
        raise AssertionError('Slurm chunks omit or duplicate manifest jobs.')
    if sorted(w for stage in stages for w in stage['workers']) != list(range(len(manifest['workers']))):
        raise AssertionError('Slurm worker inventory differs from declared panels.')
    for stage in stages:
        for worker_index in stage['workers']:
            worker = manifest['workers'][worker_index]
            if worker['dependencies'] != stage['dependencies'] or worker['resource'] != stage['resource']:
                raise AssertionError('Array dependency/resource differs from its worker.')
            for index in worker['indices']:
                job = manifest['jobs'][index]
                if job['dependencies'] != worker['dependencies'] or job['resource'] != worker['resource']:
                    raise AssertionError('Worker dependency/resource differs from its logical fit.')
    ids = {j['id'] for j in manifest['jobs']}
    allowed = set(range(len(manifest['workers'])))
    initially_ready = ready_workers(manifest, set(), set(), allowed)
    if any(manifest['workers'][i]['dependencies'] for indices in initially_ready.values() for i in indices):
        raise AssertionError('Dispatcher released a missing dependency.')
    completed_dependencies = {d for w in manifest['workers'] for d in w['dependencies']}
    ready = ready_workers(manifest, completed_dependencies, set(), allowed)
    expected_workers = {i for i, w in enumerate(manifest['workers']) if not all(manifest['jobs'][j]['id'] in completed_dependencies for j in w['indices'])}
    if {i for indices in ready.values() for i in indices} != expected_workers:
        raise AssertionError('Ready work was blocked by an unrelated scheduling lane.')
    if seconds('00:30:00') != 1800:
        raise AssertionError('Short allocation override changed.')
    if len(ids) != len(manifest['jobs']) or any(set(j['dependencies']) - ids for j in manifest['jobs']):
        raise AssertionError('Execution dependencies are missing or duplicated.')
    if any(j['options'].get('fold') == 'final' for j in manifest['jobs']):
        raise AssertionError('Development manifest includes final fits.')
    manifest_path = data.root / 'artifacts/execution' / manifest['content_hash'] / 'manifest.json'
    write_report(manifest_path, manifest)
    write_report(data.root / 'artifacts/execution/latest.json',
                 {'path': manifest_path.relative_to(data.root).as_posix(), 'content_hash': manifest['content_hash']})
    return {'status': 'verified', 'device': device, 'torch': str(torch.__version__),
            'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
            'semantic_build_hash': data.semantics.build_hash, 'code': code_identity(),
            'config_hash': object_hash(data.config), 'compute_config_hash': object_hash(manifest['resources']),
            'verification_script_sha256': file_hash(Path(__file__)),
             'execution_manifest_hash': manifest['content_hash'], 'fit_job_counts': manifest['counts'],
             'worker_counts': manifest['worker_counts'], 'submission_arrays': len(stages),
            'ridge_operator_max_error': operator_error, 'ridge_prediction_max_error': prediction_error,
            'ridge_selection_mse_max_error': score_error, 'decoder_logit_max_errors': errors,
              'source_batch_errors': batch_errors,
              'cross_source_batch_errors': cross_source_errors,
              'protocol_checks': protocol_checks,
              'export_checks': export_checks,
              'geometry_storage_checks': geometry_storage_checks,
             'source_batch_query_ids': [e['query_id'] for e in batch_examples],
             'merged_pca_projection_errors': pca_errors,
            'checkpoint_parameter_error': checkpoint_error, 'query_ids': [e['query_id'] for e in examples],
             'checks': ['adaptive primal/dual FP64 ridge, compact group operators and selection',
                        'dense/factorized decoder logits/gradients/routing and observation reuse',
                        'source-batched public programs, weighted losses, gradients and candidate order',
                        'multi-source ownership, macro objective/gradients and partial minibatches',
                       'validation metric transfers', 'actual query-only prior eligibility/repeats and fit reuse',
                         'fold cache serialization/invalidation/budgets', 'epoch and mid-source resume including optimizer/RNG/selection',
                        'story-moment PCA, subspaces/ranks and device projection', 'OS lock release and HDF trace sharing',
                       'label permutation ranks including ties', 'float32/float64 map permutation products',
                       'complete fit dependency inventory'],
            'scientific_fits_executed': False,
            'scope': 'Numerical checks use actual released text features and accepted query targets; no fMRI or modern hidden states are substituted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--build', type=Path)
    args = parser.parse_args()
    if os.name != 'nt' and (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login')):
        raise RuntimeError('Cluster numerical verification requires an allocated compute node.')
    print('COMPUTE CHECK: opening and validating the complete immutable build', flush=True)
    report = verify(AnalysisData(ROOT, build=args.build), device=args.device)
    path = ROOT / 'artifacts' / ('compute-verification-' + args.device.replace(':', '-') + '.json')
    write_report(path, report)
    if args.device.startswith('cuda'):
        write_report(ROOT / 'artifacts/compute-devices' / (object_hash(report['gpu']) + '.json'), report)
    print('COMPUTE NUMERICS VERIFIED:', path, flush=True)


if __name__ == '__main__':
    deadline.install()
    try:
        main()
    except YieldRequested as error:
        print('COMPUTE VERIFICATION YIELDED:', error, flush=True)
        sys.exit(75)
