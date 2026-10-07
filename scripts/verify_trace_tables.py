"""Compare new trace execution to collected real fitted GPU reference traces."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--limit', type=int, help='Short diagnostic: first N actual queries per source')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--geometry', action='store_true', help='Also verify exact signature deduplication and aggregation on real occurrences')
    args = parser.parse_args()
    wall_started = time.perf_counter()
    root = args.root.resolve()
    args.audit = args.audit or root / 'artifacts/trace-audit-1024054'
    import h5py
    import numpy as np
    import torch
    from neurosym.decoders import SemanticDecoder
    from neurosym.trace_store import Fields, iter_sources, export
    from neurosym.runtime import deadline, YieldRequested
    from neurosym.io import save_json
    from neurosym.execution import code_identity
    from neurosym.semantic_features import freeze_descriptor
    from neurosym.geometry import prepare_trace, trace_signature, vector_digest, semantic_occurrences
    if args.geometry:
        from types import SimpleNamespace
        from neurosym.semantic_features import SemanticDataset
        from neurosym.analysis_data import latest
        cfg = json.loads((root / 'configs/analysis.json').read_text())
        data = SimpleNamespace(semantics=SemanticDataset(latest(root / cfg['semantics'])))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    receipt = root / ('artifacts/trace-verification-' + args.device.split(':')[0] + '.json')
    save_json(receipt, {'status': 'running', 'code': code_identity(), 'complete_samples': args.limit is None})
    errors, reports = {}, []
    def close(a, b, key):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise AssertionError((key, a.shape, b.shape))
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        errors[key] = max(errors.get(key, 0.), error)
        if not np.allclose(a, b, atol=3e-5, rtol=3e-5):
            raise AssertionError((key, error))
    for path in sorted((args.audit / 'fits').iterdir()):
        description = json.loads((path / 'decoder.json').read_text())
        model = SemanticDecoder('structured', description['shape'], description['vocabulary'],
            hidden=description['hidden'], site_mask=description['site_mask']).to(args.device)
        model.load_state_dict(torch.load(path / 'weights.pt', map_location=args.device, weights_only=True))
        model.eval()
        names = json.loads((path / 'windows.json').read_text())
        with gzip.open(path / 'sample-examples.jsonl.gz', 'rt', encoding='utf-8') as file:
            examples = [json.loads(line) for line in file]
        from collections import defaultdict
        groups = defaultdict(list)
        for example in examples:
            example['inputs'] = freeze_descriptor(example['inputs']); groups[example['source_id']].append(example)
        if args.limit:
            groups = {source: rows[:args.limit] for source, rows in groups.items()}
        current_examples = [row for rows in groups.values() for row in rows]
        with tempfile.TemporaryDirectory(prefix='trace-verify-', dir=root / 'artifacts') as temporary, np.load(path / 'windows.npz') as windows, h5py.File(path / 'traces-sample.h5', 'r') as original, torch.no_grad():
            compact_path = Path(temporary) / 'traces.h5'
            actual_windows = {source: windows[str(names.index(source))] for source in groups}
            if args.geometry:
                from neurosym.decoder_minibatch import PackedBatches
                from neurosym.decoder_fit import device_windows
                original_identity = json.loads((path / 'identity.json').read_text())
                model.program_namespace = original_identity['semantic_build_hash']
                model.source_program_budget = 1024 * 2**20
                staged = device_windows(actual_windows, args.device)
                packed_first = PackedBatches(model, current_examples, staged, size=16)
                alternatives = []
                options = {k: v for k, v in original_identity['options'].items() if k != 'seed'}
                for candidate_path in (args.audit / 'partial-fits').iterdir():
                    other_identity = json.loads((candidate_path / 'identity.json').read_text())
                    if {k: v for k, v in other_identity['options'].items() if k != 'seed'} == options:
                        alternatives.append(candidate_path)
                if len(alternatives) != 1:
                    raise ValueError('Expected one real completed alternative-seed refit for program-cache validation.')
                d = json.loads((alternatives[0] / 'decoder.json').read_text())
                other = SemanticDecoder('structured', d['shape'], d['vocabulary'], hidden=d['hidden'], site_mask=d['site_mask']).to(args.device)
                other.load_state_dict(torch.load(alternatives[0] / 'weights.pt', map_location=args.device, weights_only=True))
                other.program_namespace, other.source_program_budget = model.program_namespace, model.source_program_budget
                reused = PackedBatches(other, current_examples, staged, size=16)
                other.program_namespace = None
                fresh = PackedBatches(other, current_examples, staged, size=16)
                for index in range(len(reused.batches)):
                    if reused.batches[index][0] is not packed_first.batches[index][0]:
                        raise AssertionError('Compatible real fits did not reuse query constants')
                    if not torch.equal(reused.logits(other, index)[0], fresh.logits(other, index)[0]):
                        raise AssertionError('Program reuse leaked fitted parameters across actual seeds')
                from neurosym.decoder_stages import commit_fitted, load_fitted, prediction_stage
                from neurosym.analysis_data import SiteProjector
                import shutil
                fitted_path = Path(temporary) / 'fitted'
                fitted_path.mkdir()
                shutil.copyfile(path / 'selection.json', fitted_path / 'selection.json')
                projector = SiteProjector.load(path / 'projector')
                commit_fitted(fitted_path, original_identity, model, projector, description, None)
                restored, _, _ = load_fitted(fitted_path, original_identity, args.device, cfg['decoder'])
                if any(not torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()):
                    raise AssertionError('Committed fitted stage changed actual trained weights')
                del packed_first, reused, fresh, staged, other, restored
            # Exercise a real safe-point interruption, then resume the same
            # source file. No generated observations or substitute queries.
            previous_advance = deadline.advance
            def request_yield():
                previous_advance(); deadline.requested = True
            deadline.advance = request_yield
            try:
                with h5py.File(compact_path, 'w') as file:
                    file.attrs['run_identity'] = path.name
                    export(model, current_examples, actual_windows, file)
            except YieldRequested:
                pass
            finally:
                deadline.advance = previous_advance
                deadline.requested = False
            import neurosym.trace_store as store
            previous_capture, captured_sources = store.capture_source, []
            def resumed_capture(model, rows, value):
                captured_sources.append(rows[0]['source_id'])
                return previous_capture(model, rows, value)
            store.capture_source = resumed_capture
            try:
                with h5py.File(compact_path, 'a') as file:
                    predictions = export(model, current_examples, actual_windows, file)
                    file.attrs['complete'] = True
            finally:
                store.capture_source = previous_capture
            first_source = next(iter(groups))
            if len(actual_windows[first_source]) == 1 and first_source in captured_sources:
                raise AssertionError('Committed source was recomputed on resume')
            with h5py.File(compact_path, 'r') as file:
                export_phases = json.loads(file.attrs['phase_seconds'])
                candidates = {source: {(q, int(r)): group for q, r, group in stream} for source, stream in iter_sources(file)}
            predicted = {(r['query_id'], r['repeat']): r for r in predictions}
            if args.geometry:
                prediction_stage(fitted_path, 'predictions', original_identity, lambda: predictions)
                def no_recomputation():
                    raise AssertionError('A committed prediction stage was recomputed')
                if prediction_stage(fitted_path, 'predictions', original_identity, no_recomputation) != predictions:
                    raise AssertionError('Prediction-stage resume changed results')
            source_expectations = {}
            for source, rows in groups.items():
                started = time.perf_counter()
                records = []
                collected = [{}, {}, {}]
                first_signature_keys = [{}, {}, {}]
                if args.geometry:
                    for scope in ['pooled', 'scoped']:
                        for kind in ['concept', 'predicate', 'literal', 'role', 'discourse', 'scope', 'reference', 'identity', 'qualification', 'state_update']:
                            records.extend(r for r in semantic_occurrences(data, [rows[0]['story_id']], kind, scope_mode=scope) if r['source_id'] == source)
                for repeat, value in enumerate(windows[str(names.index(source))]):
                    if args.geometry:
                        model.descriptors.begin_source()
                        scalar_observed = model.encode_observation(torch.tensor(value, device=args.device))
                    for index, example in enumerate(rows):
                        row = dict(example, repeat=repeat)
                        reference = original[example['query_id'] + '/' + str(repeat)]
                        candidate = candidates[source][example['query_id'], repeat]
                        prediction = json.loads(bytes(reference['prediction_record'][()]).decode())
                        close(prediction['probabilities'], predicted[example['query_id'], repeat]['probabilities'], 'probabilities')
                        for key in ('site_mask', 'latent'):
                            close(reference[key][()], candidate[key][()], key)
                        for key, names_to_check, attribute in [('primitive', ('scores', 'routing'), 'description'),
                                ('relations', ('left_factor', 'right_factor'), 'operation'),
                                ('intermediate', ('routing',), 'operation')]:
                            if len(reference[key]) != len(candidate[key]):
                                raise AssertionError('Changed trace membership: ' + key)
                            for slot in reference[key]:
                                a, b = reference[key][slot], candidate[key][slot]
                                if json.loads(a.attrs[attribute]) != json.loads(b.attrs[attribute]):
                                    raise AssertionError('Changed descriptor order: ' + key)
                                for name in names_to_check:
                                    close(a[name][()], b[name][()], key + '/' + name)
                        if args.geometry:
                            from neurosym.decoders import detached_trace
                            _, captured = model.answer(scalar_observed, example['inputs'], capture=True)
                            captured = detached_trace(captured)
                            scalar = Fields({k: v for k, v in captured.items() if isinstance(v, np.ndarray)})
                            for key in ('primitive', 'relations', 'intermediate'):
                                scalar[key] = {str(i): Fields({k: v for k, v in item.items() if isinstance(v, np.ndarray)},
                                    **{k: json.dumps(v) for k, v in item.items() if not isinstance(v, np.ndarray)})
                                    for i, item in sorted(enumerate(captured[key]), key=lambda row: str(row[0]))}
                            for which, group in enumerate([reference, candidate, scalar]):
                                prepared = prepare_trace(group)
                                for record in records:
                                    vector = trace_signature(group, record, 'grounding', prepared=prepared)
                                    if vector is not None:
                                        item, digest = record['item_id'], vector_digest(vector)
                                        collected[which].setdefault(item, {})[digest] = vector
                                        order = (example['query_id'], str(repeat))
                                        first = first_signature_keys[which].setdefault(item, {})
                                        first[digest] = min(first.get(digest, order), order)
                            for key in ['site_mask', 'latent']:
                                if not np.array_equal(candidate[key][()], scalar[key][()]):
                                    raise AssertionError('Compact/scalar values are not bit-identical: ' + key)
                            for key in ['primitive', 'relations', 'intermediate']:
                                if list(candidate[key]) != list(scalar[key]):
                                    raise AssertionError('Legacy field iteration order changed: ' + key)
                                for slot in scalar[key]:
                                    for field in scalar[key][slot]:
                                        if not np.array_equal(candidate[key][slot][field][()], scalar[key][slot][field][()]):
                                            raise AssertionError('Compact/scalar arrays are not bit-identical: ' + key + '/' + field)
                if args.geometry:
                    # Execute scalar inference in the original annotation order,
                    # but reduce in the legacy HDF query/repeat lexical order.
                    # First occurrence controls each distinct signature's position
                    # in the mean; even float32 summation order must be preserved.
                    for which, items in enumerate(collected):
                        collected[which] = {item: {key: values[key] for key in
                            sorted(values, key=first_signature_keys[which][item].get)}
                            for item, values in items.items()}
                    if set(collected[0]) != set(collected[1]) or set(collected[0]) != set(collected[2]):
                        raise AssertionError('Changed available grounding items')
                    mismatches = []
                    for key, values in collected[0].items():
                        other = collected[1][key]
                        third = collected[2][key]
                        mean_error = float(np.max(np.abs(np.mean(list(values.values()), 0) - np.mean(list(other.values()), 0))))
                        same_device_error = float(np.max(np.abs(np.mean(list(third.values()), 0) - np.mean(list(other.values()), 0))))
                        if len(values) != len(other) or len(third) != len(other) or mean_error > 3e-5 or same_device_error > 3e-5:
                            record = next(r for r in records if r['item_id'] == key)
                            mismatches.append({'item': key, 'kind': record['kind'], 'old_gpu': len(values),
                                'new': len(other), 'scalar_current_device': len(third),
                                'mean_error_vs_old_gpu': mean_error, 'mean_error_vs_scalar_current_device': same_device_error})
                    if mismatches:
                        from neurosym.io import save_json
                        output = root / 'artifacts/trace-geometry-diagnostic.json'
                        save_json(output, {'status': 'blocked', 'fit': path.name, 'source': source, 'queries': len(rows),
                            'device': args.device, 'mismatches': mismatches})
                        print(json.dumps({'status': 'blocked', 'report': str(output), 'mismatches': mismatches}), flush=True)
                        raise AssertionError('Grounding aggregation changed; inspect trace-geometry-diagnostic.json')
                    source_expectations[source] = (records, collected[2])
                result = {'fit': path.name, 'source': source, 'queries': len(rows), 'seconds': time.perf_counter() - started,
                    'compact_sample_bytes': compact_path.stat().st_size, 'compact_export_phase_seconds': export_phases,
                    'original_copied_sample_bytes': (path / 'traces-sample.h5').stat().st_size,
                    'storage_scope': 'Whole two-source sample per fit; old HDF group copying can expand hard links. Not a whole-study storage forecast.'}
                reports.append(result); print(json.dumps(result), flush=True)
            if args.geometry:
                from neurosym.trace_geometry import build_tables, VectorTable
                table_path = Path(temporary) / 'source-tables.h5'
                build_tables(temporary, table_path, [r for r, _ in source_expectations.values()])
                with h5py.File(table_path, 'r') as file:
                    for source, (records, expected) in source_expectations.items():
                        table = VectorTable(file, records)
                        if set(table) != {(source, key) for key in expected}:
                            raise AssertionError('Shared reducer changed available source/items')
                        for item, values in expected.items():
                            actual, reference = table[source, item], np.mean(list(values.values()), 0)
                            count = table.index[source, item][2]
                            if count != len(values) or not np.array_equal(actual, reference):
                                diagnostic = {'status': 'blocked', 'stage': 'shared geometry reducer',
                                    'fit': path.name, 'source': source, 'item': item, 'device': args.device,
                                    'expected_unique_signatures': len(values), 'actual_unique_signatures': count,
                                    'expected_shape': list(reference.shape), 'actual_shape': list(actual.shape),
                                    'max_absolute_error': float(np.max(np.abs(actual - reference)))
                                        if actual.shape == reference.shape else None,
                                    'reference_order': 'legacy lexical query_id, repeat; first occurrence of each exact signature'}
                                save_json(root / 'artifacts/trace-geometry-diagnostic.json', diagnostic)
                                raise AssertionError('Shared geometry reducer mismatch; inspect artifacts/trace-geometry-diagnostic.json')
    result = {'status': 'verified', 'complete_samples': args.limit is None, 'errors': errors, 'sources': reports,
        'geometry_verified': args.geometry, 'code': code_identity(), 'device': args.device,
        'wall_seconds': time.perf_counter() - wall_started,
        'checks': ['real weights/observations', 'complete query-local membership and legacy field ordering',
                   'parameter-free program reuse across two actual fitted seeds', 'committed fitted/prediction-stage restoration',
                   'source-table serialization and safe-point resume', 'same-device scalar bitwise equivalence',
                   'exact signature counts and legacy-ordered pooled/scoped source aggregation']}
    if args.device.startswith('cuda'):
        result.update(gpu=torch.cuda.get_device_name(args.device),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    save_json(receipt, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
