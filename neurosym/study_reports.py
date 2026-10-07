"""Derived measurements and final reports with explicit unavailable support."""
from collections import defaultdict, Counter
from pathlib import Path
import itertools

from .analysis_runs import run_directory, write_report
from .io import read_json, object_hash
from .runtime import analysis_lock, deadline


def parent_path(data, execution, identity):
    receipt = read_json(execution / (identity + '.json'))
    if receipt['status'] != 'complete' or len(receipt['results']) != 1:
        raise ValueError('A derived measurement requires one completed parent.')
    return data.root / receipt['results'][0]


def geometry_panel(data, options, execution, directory):
    from contextlib import ExitStack
    import h5py
    from .geometry import run_geometry, semantic_occurrences
    from .analysis_runs import partition
    from .trace_geometry import build_tables
    from .trace_store import read_json_array
    plan = read_json(data.root / 'configs/experiments.json')['geometry']
    base = {k: v for k, v in options.items() if k not in {'views', 'parent_job'}}
    if options.get('parent_job'):
        base['from_run'] = str(parent_path(data, execution, options['parent_job']))
    members, tables = [], None
    table_path = directory / 'grounding-source-tables.h5'
    with ExitStack() as stack:
        if 'grounding' in options['views']:
            split = partition(data.semantics, base['fold'])
            panels = [semantic_occurrences(data, split['test'], kind,
                reviewed_only=base.get('reviewed_only', False), scope_mode=scope)
                for kind, scope in itertools.product(plan['kinds'], plan['scope_modes']) if kind != 'configuration']
            build_tables(base['from_run'], table_path, panels,
                reserve_bytes=data.compute_config['storage']['minimum_free_gib'] * 2**30)
            tables = stack.enter_context(h5py.File(table_path, 'r'))
            write_report(directory / 'grounding-reduction.json', {'identity': tables.attrs['identity'],
                'sources': {source: {'items': read_json_array(group['items']), 'seconds': float(group.attrs['seconds'])}
                            for source, group in tables.items()},
                'item_reference': '[vector table, row, unique floating-value signature count]',
                'aggregation': 'Unchanged exact-value deduplication; lexical query/repeat/trace-field order retained.'})
        for view, kind, scope in itertools.product(options['views'], plan['kinds'], plan['scope_modes']):
            if view == 'grounding' and kind == 'configuration':
                continue
            deadline.check()
            result = run_geometry(data, {**base, 'view': view, 'kind': kind, 'scope_mode': scope,
                                         'min_stories': plan['minimum_stories_for_context_analysis'],
                                         'retain_signatures': view == 'grounding', 'retain_story_means': False},
                                  grounding_tables=tables if view == 'grounding' else None)
            receipt = read_json(result / 'complete.json')
            members.append({'view': view, 'kind': kind, 'scope_mode': scope,
                            'path': result.relative_to(data.root).as_posix(), 'available': receipt['geometry'].get('available', True)})
            deadline.advance()
    # Only regenerable vectors are removed after every panel is durable; raw
    # traces, parent weights, signatures, RDMs and reduction counts are retained.
    if tables is not None:
        table_path.unlink()
    return {'members': members}


def compare_panel(data, options, execution, directory):
    from .geometry import compare_geometry, grounding_stability
    plan = read_json(data.root / 'configs/experiments.json')['geometry']
    first = read_json(parent_path(data, execution, options['first_job']) / 'complete.json')['members']
    second = read_json(parent_path(data, execution, options['second_job']) / 'complete.json')['members']
    comparisons = []
    for a, b in itertools.product(first, second):
        if (a['kind'], a['scope_mode']) != (b['kind'], b['scope_mode']):
            continue
        if options['mode'] == 'maps' and (a['view'] != 'grounding' or b['view'] != 'grounding'):
            continue
        if a['view'] != b['view'] and b['view'] != 'cooccurrence' and not options.get('cross_views'):
            continue
        identity = {**options, 'first': a, 'second': b}
        path = directory / 'comparisons' / (object_hash(identity) + '.json')
        if path.exists():
            value = read_json(path)
        else:
            deadline.check()
            if not a['available'] or not b['available']:
                value = {'available': False, 'reason': 'Insufficient parent semantic support.'}
            else:
                fn = grounding_stability if options['mode'] == 'maps' else compare_geometry
                value = fn(data.root / a['path'], data.root / b['path'], permutations=plan['permutations'],
                           seed=11, device=options['device'])
            value = {**identity, 'result': value}
            write_report(path, value); deadline.advance()
        comparisons.append(value)
    return {'comparisons': comparisons}


def semantic_coverage(data, options, directory):
    """Count actual label support; this audit does not validate label correctness."""
    stories = data.semantics.splits['development' if options['phase'] == 'development' else 'heldout']
    items, families, rows = {}, defaultdict(lambda: {'queries': 0, 'sources': set(), 'stories': set()}), []
    for story in stories:
        sources = {s['id']: s for s in data.semantics.records(story, 'sources')}
        occurrences = list(data.semantics.records(story, 'occurrences'))
        examples = list(data.examples(story))
        rows.append({'story': story, 'sources': len(sources),
                     'timing_eligible_sources': sum(bool(s['decoder_timing_eligible']) for s in sources.values()),
                     'scorable_queries': len(examples), 'occurrences': len(occurrences),
                     'unresolved_grounding_occurrences': sum(r.get('grounding_resolved') is False for r in occurrences),
                     'kinds': dict(Counter(r['kind'] for r in occurrences)),
                     'review_status': dict(Counter(r['annotation_review_status'] for r in occurrences))})
        for e in examples:
            if not sources[e['source_id']]['decoder_timing_eligible']:
                continue
            entry = families[e['family']]
            entry['queries'] += 1; entry['sources'].add(e['source_id']); entry['stories'].add(story)
        for r in occurrences:
            for mode in ('pooled', 'scoped'):
                descriptor = {'kind': r['kind'], 'label': r['label']}
                if mode == 'scoped':
                    descriptor['scope'] = r.get('scope')
                key = (mode, object_hash(descriptor))
                entry = items.setdefault(key, {'scope_mode': mode, 'descriptor': descriptor, 'sources': set(), 'stories': set(), 'occurrences': 0})
                entry['occurrences'] += 1; entry['sources'].add(r['source_id']); entry['stories'].add(story)
    def serial(entry):
        return {k: sorted(v) if isinstance(v, set) else v for k, v in entry.items()}
    write_report(directory / 'semantic-items.json', [serial(v) for v in items.values()])
    return {'stories': rows, 'query_families': {k: serial(v) for k, v in families.items()},
            'semantic_items_file': 'semantic-items.json',
            'interpretation': 'Observed annotation support, including unresolved occurrences; per-view geometry separately records eligible support. Counts are not annotation accuracy, independent query samples, or evidence of systematic compositional generalization.'}


def study_report(data, job, manifest, execution, directory):
    from .geometry import crossed_interval, fdr_bh
    from .encoding import compare_encoding
    fits, geometry, comparisons, coverage = [], [], [], []
    for identity in job['dependencies']:
        path = parent_path(data, execution, identity)
        record = read_json(path / 'complete.json')
        kind = record['identity']['kind']
        if kind in {'decoder', 'encoding'}:
            fits.append((path, record))
        elif kind == 'geometry-panel':
            geometry.append(record)
        elif kind == 'compare-geometry-panel':
            comparisons.extend(record['comparisons'])
        elif kind == 'semantic-coverage':
            coverage.append({'path': str(path), 'report': record})
    # A complete human x story panel, after averaging fitted seed repetitions.
    # Each contrast's family/task scores remain available in the source reports.
    contrasts = defaultdict(list)
    decoders, decoder_paths = {}, {}
    priors = {}
    encodings = defaultdict(list)
    for path, r in fits:
        o = r['identity']['options']
        if r['identity']['kind'] == 'encoding':
            k = (o['subject'], o['fold'], o['comparison'], tuple(o.get('mask_models', [])))
            encodings[k].append((path, r))
        elif o['family'] == 'prior':
            priors[(o['fold'], o['seed'])] = r
        else:
            key = (o.get('subject'), o.get('model'), o.get('layer'), o['fold'], o['seed'], o['family'], o.get('retrain_null', False))
            decoders[key], decoder_paths[key] = r, path
    def paired(first, second):
        if not first.get('evaluation_support_hash') or first['evaluation_support_hash'] != second.get('evaluation_support_hash'):
            raise ValueError('Decoder contrasts require identical queries, targets and weights.')
        return second['matched']
    decoder_rows = []
    for key, r in decoders.items():
        subject, model, layer, fold, seed, family, null = key
        if fold.startswith('stories:'):
            continue
        references = [('test-mismatch', r['test_mismatch'])]
        prior = priors.get((fold, seed))
        if prior:
            references.append(('query-prior', paired(r, prior)))
        if family == 'structured' and not null:
            for other in ['linear', 'mlp']:
                reference = decoders.get((subject, model, layer, fold, seed, other, False))
                if reference:
                    references.append((other, paired(r, reference)))
            reference = decoders.get((subject, model, layer, fold, seed, family, True))
            if reference:
                references.append(('retrained-null', paired(r, reference)))
        faithfulness_path = decoder_paths[key] / 'faithfulness.json'
        if faithfulness_path.exists():
            faithfulness = read_json(faithfulness_path)
            references.extend((('selected-site-replacement', faithfulness['selected']),
                               ('random-site-replacement', faithfulness['control'])))
        else:
            faithfulness = None
        for label, baseline in references:
            for story, values in r['matched']['stories'].items():
                if story not in baseline['stories']:
                    raise ValueError('Paired decoder story support differs.')
                for metric in ['accuracy_minus_chance', 'nll']:
                    difference = values[metric] - baseline['stories'][story][metric]
                    row = {'subject': subject, 'model': model, 'layer': layer, 'story': story, 'seed': seed,
                           'family': family, 'null': null, 'reference': label, 'metric': metric, 'difference': difference}
                    decoder_rows.append(row)
                    if subject:
                        contrasts[('decoder', family, null, label, metric)].append(row)
        if faithfulness:
            for story, values in faithfulness['selected']['stories'].items():
                for metric in ['accuracy_minus_chance', 'nll']:
                    row = {'subject': subject, 'model': model, 'layer': layer, 'story': story, 'seed': seed,
                           'family': family, 'null': null, 'reference': 'selected-minus-random-replacement', 'metric': metric,
                           'difference': values[metric] - faithfulness['control']['stories'][story][metric]}
                    decoder_rows.append(row)
                    if subject:
                        contrasts[('faithfulness', metric)].append(row)
    encoding_rows = []
    for key, group in encodings.items():
        if key[1].startswith('stories:'):
            continue
        group.sort(key=lambda pair: len(pair[1]['identity']['options']['groups']))
        for (ap, a), (bp, b) in zip(group, group[1:]):
            deadline.check()
            report_path = directory / 'encoding-comparisons' / (object_hash([a['identity'], b['identity']]) + '.json')
            if not report_path.exists():
                write_report(report_path, compare_encoding(ap, bp)); deadline.advance()
            result = read_json(report_path)
            for story, values in result['stories'].items():
                row = {'subject': key[0], 'story': story, 'comparison': key[2], 'mask_models': key[3],
                       'baseline': a['identity']['options']['groups'], 'augmented': b['identity']['options']['groups'],
                       'difference': values['mean_voxel_delta_r'], 'finite_paired_voxels': values['finite_paired_voxels']}
                encoding_rows.append(row)
                if row['difference'] is not None:
                    contrasts[('encoding', key[2], key[3], tuple(row['baseline']), tuple(row['augmented']))].append(row)
    intervals = []
    for key, rows in contrasts.items():
        deadline.check()
        cached = directory / 'intervals' / (object_hash({'key': key, 'rows': rows}) + '.json')
        if cached.exists():
            intervals.append(read_json(cached))
            continue
        subjects, stories = {r['subject'] for r in rows}, {r['story'] for r in rows}
        if len(subjects) >= 2 and len(stories) >= 2 and len({(r['subject'], r['story']) for r in rows}) == len(subjects) * len(stories):
            result = crossed_interval(rows)
        else:
            result = {'available': False, 'reason': 'Not a complete multi-participant multi-story panel.',
                      'participants': len(subjects), 'stories': len(stories)}
        value = {'contrast': key, 'result': result}
        write_report(cached, value); deadline.advance()
        intervals.append(value)
    # Declare the family rather than silently selecting significant labels.
    families = defaultdict(list)
    for i, row in enumerate(comparisons):
        result = row['result']
        p = result.get('two_sided_label_permutation_p', result.get('semantic_label_permutation_p'))
        if p is not None:
            family = (row['purpose'], row['first']['view'], row['second']['view'], row['first']['scope_mode'])
            families[family].append((i, p))
    for family, members in families.items():
        for (i, _), q in zip(members, fdr_bh([p for _, p in members]), strict=True):
            comparisons[i]['bh_family'] = family
            comparisons[i]['bh_adjusted_p'] = float(q)
    write_report(directory / 'decoder-effects.json', decoder_rows)
    write_report(directory / 'encoding-effects.json', encoding_rows)
    write_report(directory / 'participant-story-intervals.json', intervals)
    write_report(directory / 'geometry-comparisons.json', comparisons)
    write_report(directory / 'geometry-coverage.json', [r['members'] for r in geometry])
    write_report(directory / 'semantic-coverage.json', coverage)
    return {'manifest': manifest['content_hash'], 'fits': len(fits), 'geometry_panels': len(geometry),
            'geometry_comparisons': len(comparisons), 'intervals': len(intervals),
            'reports': ['decoder-effects.json', 'encoding-effects.json', 'participant-story-intervals.json',
                        'geometry-comparisons.json', 'geometry-coverage.json', 'semantic-coverage.json'],
            'interpretation': 'Effect sizes and crossed participant/story intervals; geometry permutation tests are conditional on supported items, with BH across each stated purpose/view/scope family. Final single-story intervals are unavailable. No causal or systematic-composition conclusion is automatic.'}


def run_derived(data, job, manifest, execution):
    options = job['options']
    # Logical IDs alone do not pin the actual fitted parent receipts.
    parents = {key: object_hash(read_json(parent_path(data, execution, key) / 'complete.json')['identity'])
               for key in job['dependencies']}
    directory, identity = run_directory(data, job['kind'],
        {**options, 'execution_manifest_hash': manifest['content_hash'], 'parent_identities': parents})
    with analysis_lock(directory / 'RUNNING.lock'):
        if (directory / 'complete.json').exists():
            return directory
        if job['kind'] == 'geometry-panel':
            report = geometry_panel(data, options, execution, directory)
        elif job['kind'] == 'compare-geometry-panel':
            report = compare_panel(data, options, execution, directory)
        elif job['kind'] == 'study-report':
            report = study_report(data, job, manifest, execution, directory)
        elif job['kind'] == 'semantic-coverage':
            report = semantic_coverage(data, options, directory)
        else:
            raise ValueError('Unknown derived measurement: ' + job['kind'])
        write_report(directory / 'complete.json', {'identity': identity, **report})
    return directory
