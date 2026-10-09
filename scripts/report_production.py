"""Read completed scientific outputs; publish descriptive interim reports only.

No fitting, trace processing, permutation testing or final-story evaluation.
Encoding differences call the installed, support-checking comparison routine.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_plan import (CODE_ROOT, MANIFEST, digest, inside, load_manifest,
                             make_plan, milestone_progress, read, write)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_key(kind, options):
    if kind in {'select-decoder', 'decoder-selection'}:
        return ('selection', tuple((k, options.get(k)) for k in
                ('modality', 'subject', 'model', 'layer', 'fold', 'family')))
    if kind == 'encoding':
        options = {k: v for k, v in options.items() if k != 'comparison_support'}
    return (kind, digest(options))


def collect_records(root, manifest):
    """Execution receipts first; qualified archived fits remain visibly archived."""
    execution = root / manifest['analysis_config']['output'] / 'execution' / MANIFEST
    records, completed = {}, set()
    by_key = {job_key(j['kind'], j['options']): j['id'] for j in manifest['jobs']}
    for j in manifest['jobs']:
        path = execution / (j['id'] + '.json')
        if not path.exists():
            continue
        receipt = read(path)
        if (receipt.get('status') != 'complete' or receipt.get('manifest_hash') != MANIFEST or
                receipt.get('job_id') != j['id'] or len(receipt.get('results', [])) != 1):
            raise ValueError('Invalid completed execution receipt: ' + str(path))
        result = inside(root, receipt['results'][0])
        record = read(result / 'complete.json')
        records[j['id']] = (result, record, 'current-execution')
        completed.add(j['id'])
    registry_path = root / 'artifacts/fit-reuse.json'
    if registry_path.exists():
        registry = read(registry_path)
        if not registry.get('applied') or any(manifest['code'].get(k) != v for k, v in registry['target_code'].items()):
            raise ValueError('Reuse registry is not qualified for this installation.')
        for target, entry in registry['completed'].items():
            path = inside(root, entry['path'])
            if file_hash(path / 'complete.json') != entry['complete_sha256']:
                raise ValueError('Qualified archived result changed: ' + str(path))
            r = read(path / 'complete.json')
            identity = r['identity']
            if (digest({**identity, 'code': registry['target_code']}) != target or
                    identity['semantic_build_hash'] != manifest['semantic_build_hash']):
                raise ValueError('Reuse identity mismatch.')
            key = job_key(identity['kind'], identity['options'])
            job = by_key.get(key)
            if job and job not in records:
                records[job] = (path, r, 'qualified-archive; execution receipt pending')
    return records, completed


def score_differences(first, second):
    """Use the stored weighting exactly; never average queries anew."""
    rows = []
    panels = [('all_tasks', first.get('stories', {}), second.get('stories', {}))]
    for family in sorted(set(first.get('family_stories', {})) | set(second.get('family_stories', {}))):
        panels.append((family, first.get('family_stories', {}).get(family, {}),
                       second.get('family_stories', {}).get(family, {})))
    for family, a, b in panels:
        for story in sorted(set(a) | set(b)):
            if story not in a or story not in b:
                rows.append({'task': family, 'story': story, 'available': False, 'reason': 'Unmatched task/story support.'})
                continue
            row = {'task': family, 'story': story, 'available': True,
                   'matched': a[story], 'reference': b[story]}
            row['matched_minus_reference'] = {k: a[story][k] - b[story][k]
                                              for k in ('correct', 'chance', 'accuracy_minus_chance', 'nll')}
            rows.append(row)
    return rows


def prediction_support(path, record, cache):
    key = record.get('evaluation_support_hash') or file_hash(path / 'complete.json')
    destination = cache / 'support' / (key + '.json')
    if destination.exists():
        return read(destination)
    files = record.get('prediction_files', [])
    # This is a fixed production filename; mismatch predictions are not counted
    # as additional observed queries. Raw probabilities are never copied.
    source = path / 'predictions.jsonl.gz'
    if not source.exists():
        return {'available': False, 'reason': 'Prediction support file missing.', 'declared_files': files}
    groups = defaultdict(lambda: {'queries': set(), 'sources': set(), 'repeat_predictions': 0})
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            group = groups[(r['story_id'], r['family'])]
            group['queries'].add(r['query_id'])
            group['sources'].add(r['source_id'])
            group['repeat_predictions'] += 1
    result = {'available': True, 'evaluation_support_hash': record.get('evaluation_support_hash'),
              'rows': [{'story': k[0], 'task': k[1], 'queries': len(v['queries']),
                        'sources': len(v['sources']), 'repeat_predictions': v['repeat_predictions']}
                       for k, v in sorted(groups.items())]}
    write(destination, result)
    return result


def decoder_reports(root, manifest, records, cache):
    decoders, priors, selections = {}, {}, []
    scores, effects, missing = [], [], []
    for j in manifest['jobs']:
        if j['id'] not in records:
            continue
        path, r, provenance = records[j['id']]
        o = j['options']
        if j['kind'] == 'select-decoder':
            selections.append({'job': j['id'], 'options': o, 'provenance': provenance,
                'path': str(path.relative_to(root)), 'selected': {k: v for k, v in r.items() if k != 'identity'},
                'inner_histories': {p.name: read(p) for p in sorted(path.glob('inner-*.json'))}})
        if j['kind'] != 'decoder':
            continue
        decoders[job_key('decoder', o)] = (j, path, r)
        if o['family'] == 'prior':
            priors[(o['fold'], o['seed'])] = (j, path, r)
        scores.append({'job': j['id'], 'options': o, 'provenance': provenance,
                       'path': str(path.relative_to(root)), 'partition': r['partition'],
                       'matched': r['matched'], 'test_mismatch': r['test_mismatch'],
                       'evaluation_support_hash': r.get('evaluation_support_hash'),
                       'support': prediction_support(path, r, cache),
                       'selection': read(path / 'selection.json') if (path / 'selection.json').exists() else None})

    def lookup(o, family, null=False):
        matches = [value for value in decoders.values() if
                   all(value[0]['options'].get(k) == o.get(k) for k in ('modality', 'subject', 'model', 'layer', 'fold', 'seed'))
                   and value[0]['options']['family'] == family and bool(value[0]['options'].get('retrain_null')) == null]
        if len(matches) > 1:
            raise ValueError('Ambiguous decoder comparator.')
        return matches[0] if matches else None

    for j, path, r in decoders.values():
        o = j['options']
        if o['family'] == 'prior':
            continue
        refs = [('test-mismatch', r['test_mismatch'], None)]
        context = o['fold'].startswith('stories:')
        if not context:
            targets = [('query-prior', priors.get((o['fold'], o['seed'])))]
            if o['family'] == 'structured' and not o.get('retrain_null'):
                targets += [('linear', lookup(o, 'linear')), ('mlp', lookup(o, 'mlp')),
                            ('retrained-null', lookup(o, 'structured', True))]
            for label, other in targets:
                if other is None:
                    missing.append({'job': j['id'], 'reference': label, 'reason': 'Comparator not completed.'})
                elif not r.get('evaluation_support_hash') or r['evaluation_support_hash'] != other[2].get('evaluation_support_hash'):
                    missing.append({'job': j['id'], 'reference': label, 'reason': 'Query/target/weight support differs; no subtraction.'})
                else:
                    refs.append((label, other[2]['matched'], other[0]['id']))
        else:
            missing.append({'job': j['id'], 'reference': 'ordinary-fold controls',
                            'reason': 'Not applicable: context fits have different training/test partitions.'})
        faithfulness = path / 'faithfulness.json'
        if faithfulness.exists():
            f = read(faithfulness)
            refs += [('selected-site-replacement', f['selected'], None), ('random-site-replacement', f['control'], None)]
            effects.append({'job': j['id'], 'options': o, 'comparison': 'selected-minus-random-replacement',
                            'rows': score_differences(f['selected'], f['control'])})
        elif o.get('faithfulness'):
            missing.append({'job': j['id'], 'reference': 'faithfulness', 'reason': 'Expected faithfulness output missing.'})
        for label, ref, ref_job in refs:
            effects.append({'job': j['id'], 'options': o, 'comparison': label, 'reference_job': ref_job,
                            'rows': score_differences(r['matched'], ref)})
    return {'scores': scores, 'effects': effects, 'unavailable_comparisons': missing, 'selection': selections}


def encoding_reports(root, manifest, records, cache, code):
    sys.path.insert(0, str(code))
    from neurosym.encoding import compare_encoding
    groups, summaries = defaultdict(list), []
    for j in manifest['jobs']:
        if j['kind'] != 'encoding':
            continue
        o = j['options']
        if o['fold'].startswith('stories:'):
            continue
        key = (o['subject'], o['fold'], o['comparison'], tuple(o.get('mask_models', [])))
        groups[key].append(j)
    for key, jobs in groups.items():
        jobs.sort(key=lambda j: len(j['options']['groups']))
        # Pair declared adjacent conditions, never bridge a missing middle model condition.
        for a, b in zip(jobs, jobs[1:]):
            row = {'subject': key[0], 'fold': key[1], 'comparison': key[2], 'mask_models': key[3],
                   'baseline_job': a['id'], 'augmented_job': b['id'],
                   'baseline': a['options']['groups'], 'augmented': b['options']['groups']}
            if a['id'] not in records or b['id'] not in records:
                row.update(available=False, reason='Both declared conditions are not yet completed.')
            else:
                ap, ar, _ = records[a['id']]
                bp, br, _ = records[b['id']]
                dest = cache / 'encoding' / (digest([ar['identity'], br['identity']]) + '.json')
                if not dest.exists():
                    # The installed routine enforces fitted contracts and exact
                    # training/test response rows, including qualified old fits.
                    write(dest, compare_encoding(ap, bp))
                row.update(available=True, result=read(dest))
            summaries.append(row)
    return summaries


def geometry_reports(root, manifest, records):
    panels, comparisons, coverage = [], [], []
    for j in manifest['jobs']:
        if j['id'] not in records:
            continue
        path, r, provenance = records[j['id']]
        if j['kind'] == 'geometry-panel':
            members = []
            for member in r['members']:
                meta_path = inside(root, member['path']) / 'geometry.json'
                meta = read(meta_path)
                members.append({**member, 'support': meta})
            panels.append({'job': j['id'], 'options': j['options'], 'members': members})
        elif j['kind'] == 'compare-geometry-panel':
            for c in r['comparisons']:
                result = c['result']
                # Scalars and support counts suffice for the interim index;
                # complete item/site lists remain in the original result.
                summary = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
                summary.update({k + '_count': len(v) for k, v in result.items() if isinstance(v, list)})
                comparisons.append({'job': j['id'], 'path': str(path.relative_to(root)),
                    'purpose': c['purpose'], 'mode': c['mode'], 'first': c['first'], 'second': c['second'],
                    'result': summary, 'multiplicity': 'Unadjusted descriptive snapshot; final declared BH families unchanged.'})
        elif j['kind'] == 'semantic-coverage':
            coverage.append({'job': j['id'], 'path': str(path.relative_to(root)), 'report': r})
    return {'panels': panels, 'comparisons': comparisons, 'semantic_coverage': coverage}


def report(root, manifest_path, output):
    manifest = load_manifest(manifest_path)
    plan = make_plan(manifest)
    records, completed = collect_records(root, manifest)
    # A new administrative selection or newly adopted execution receipt must
    # refresh counts even when the underlying fitted scientific outputs match.
    generation = digest({'plan': digest(plan), 'completed': sorted(completed),
                         'records': {k: file_hash(v[0] / 'complete.json') for k, v in records.items()}})
    previous = output / 'complete.json'
    if previous.exists() and read(previous).get('generation') == generation:
        print('INTERIM UNCHANGED', len(records), flush=True)
        return
    code = inside(root, CODE_ROOT)
    cache = output.parent / 'report-cache'
    decoder = decoder_reports(root, manifest, records, cache)
    encoding = encoding_reports(root, manifest, records, cache, code)
    geometry = geometry_reports(root, manifest, records)
    inventory = []
    early_ids = {manifest['jobs'][i]['id'] for row in plan['workers'] if row['early_panel']
                 for i in manifest['workers'][row['worker']]['indices']}
    milestone_ids = set(plan['milestone']['job_ids'])
    for j in manifest['jobs']:
        inventory.append({'job': j['id'], 'kind': j['kind'], 'options': j['options'],
            'early_panel': j['id'] in early_ids, 'first_day_milestone': j['id'] in milestone_ids,
            'status': records[j['id']][2] if j['id'] in records else 'not completed',
            'path': str(records[j['id']][0].relative_to(root)) if j['id'] in records else None})
    expected = Counter(j['kind'] for j in manifest['jobs'])
    available = Counter(j['kind'] for j in manifest['jobs'] if j['id'] in records)
    missing_early = sorted(early_ids - set(records))
    # Explicit complete observation/fold panels; partial seed runs are not labelled complete.
    panel_jobs = defaultdict(list)
    for j in manifest['jobs']:
        o = j['options']
        if j['kind'] == 'decoder' and o['family'] != 'prior' and o['fold'] in plan['early_folds']:
            if o.get('modality') == 'model' and o.get('layer') != plan['primary_layers'].get(o.get('model')):
                continue
            panel_jobs[(o.get('subject', o.get('model')), o['fold'])].append(j['id'])
    panels = []
    for (system, fold), ids in panel_jobs.items():
        ids += [j['id'] for j in manifest['jobs'] if j['kind'] == 'decoder' and
                j['options']['family'] == 'prior' and j['options']['fold'] == fold]
        panels.append({'system': system, 'fold': fold, 'expected_fits': len(ids),
                       'available_fits': sum(i in records for i in ids), 'complete': all(i in records for i in ids)})
    summary = {'manifest': MANIFEST, 'generation': generation,
        'plan_sha256': digest(plan), 'milestone': milestone_progress(plan, records, completed),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'current_execution_completed': len(completed), 'available_including_qualified_reuse': len(records),
        'expected_logical_items': len(manifest['jobs']), 'expected_by_kind': dict(expected), 'available_by_kind': dict(available),
        'early_panel_expected': len(early_ids), 'early_panel_available': len(early_ids) - len(missing_early),
        'early_panel_complete': not missing_early, 'missing_early_job_ids': missing_early,
        'decoder_panels': panels,
        'interpretation': 'Interim descriptive development evidence. Missing is not zero; seeds are not participants. No population intervals, significance declarations, partial-family BH or final-story evaluation. Context and ordinary fits remain separate.'}
    # One replaceable latest snapshot, bounded caches and a small history index.
    # All files identify their generation; complete.json is published last.
    for name, value in [('inventory', inventory), ('decoder', decoder), ('encoding', encoding), ('geometry', geometry)]:
        write(output / (name + '.json'), {'generation': generation, 'data': value})
    lines = ['# Interim production evidence', '', summary['interpretation'], '',
             f"First-day milestone: {summary['milestone']['available_including_qualified_reuse']}/{len(milestone_ids)} items available; {summary['milestone']['current_execution_completed']} current execution receipts; {plan['milestone']['workers']} selected workers.",
             'The controller stops after this milestone. Remaining development work stays deferred; story 11 stays reserved.',
             f"Full inventory available: {len(records)}/{len(manifest['jobs'])} logical items; current execution receipts: {len(completed)}.",
             f"Early panel: {summary['early_panel_available']}/{len(early_ids)} items; complete decoder system/fold panels: {sum(p['complete'] for p in panels)}/{len(panels)}.", '',
             '| Kind | Available | Expected |', '|---|---:|---:|']
    lines += [f'| {k} | {available[k]} | {expected[k]} |' for k in sorted(expected)]
    lines += ['', 'Detailed files: decoder.json (task/story effects, support, selection), encoding.json (paired supported differences), geometry.json (support and raw comparisons), inventory.json (all expected conditions).',
              '', 'Matched-minus-reference accuracy: positive favors matched. Matched-minus-reference NLL: negative favors matched.',
              'Context fits have no ordinary-fold comparator substitution. Current anatomical interpretations require observation dependence as well as stability.',
              '', '## Completed matched versus mismatch aggregates', '',
              '| System | Fold | Seed | Readout | Null training | Accuracy difference (pp) | NLL difference |',
              '|---|---|---:|---|---|---:|---:|']
    for s in decoder['scores']:
        o = s['options']
        if o['family'] == 'prior':
            continue
        a, b = s['matched']['story_macro'], s['test_mismatch']['story_macro']
        system = o.get('subject', o.get('model', 'prior'))
        if o.get('modality') == 'model':
            system += ':' + str(o['layer'])
        lines.append(f"| {system} | {o['fold']} | {o['seed']} | {o['family']} | {bool(o.get('retrain_null'))} | {(a['correct']-b['correct'])*100:.4f} | {a['nll']-b['nll']:.6f} |")
    temporary = output / 'summary.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(output / 'summary.md')
    with (output.parent / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**{k: summary[k] for k in ('generation', 'created_utc', 'current_execution_completed', 'early_panel_available')},
                                'milestone_available': summary['milestone']['available_including_qualified_reuse'],
                                'milestone_selection': plan['milestone']['selection_sha256']}) + '\n')
    write(previous, summary)
    print('INTERIM UPDATED', summary['early_panel_available'], '/', len(early_ids), 'early items;', len(records), 'available;', previous, flush=True)
    print('MILESTONE EVIDENCE', summary['milestone']['available_including_qualified_reuse'], '/', len(milestone_ids),
          'execution_receipts=', summary['milestone']['current_execution_completed'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import os
    import socket
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Scientific interim summaries run on allocated compute only.')
    root = args.root.resolve()
    from submit_production import verify_installation
    verify_installation(root, load_manifest(args.manifest))
    report(root, args.manifest, root / 'artifacts/production-reports' / MANIFEST / 'latest')


if __name__ == '__main__':
    main()
