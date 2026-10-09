"""Validate administrative scheduling and summaries using real study artifacts."""
from collections import Counter
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_plan import (MANIFEST, choose_wave, digest, load_manifest, make_plan,
                             milestone_progress, read, write)
from report_production import decoder_reports, job_key, score_differences


def validate(root):
    m = load_manifest(root / 'artifacts/execution' / MANIFEST / 'manifest.json')
    plan = make_plan(m)
    assert plan['early_panel']['workers'] == 294
    assert plan['early_panel']['logical_items'] == 1050
    assert len(plan['workers']) == 4033 and len(m['jobs']) == 10285
    milestone = plan['milestone']
    selected_ids = set(milestone['job_ids'])
    selected_workers = set(milestone['worker_indices'])
    assert milestone['workers'] == 810 and milestone['logical_items'] == 1602
    assert milestone['counts'] == {'prepare-decoder': 38, 'select-decoder': 96, 'decoder': 372,
                                    'encoding': 612, 'geometry-panel': 106,
                                    'compare-geometry-panel': 377, 'semantic-coverage': 1}
    prior_plan = read(root / 'artifacts/first-day-runtime-plan.json')['panels']['recommended_first_day']
    assert selected_workers == set(prior_plan['worker_indices']), 'Selection drifted from the authorized proposal'
    for w in selected_workers:
        worker = m['workers'][w]
        assert set(worker['dependencies']) <= selected_ids
        assert all(m['jobs'][i]['id'] in selected_ids for i in worker['indices'])
    assert all(j['kind'] != 'study-report' for j in m['jobs'] if j['id'] in selected_ids)
    for name in ('production_plan.py', 'submit_production.py', 'report_production.py', 'verify_production.py',
                 'package_production.py', 'install_production_packet.py'):
        ast.parse((root / 'scripts' / name).read_text(encoding='utf-8'), feature_version=(3, 11))
    # Dry scheduling over the actual dependency graph. No observation, fit or
    # scientific score is synthesized, and no scheduler command is invoked.
    totals = {}
    for cap in (1, 4, 6):
        completed, seen, admitted = set(), set(), Counter()
        waves = 0
        while seen != selected_workers:
            selected = choose_wave(m, plan, completed, set(), {}, {}, max_gpus=cap, admitted=admitted)
            assert selected, 'Deadlocked real dependency graph'
            assert not set(selected) & seen
            assert set(selected) <= selected_workers
            resources = Counter(m['workers'][w]['resource'] for w in selected)
            assert sum(n * m['resources']['resources'][r]['gpus'] for r, n in resources.items()) <= cap
            for resource, number in resources.items():
                assert number <= m['resources']['resources'][resource]['concurrency']
            for w in selected:
                assert set(m['workers'][w]['dependencies']) <= completed
            for w in selected:
                completed.update(m['jobs'][i]['id'] for i in m['workers'][w]['indices'])
                seen.add(w)
                admitted[plan['workers'][w]['lane']] += 1
            waves += 1
        assert completed == selected_ids
        assert choose_wave(m, plan, completed, set(), {}, {}, max_gpus=cap) == [], 'Expanded beyond milestone'
        totals[cap] = {'workers': len(seen), 'logical_items': len(completed), 'metadata_waves': waves}

    from submit_production import controller_action, submit_wave, successor_command
    assert controller_action(plan, selected_ids, False, True) == 'complete'
    assert controller_action(plan, selected_ids, True, True) == 'complete'
    one_missing = selected_ids - {next(iter(selected_ids))}
    assert controller_action(plan, one_missing, True, True) == 'handoff'
    assert controller_action(plan, one_missing, True, False) == 'stop'
    assert controller_action(plan, one_missing, False, True) == 'dispatch'
    previous_packet = root / 'artifacts/production/2e0ac8b23d2a005ce9b88598c05d8d19e61d38a12a36a968d31a22e751b6683c'
    continuation = successor_command(root, previous_packet, 6, '1027340')
    assert continuation[-3:] == [str(previous_packet / 'run_production.sbatch'), str(previous_packet), '6']
    assert '--dependency=afterany:1027340' in continuation
    excluded = next(w for w in range(len(m['workers'])) if w not in selected_workers)
    try:
        submit_wave(root, m, plan, [excluded], root, {}, root, 'not-used')
    except ValueError as exc:
        assert 'escape' in str(exc)
    else:
        raise AssertionError('Submission accepted an excluded actual worker')

    # Real downloaded inventory: archive evidence is useful immediately but the
    # executor still obtains its own qualified adoption receipts before stopping.
    with zipfile.ZipFile(root / 'artifacts/production-evidence-latest.zip') as archive:
        inventory = json.loads(archive.read('inventory.json'))['data']
    available = {r['job'] for r in inventory if r['status'] != 'not completed'}
    current = {r['job'] for r in inventory if r['status'] == 'current-execution'}
    progress = milestone_progress(plan, available, current)
    assert progress['available_including_qualified_reuse'] == 82
    assert not progress['execution_complete'] and not progress['evidence_complete']
    partial = milestone_progress(plan, selected_ids, current)
    assert partial['evidence_complete'] and not partial['execution_complete']
    final = milestone_progress(plan, selected_ids, selected_ids)
    assert final['execution_complete'] and final['evidence_complete']
    # A lowered cap drains existing work; it must not admit more GPU tasks.
    wave = choose_wave(m, plan, set(), set(), {'gpu-decoder': 6}, {'prediction': 6}, max_gpus=2)
    assert all(m['resources']['resources'][m['workers'][w]['resource']]['gpus'] == 0 for w in wave)
    wave = choose_wave(m, plan, set(), set(), {}, {}, other_gpu=5, max_gpus=6)
    assert sum(m['resources']['resources'][m['workers'][w]['resource']]['gpus'] for w in wave) <= 1
    try:
        choose_wave(m, plan, set(), set(), {}, {}, max_gpus=7)
    except ValueError:
        pass
    else:
        raise AssertionError('Unauthorized GPU cap accepted')
    # All three GPU branches with no fit dependencies receive initial capacity.
    first = choose_wave(m, plan, set(), set(), {}, {}, max_gpus=6)
    assert {'prediction', 'context', 'encoding', 'geometry'} <= {plan['workers'][w]['lane'] for w in first}
    # Scheduling priorities must not leak a descriptive depth fit into the first
    # ready prediction selection when primary early work remains ready.
    setup = {j['id'] for j in m['jobs'] if j['kind'] in {'prepare-decoder', 'select-decoder'}}
    ready = choose_wave(m, plan, setup, set(), {}, {}, max_gpus=6)
    assert all(plan['workers'][w]['early_panel'] for w in ready if plan['workers'][w]['lane'] == 'prediction')

    audit = root / 'artifacts/trace-audit-1024054'
    records, known_scores = {}, {}
    for path in sorted((audit / 'fits').glob('*/complete.json')):
        r = read(path)
        key = job_key('decoder', r['identity']['options'])
        j = next(j for j in m['jobs'] if job_key(j['kind'], j['options']) == key)
        records[j['id']] = (path.parent, r, 'actual archived audit')
        known_scores[j['id']] = r
    assert len(records) == 2
    with tempfile.TemporaryDirectory(prefix='production-validation-', dir=root / 'artifacts') as temporary:
        report = decoder_reports(audit, m, records, Path(temporary))
        assert len(report['scores']) == 2
        for score in report['scores']:
            r = known_scores[score['job']]
            assert score['matched'] == r['matched'] and score['test_mismatch'] == r['test_mismatch']
            assert sum(v['queries'] for v in score['support']['rows']) == r['matched']['n_queries']
        for effect in report['effects']:
            assert effect['comparison'] == 'test-mismatch', 'Ordinary baselines incorrectly paired to context fits'
            r = known_scores[effect['job']]
            for row in effect['rows']:
                assert row['available']
                a = (r['matched']['stories'] if row['task'] == 'all_tasks' else r['matched']['family_stories'][row['task']])[row['story']]
                b = (r['test_mismatch']['stories'] if row['task'] == 'all_tasks' else r['test_mismatch']['family_stories'][row['task']])[row['story']]
                assert row['matched_minus_reference']['nll'] == a['nll'] - b['nll']
                assert row['matched_minus_reference']['correct'] == a['correct'] - b['correct']
        assert len(report['unavailable_comparisons']) == 2
        # Fault injection into copies of actual reports: missing support stays
        # unavailable, not a fabricated zero-valued effect.
        r = next(iter(known_scores.values()))
        damaged = copy.deepcopy(r['test_mismatch'])
        del damaged['stories']['story_01']
        rows = score_differences(r['matched'], damaged)
        assert any(v['task'] == 'all_tasks' and v['story'] == 'story_01' and not v['available'] for v in rows)

    from submit_production import reconcile
    # Evaluate clean-yield handling against real archived attempt/status pairs.
    audit_ledger = read(audit / 'dispatch.json')
    successful = 0
    import submit_production as controller
    original = controller.receipt
    try:
        controller.receipt = lambda *args: False
        for a in audit_ledger['attempts']:
            status_path = audit / 'execution/attempts' / (str(a['worker']) + '.json')
            if not status_path.exists():
                continue
            status = read(status_path)
            if status.get('status') != 'yielded' or status.get('attempt') != a['token']:
                continue
            attempt = {**a, 'state': 'submitted'}
            ledger = {'attempts': [attempt]}
            _, _, _, failures = reconcile(audit, m, ledger, set(), {a['task_id']: ('FAILED', '75:0')}, set(), audit / 'execution')
            assert not failures and attempt['state'] == 'yielded'
            bad = {**a, 'state': 'submitted', 'token': a['token'] + '-changed'}
            _, _, _, failures = reconcile(audit, m, {'attempts': [bad]}, set(), {a['task_id']: ('FAILED', '75:0')}, set(), audit / 'execution')
            assert failures and bad['state'] == 'failed'
            successful += 1
    finally:
        controller.receipt = original
    source_files = ('production_plan.py', 'submit_production.py', 'report_production.py', 'verify_production.py',
                    'package_production.py', 'install_production_packet.py', 'run_production.sbatch', 'start_production.sbatch')
    result = {'status': 'verified', 'manifest_hash': MANIFEST, 'plan_sha256': digest(plan),
              'verified_source_files': {n: hashlib.sha256((root / 'scripts' / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in source_files},
              'default_max_active_gpus': 6, 'early_panel': plan['early_panel'],
              'milestone': {k: v for k, v in milestone.items() if k not in {'job_ids', 'worker_indices'}},
              'complete_real_selection_scheduling': totals,
              'milestone_stop_and_out_of_scope_rejection_verified': True,
              'continuation_pins_same_packet_and_gpu_cap': True,
              'actual_downloaded_evidence_progress': {k: v for k, v in progress.items() if not k.startswith('missing_')},
              'actual_decoder_summaries_verified': len(records),
              'actual_yield_records_verified': successful,
              'limitations': ['No cluster command or scientific fit executed.',
                              'Full encoding arrays remain remote; their unchanged installed comparator passed inspection 1027340.',
                              'Scheduler metadata simulation establishes inventory/cap/dependency behavior, not elapsed runtime or Slurm availability.']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = validate(args.root.resolve())
    write(args.root / 'artifacts/production-verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
