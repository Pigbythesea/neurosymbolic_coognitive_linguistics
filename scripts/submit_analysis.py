"""Dispatch ready workers only; dry run unless --submit, on your allocated node.

--watch continues dispatching; otherwise one ready wave is submitted. --resume
reopens the ledger. Only clean exit-75 yields are automatically continued.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid


def submission_groups(manifest):
    if manifest.get('format_version') != 2:
        raise ValueError('Version 2 worker manifest required.')
    profiles, groups = manifest['resources']['resources'], {}
    size = manifest['resources']['array_chunk']
    if not isinstance(size, int) or not 1 <= size <= 500:
        raise ValueError('Array chunk must be 1..500.')
    for index, worker in enumerate(manifest['workers']):
        resource = worker['resource']
        p = profiles[resource]
        if any(p[k] < 1 for k in ('cpus', 'concurrency')) or p['gpus'] not in (0, 1):
            raise ValueError('Invalid resource profile.')
        if set(p['partition'].split(',')) & set(manifest['resources']['documentation']['excluded_by_user']):
            raise ValueError('An excluded GPU partition was requested.')
        groups.setdefault((resource, tuple(worker['dependencies'])), []).append(index)
    stages = []
    for (resource, deps), members in groups.items():
        for offset in range(0, len(members), size):
            workers = members[offset:offset + size]
            stages.append({'resource': resource, 'dependencies': list(deps), 'workers': workers,
                'indices': [i for w in workers for i in manifest['workers'][w]['indices']], 'profile': profiles[resource]})
    if sorted(i for s in stages for i in s['indices']) != list(range(len(manifest['jobs']))):
        raise ValueError('Worker inventory omits or duplicates logical jobs.')
    return stages


def seconds(value):
    parts = value.split(':')
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise ValueError('Walltime must be HH:MM:SS.')
    h, m, s = map(int, parts)
    if m >= 60 or s >= 60 or h * 3600 + m * 60 + s < 300:
        raise ValueError('Walltime needs valid fields and at least five minutes.')
    return h * 3600 + m * 60 + s


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def ready_workers(manifest, completed, occupied, allowed):
    ready = defaultdict(list)
    for index, worker in enumerate(manifest['workers']):
        if index not in allowed or index in occupied:
            continue
        if all(manifest['jobs'][i]['id'] in completed for i in worker['indices']):
            continue
        if set(worker['dependencies']) <= completed:
            ready[worker['resource']].append(index)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('NEUROSYM_ROOT', Path.cwd())))
    for flag in ('submit', 'resume', 'watch', 'retry-failed', 'allow-final'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--time', help='Allocation HH:MM:SS override; scientific fit definitions remain unchanged')
    parser.add_argument('--worker', type=int, action='append', help='Selected actual workers for timing; include prerequisites')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    expected = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'content_hash'},
        sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if expected != manifest['content_hash']:
        raise ValueError('Manifest content hash changed.')
    if manifest['phase'] == 'final' and args.submit and not args.allow_final:
        raise ValueError('Final evaluation requires explicit --allow-final.')
    stages = submission_groups(manifest)
    profiles = manifest['resources']['resources']
    reserve = manifest['resources']['execution']['yield_reserve_seconds']
    for resource, profile in profiles.items():
        wall = args.time or profile['time']
        if seconds(wall) <= reserve:
            raise ValueError('Walltime must exceed the continuation reserve.')
        print(resource, 'workers=' + str(sum(len(s['workers']) for s in stages if s['resource'] == resource)),
              'concurrency=' + str(profile['concurrency']), 'partition=' + profile['partition'], 'time=' + wall)
    allowed = set(args.worker if args.worker is not None else range(len(manifest['workers'])))
    if not allowed or min(allowed) < 0 or max(allowed) >= len(manifest['workers']):
        raise ValueError('Worker selection is outside the manifest.')
    print('Logical jobs:', len(manifest['jobs']), 'workers:', len(manifest['workers']))
    if not args.submit:
        print('DRY RUN: nothing submitted or created. No artificial serial lanes; only ready tasks dispatch.')
        return
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Dispatch requires your allocated cluster session.')
    root = args.root.resolve()
    if root != Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics'):
        raise ValueError('Unexpected cluster workspace.')
    directory = root / 'artifacts/submissions' / manifest['content_hash']
    directory.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    with (directory / 'controller.oslock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(args, manifest, root, directory, allowed, profiles, reserve)


def dispatch(args, manifest, root, directory, allowed, profiles, reserve):
    ledger_path = directory / 'dispatch.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'format_version': 1, 'attempts': []}
    if not ledger_path.exists() and list(directory.glob('*-submission.json')):
        raise RuntimeError('Legacy submission ledger exists; inspect existing jobs before migration.')
    installed = json.loads((root / '.analysis-source.json').read_text())
    code = root / installed['code_root']
    worker_script = code / 'scripts/run_analysis_array.sbatch'
    if not worker_script.is_file():
        raise FileNotFoundError('Install the verified analysis bundle first.')
    results = root / manifest['analysis_config']['output'] / 'execution' / manifest['content_hash']
    completed, known_ids = set(), {j['id'] for j in manifest['jobs']}
    def receipt(identity):
        path = results / (identity + '.json')
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if value.get('status') != 'complete' or value.get('manifest_hash') != manifest['content_hash'] or value.get('job_id') != identity:
            raise ValueError('Invalid logical completion receipt: ' + str(path))
        if any(not (root / p / 'complete.json').is_file() for p in value['results']):
            raise ValueError('A completed dependency output is missing.')
        return True
    for path in results.glob('*.json'):
        if path.stem in known_ids and receipt(path.stem):
            completed.add(path.stem)
    (root / 'logs').mkdir(exist_ok=True)
    while True:
        if any(a['state'] == 'submitting' for a in ledger['attempts']):
            raise RuntimeError('Uncertain sbatch outcome in dispatch.json. Inspect scheduler and reconcile the saved submission before retrying.')
        queue = subprocess.run(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i|%T'],
                               check=True, capture_output=True, text=True).stdout
        active = {line.split('|')[0].strip() for line in queue.splitlines() if '|' in line}
        occupied, counts = set(), defaultdict(int)
        absent = [a['task_id'] for a in ledger['attempts'] if a['state'] == 'submitted' and a['task_id'] not in active]
        accounting = {}
        if absent:
            report = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(absent), '--format=JobID%64,State,ExitCode'],
                                    check=True, capture_output=True, text=True).stdout
            accounting = {p[0].strip(): (p[1].strip(), p[2].strip()) for line in report.splitlines() if len(p := line.split('|')) >= 3}
        failures = []
        for attempt in ledger['attempts']:
            if attempt['state'] == 'failed':
                if args.retry_failed:
                    attempt['state'] = 'retry-authorized'
                else:
                    failures.append(attempt)
                continue
            if attempt['state'] != 'submitted':
                continue
            index = attempt['worker']
            state, exitcode = accounting.get(attempt['task_id'], ('UNKNOWN', ''))
            terminal = (state.split()[0].split('+')[0] if state else 'UNKNOWN') in {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
            if attempt['task_id'] in active or not terminal:
                occupied.add(index)
                counts[attempt['resource']] += 1
                continue
            worker = manifest['workers'][index]
            for i in worker['indices']:
                identity = manifest['jobs'][i]['id']
                if receipt(identity):
                    completed.add(identity)
            status_path = results / 'attempts' / (str(index) + '.json')
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if all(manifest['jobs'][i]['id'] in completed for i in worker['indices']):
                attempt['state'] = 'complete'
            elif exitcode == '75:0' and status.get('status') == 'yielded' and status.get('attempt') == attempt['token']:
                attempt['state'] = 'yielded'
                attempt['progress_steps'] = status.get('progress_steps', 0)
                prior = [a for a in ledger['attempts'] if a['worker'] == index and a is not attempt and a['state'] == 'yielded']
                if not attempt['progress_steps'] and prior and not prior[-1].get('progress_steps', 0):
                    attempt.update(state='failed', slurm_state=state, exitcode=exitcode,
                                   reason='Two allocations yielded without saved progress; inspect logs and increase walltime.')
                    failures.append(attempt)
            else:
                attempt.update(state='failed', slurm_state=state, exitcode=exitcode)
                failures.append(attempt)
        write_json(ledger_path, ledger)
        if failures:
            raise RuntimeError('Diagnose failed workers; submitted jobs were not cancelled: ' + ', '.join(a['task_id'] for a in failures))
        if all(all(manifest['jobs'][i]['id'] in completed for i in manifest['workers'][w]['indices']) for w in allowed):
            print('SELECTED WORKERS COMPLETE', flush=True)
            return
        ready = ready_workers(manifest, completed, occupied, allowed)
        submitted = 0
        for resource, candidates in ready.items():
            p = profiles[resource]
            selected = candidates[:max(0, p['concurrency'] - counts[resource])]
            if not selected:
                continue
            token, wall = uuid.uuid4().hex, args.time or p['time']
            mapping = directory / (token + '.json')
            write_json(mapping, {'manifest': str(args.manifest.resolve()), 'workers': selected,
                                'manifest_hash': manifest['content_hash'], 'attempt': token})
            command = ['sbatch', '--parsable', '--account=' + manifest['resources']['account'],
                '--qos=' + manifest['resources']['qos'], '--partition=' + p['partition'],
                '--cpus-per-task=' + str(p['cpus']), '--mem=' + p['memory'], '--time=' + wall,
                '--signal=B:USR1@' + str(reserve), '--array=0-' + str(len(selected)-1) + '%' + str(len(selected)),
                '--job-name=ns-' + resource, '--chdir=' + str(root), '--comment=accept_cost',
                '--output=' + str(root / 'logs/analysis-%A_%a.log'),
                '--export=ALL,NEUROSYM_ANALYSIS_CODE=' + str(code) + ',NEUROSYM_WALL_SECONDS=' + str(seconds(wall)) +
                ',NEUROSYM_YIELD_RESERVE=' + str(reserve)]
            if p['gpus']:
                command.append('--gres=gpu:1')
            command.extend([str(worker_script), str(mapping)])
            if args.allow_final:
                command.append('--allow-final')
            attempts = [{'worker': w, 'resource': resource, 'state': 'submitting', 'token': token,
                         'mapping': str(mapping), 'command': command} for w in selected]
            ledger['attempts'].extend(attempts)
            write_json(ledger_path, ledger)
            job = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]
            if not job.isdigit():
                raise ValueError('Unexpected sbatch output; inspect the saved ledger.')
            for slot, attempt in enumerate(attempts):
                attempt.update(state='submitted', task_id=job + '_' + str(slot))
            write_json(ledger_path, ledger)
            submitted += len(selected)
            print('SUBMITTED', resource, job, 'workers=', selected, 'walltime=', wall, flush=True)
        if not args.watch:
            print('Wave dispatched. Continue with --submit --resume; add --watch for ongoing dispatch on an allocated node.')
            return
        if not submitted and not occupied:
            raise RuntimeError('Selected workers await preparation outside this selection; include those dependencies.')
        time.sleep(manifest['resources']['execution']['poll_seconds'])


if __name__ == '__main__':
    main()
