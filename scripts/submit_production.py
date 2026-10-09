"""First-day production controller with a fixed selection, stop and six-GPU cap.

Uses the existing immutable workers and dispatch ledger. Dry-run by default.
Never installs code, changes scientific definitions, or releases final evaluation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_plan import (CLUSTER_ROOT, CODE_ROOT, DEFAULT_GPUS, MANIFEST,
                             choose_wave, digest, inside, load_manifest, make_plan,
                             milestone_progress, read, write)

STOP = False


def request_handoff(signum, frame):
    global STOP
    STOP = True


def verify_packet(directory):
    packet = read(directory / 'packet.json')
    if packet['manifest_hash'] != MANIFEST or packet['code_root'] != CODE_ROOT or digest(packet) != directory.name:
        raise ValueError('Wrong production transfer packet.')
    for name, expected in packet['files'].items():
        if hashlib.sha256(inside(directory, name).read_bytes()).hexdigest() != expected:
            raise ValueError('Production packet changed: ' + name)
    return packet


def verify_installation(root, m):
    installed = read(root / '.analysis-source.json')
    if installed['code_root'] != CODE_ROOT:
        raise ValueError('Expected qualified analysis installation is not selected.')
    code = inside(root, CODE_ROOT)
    for name, expected in m['code'].items():
        path = code / (name if name.startswith('scripts/') else 'neurosym/' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned analysis source changed: ' + name)
    return code


def receipt(root, execution, identity):
    path = execution / (identity + '.json')
    if not path.exists():
        return False
    value = read(path)
    if (value.get('status') != 'complete' or value.get('manifest_hash') != MANIFEST or
            value.get('job_id') != identity or not value.get('results')):
        raise ValueError('Invalid execution completion: ' + str(path))
    if any(not (inside(root, p) / 'complete.json').is_file() for p in value['results']):
        raise ValueError('Completed output is missing: ' + str(path))
    return True


def command_output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def reconcile(root, m, ledger, active, accounting, completed, execution):
    """Retain the qualified controller's receipt/token/exit-75 continuation rules."""
    occupied, resources, lanes = set(), Counter(), Counter()
    failures = []
    policy = make_plan(m)['workers']
    for attempt in ledger['attempts']:
        if attempt['state'] == 'submitting':
            raise RuntimeError('Uncertain worker submission in dispatch.json; reconcile Slurm before retrying.')
        if attempt['state'] == 'failed':
            failures.append(attempt)
            continue
        if attempt['state'] != 'submitted':
            continue
        index = attempt['worker']
        state, exitcode = accounting.get(attempt['task_id'], ('UNKNOWN', ''))
        terminal = (state.split()[0].split('+')[0] if state else 'UNKNOWN') in {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY',
                                                     'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
        if attempt['task_id'] in active or not terminal:
            occupied.add(index)
            resources[attempt['resource']] += 1
            lanes[policy[index]['lane']] += 1
            continue
        worker = m['workers'][index]
        for i in worker['indices']:
            identity = m['jobs'][i]['id']
            if receipt(root, execution, identity):
                completed.add(identity)
        status_path = execution / 'attempts' / (str(index) + '.json')
        status = read(status_path) if status_path.exists() else {}
        if all(m['jobs'][i]['id'] in completed for i in worker['indices']):
            attempt['state'] = 'complete'
        elif exitcode == '75:0' and status.get('status') == 'yielded' and status.get('attempt') == attempt['token']:
            attempt.update(state='yielded', progress_steps=status.get('progress_steps', 0))
            prior = [a for a in ledger['attempts'] if a['worker'] == index and a is not attempt and a['state'] == 'yielded']
            if not attempt['progress_steps'] and prior and not prior[-1].get('progress_steps', 0):
                attempt.update(state='failed', reason='Two yields without saved progress; inspect before increasing walltime.')
                failures.append(attempt)
        else:
            attempt.update(state='failed', slurm_state=state, exitcode=exitcode)
            failures.append(attempt)
    return occupied, resources, lanes, failures


def other_allocations(root, current, active):
    gpu, cpu = 0, 0
    for path in (root / 'artifacts/submissions').glob('*/dispatch.json'):
        if path == current:
            continue
        for a in read(path)['attempts']:
            if a['state'] == 'submitting':
                raise RuntimeError('Uncertain project submission: ' + str(path))
            if a.get('task_id') in active:
                if any(c.startswith('--gres=gpu:') for c in a['command']):
                    gpu += 1
                else:
                    cpu += 1
    return gpu, cpu


def submit_wave(root, m, plan, selected, directory, ledger, code, packet_hash):
    if not set(selected) <= set(plan['milestone']['worker_indices']):
        raise ValueError('Submission would escape the authorized first-day selection.')
    by_resource = defaultdict(list)
    for index in selected:
        by_resource[m['workers'][index]['resource']].append(index)
    reserve = m['resources']['execution']['yield_reserve_seconds']
    for resource, indices in by_resource.items():
        profile = m['resources']['resources'][resource]
        token = uuid.uuid4().hex
        mapping = directory / (token + '.json')
        write(mapping, {'manifest': str(root / 'artifacts/execution' / MANIFEST / 'manifest.json'),
                        'workers': indices, 'manifest_hash': MANIFEST, 'attempt': token})
        command = ['sbatch', '--parsable', '--account=' + m['resources']['account'], '--qos=' + m['resources']['qos'],
            '--partition=' + profile['partition'], '--cpus-per-task=' + str(profile['cpus']), '--mem=' + profile['memory'],
            '--time=00:30:00', '--signal=B:USR1@' + str(reserve),
            '--array=0-' + str(len(indices)-1) + '%' + str(len(indices)), '--job-name=ns-' + resource,
            '--chdir=' + str(root), '--comment=accept_cost', '--output=' + str(root / 'logs/analysis-%A_%a.log'),
            '--export=ALL,NEUROSYM_ANALYSIS_CODE=' + str(code) + ',NEUROSYM_WALL_SECONDS=1800,NEUROSYM_YIELD_RESERVE=' + str(reserve)]
        if profile['gpus']:
            command.append('--gres=gpu:1')
        command.extend([str(code / 'scripts/run_analysis_array.sbatch'), str(mapping)])
        attempts = [{'worker': w, 'resource': resource, 'state': 'submitting', 'token': token,
                     'mapping': str(mapping), 'command': command, 'production_packet': packet_hash,
                     'production_milestone': plan['milestone']['selection_sha256'],
                     'production_lane': plan['workers'][w]['lane']} for w in indices]
        ledger['attempts'].extend(attempts)
        write(directory / 'dispatch.json', ledger)
        job = command_output(command).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain sbatch response; inspect saved ledger.')
        for slot, attempt in enumerate(attempts):
            attempt.update(state='submitted', task_id=job + '_' + str(slot))
        write(directory / 'dispatch.json', ledger)
        print('SUBMITTED', resource, job, 'workers=', indices, flush=True)


def successor_command(root, packet_dir, max_gpus, current):
    # Pin the same immutable packet, never follow a mutable transfer receipt.
    return ['sbatch', '--parsable', '--dependency=afterany:' + current,
            '--chdir=' + str(root), str(packet_dir / 'run_production.sbatch'), str(packet_dir), str(max_gpus)]


def queue_successor(root, packet_dir, state_path, max_gpus, selection_hash):
    """Only a clean controller time boundary schedules a continuation."""
    if read(packet_dir / 'production-plan.json')['milestone']['selection_sha256'] != selection_hash:
        raise ValueError('Controller continuation would change milestone selection.')
    state = read(state_path) if state_path.exists() else {'controllers': []}
    if any(x['state'] == 'submitting' for x in state['controllers']):
        raise RuntimeError('Uncertain controller continuation; inspect controller-chain.json.')
    current = os.environ['SLURM_JOB_ID']
    previous = [x for x in state['controllers'] if x['parent'] == current]
    if previous:
        print('CONTROLLER CONTINUATION ALREADY RECORDED', previous[-1], flush=True)
        return
    entry = {'parent': current, 'state': 'submitting', 'packet': str(packet_dir),
             'max_gpus': max_gpus, 'selection_sha256': selection_hash}
    state['controllers'].append(entry)
    write(state_path, state)
    command = successor_command(root, packet_dir, max_gpus, current)
    job = command_output(command).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Uncertain controller successor submission; inspect chain record.')
    entry.update(state='submitted', job=job)
    write(state_path, state)
    print('CONTROLLER CONTINUATION', job, 'after', current, flush=True)


def controller_action(plan, completed, boundary, continue_controller):
    # Completion wins even if it coincides with a walltime/signal boundary. It
    # must never schedule an unnecessary successor or expand the selection.
    if set(plan['milestone']['job_ids']) <= completed:
        return 'complete'
    if boundary:
        return 'handoff' if continue_controller else 'stop'
    return 'dispatch'


def launch_report(root, packet_dir, log_path):
    stream = log_path.open('a', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-I', '-B', '-u', str(packet_dir / 'report_production.py'),
        str(root / 'artifacts/execution' / MANIFEST / 'manifest.json'), '--root', str(root)],
        stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    stream.close()
    return process


def dispatch(root, m, plan, packet_dir, max_gpus, continue_controller, controller_seconds):
    code = verify_installation(root, m)
    directory = root / 'artifacts/submissions' / MANIFEST
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / 'dispatch.json'
    ledger = read(ledger_path) if ledger_path.exists() else {'format_version': 1, 'attempts': []}
    execution = root / m['analysis_config']['output'] / 'execution' / MANIFEST
    completed = {j['id'] for j in m['jobs'] if receipt(root, execution, j['id'])}
    control = directory / 'production-control.json'
    if not control.exists():
        write(control, {'max_active_gpus': max_gpus})
    # Explicit invocation overrides a previously persisted ceiling; subsequent
    # edits to production-control.json are picked up without cancelling jobs.
    write(control, {**read(control), 'max_active_gpus': max_gpus})
    packet_hash = file_hash(packet_dir / 'packet.json')
    history = directory / 'production-controllers.jsonl'
    with history.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'job': os.environ['SLURM_JOB_ID'], 'packet_sha256': packet_hash,
            'plan_sha256': digest(plan), 'selection_sha256': plan['milestone']['selection_sha256'],
            'started_utc': datetime.now(timezone.utc).isoformat(), 'max_gpus': max_gpus}) + '\n')
    chain = directory / 'controller-chain.json'
    if chain.exists() and any(x['state'] == 'submitting' for x in read(chain)['controllers']):
        raise RuntimeError('Uncertain controller continuation; reconcile before proceeding.')
    report_process, last_report = None, -float('inf')
    report_log = root / 'logs' / ('production-report-' + os.environ['SLURM_JOB_ID'] + '.log')
    started = time.monotonic()
    try:
        while True:
            max_gpus = read(control)['max_active_gpus']
            if type(max_gpus) is not int or not 1 <= max_gpus <= 6:
                raise ValueError('production-control.json GPU cap must be an integer from one to six.')
            queue = command_output(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i|%T'])
            active = {line.split('|')[0].strip() for line in queue.splitlines() if '|' in line}
            absent = [a['task_id'] for a in ledger['attempts'] if a['state'] == 'submitted' and a['task_id'] not in active]
            accounting = {}
            if absent:
                raw = command_output(['sacct', '-X', '-n', '-P', '-j', ','.join(absent), '--format=JobID%64,State,ExitCode'])
                accounting = {p[0].strip(): (p[1].strip(), p[2].strip()) for line in raw.splitlines() if len(p := line.split('|')) >= 3}
            occupied, counts, lanes, failures = reconcile(root, m, ledger, active, accounting, completed, execution)
            write(ledger_path, ledger)
            if failures:
                raise RuntimeError('Worker failure needs inspection; active jobs are retained: ' + ', '.join(a.get('task_id', '?') for a in failures))
            if report_process is not None and report_process.poll() is not None:
                if report_process.returncode:
                    raise RuntimeError('Interim report failed; inspect ' + str(report_log))
                report_process = None
            boundary = STOP or time.monotonic() - started >= controller_seconds - 180
            action = controller_action(plan, completed, boundary, continue_controller)
            if action != 'dispatch':
                if report_process is not None:
                    try:
                        report_process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        report_process.terminate()
                        report_process.wait(timeout=20)
                    report_process = None
                if action == 'complete':
                    # Ensure the final snapshot includes the just-completed outputs.
                    report_process = launch_report(root, packet_dir, report_log)
                    try:
                        status = report_process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        report_process.terminate()
                        report_process.wait(timeout=20)
                        report_process = None
                        raise RuntimeError('Milestone snapshot exceeded two minutes; inspect report log. Selected scientific jobs remain complete.')
                    if status != 0:
                        raise RuntimeError('Final interim snapshot failed: ' + str(report_log))
                    snapshot = read(root / 'artifacts/production-reports' / MANIFEST / 'latest/complete.json')
                    progress = snapshot.get('milestone', {})
                    if (progress.get('selection_sha256') != plan['milestone']['selection_sha256'] or
                            not progress.get('execution_complete')):
                        raise RuntimeError('Final report does not confirm the selected milestone receipts.')
                    write(directory / ('milestone-' + plan['milestone']['selection_sha256'] + '.json'),
                          {'status': 'complete', 'manifest_hash': MANIFEST,
                           'milestone': milestone_progress(plan, completed, completed),
                           'report_generation': snapshot['generation'],
                           'finished_utc': datetime.now(timezone.utc).isoformat(),
                           'remaining_inventory_automatically_submitted': False})
                    print('FIRST-DAY MILESTONE COMPLETE; stopping for scientific review. '
                          'Remaining development inventory and final story are not submitted.', flush=True)
                elif action == 'handoff':
                    queue_successor(root, packet_dir, chain, max_gpus, plan['milestone']['selection_sha256'])
                    print('CONTROLLER HANDOFF; same milestone only; running workers continue.', flush=True)
                else:
                    print('CONTROLLER STOPPED AT TIME BOUNDARY; resume this packet; workers continue.', flush=True)
                return
            if report_process is None and time.monotonic() - last_report >= 900:
                report_process = launch_report(root, packet_dir, report_log)
                last_report = time.monotonic()
            other_gpu, other_cpu = other_allocations(root, ledger_path, active)
            admitted = Counter(a.get('production_lane', plan['workers'][a['worker']]['lane']) for a in ledger['attempts'])
            selected = choose_wave(m, plan, completed, occupied, counts, lanes, other_gpu, other_cpu, max_gpus, admitted)
            if selected:
                submit_wave(root, m, plan, selected, directory, ledger, code, packet_hash)
            elif not occupied and not (other_gpu or other_cpu):
                raise RuntimeError('Unfinished milestone has no ready work; inspect dependency receipts.')
            done = len(set(plan['milestone']['job_ids']) & completed)
            print('PRODUCTION STATUS', 'milestone=' + str(done) + '/' + str(plan['milestone']['logical_items']),
                  'full_inventory=' + str(len(completed)) + '/' + str(len(m['jobs'])), 'gpu_ceiling=' + str(max_gpus),
                  'outstanding_by_lane=' + json.dumps(dict(lanes)), flush=True)
            time.sleep(m['resources']['execution']['poll_seconds'])
    finally:
        if report_process is not None and report_process.poll() is None:
            report_process.terminate()
            report_process.wait(timeout=20)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--max-gpus', type=int, choices=range(1, 7), default=DEFAULT_GPUS)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--continue-controller', action='store_true')
    parser.add_argument('--controller-seconds', type=int, default=14400)
    args = parser.parse_args()
    packet_dir = Path(__file__).resolve().parent
    packet = verify_packet(packet_dir)
    root = args.root.resolve()
    m = load_manifest(root / 'artifacts/execution' / MANIFEST / 'manifest.json')
    plan = read(packet_dir / 'production-plan.json')
    if plan != make_plan(m):
        raise ValueError('Production policy no longer agrees with its pinned manifest.')
    print('PRODUCTION PLAN', json.dumps(plan['early_panel']), 'max_gpus=', args.max_gpus, flush=True)
    print('FIRST-DAY MILESTONE', json.dumps({k: v for k, v in plan['milestone'].items()
                                          if k not in {'job_ids', 'worker_indices'}}), flush=True)
    if not args.submit:
        if os.name != 'nt' and os.environ.get('SLURM_JOB_ID'):
            verify_installation(root, m)
        print('DRY RUN; complete inventory retained; nothing submitted or created.')
        return
    if os.name == 'nt' or root.as_posix() != CLUSTER_ROOT or not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Submission requires allocated compute in the designated cluster workspace.')
    if args.controller_seconds < 600:
        raise ValueError('Controller window is too short for safe reporting and handoff.')
    signal.signal(signal.SIGUSR1, request_handoff)
    import fcntl
    lock_path = root / 'artifacts/submissions/controller.oslock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(root, m, plan, packet_dir, args.max_gpus, args.continue_controller, args.controller_seconds)


if __name__ == '__main__':
    main()
