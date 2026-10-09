"""Administrative production policy; never changes the immutable fit manifest."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os

MANIFEST = '01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e'
CODE_ROOT = 'analysis_code/17181841f6fd955de63477a406143d2b31e739b4b7507d81846cc4860abe71ff'
CLUSTER_ROOT = '/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics'
DEFAULT_GPUS = 6
EARLY_FOLDS = ('0', '5')
MILESTONE = 'first-day-integrated-v1'
GROUNDING_SYSTEMS = ('subject01', 'subject02', 'subject03', 'qwen38-27b', 'olmo3-7b-base')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('Path escapes project workspace: ' + str(relative))
    return result


def load_manifest(path):
    m = read(path)
    if (m['content_hash'] != MANIFEST or m['phase'] != 'development' or
            digest({k: v for k, v in m.items() if k != 'content_hash'}) != MANIFEST):
        raise ValueError('This production plan requires the pinned development manifest.')
    return m


def primary_layers(manifest):
    return {j['options']['model']: j['options']['layer'] for j in manifest['jobs']
            if j['kind'] == 'decoder' and j['options']['family'] == 'structured'
            and j['options'].get('modality') == 'model'}


def make_plan(m):
    primary = primary_layers(m)
    masks = {k + ':' + str(v) for k, v in primary.items()}
    systems = []
    humans = sorted({j['options']['subject'] for j in m['jobs'] if j['options'].get('modality') == 'brain' and j['options'].get('subject')})
    # Alternate human/model systems; avoid finishing every fold of one system first.
    models = list(primary)
    for i in range(max(len(humans), len(models))):
        if i < len(humans):
            systems.append(humans[i])
        if i < len(models):
            systems.append(models[i])

    def is_primary(o):
        return o.get('modality') == 'brain' or (o.get('model') in primary and o.get('layer') == primary[o['model']])

    jobs = {j['id']: j for j in m['jobs']}
    rows, early_ids = [], set()
    for index, worker in enumerate(m['workers']):
        j = m['jobs'][worker['indices'][0]]
        o, kind = j['options'], j['kind']
        fold = o.get('fold', '')
        context = fold.startswith('stories:')
        primary_job = is_primary(o)
        if kind == 'encoding':
            primary_job = not o.get('mask_models') or set(o['mask_models']) <= masks
        early = fold in EARLY_FOLDS and primary_job and kind in {'prepare-decoder', 'select-decoder', 'decoder', 'encoding'}
        if early:
            early_ids.update(m['jobs'][i]['id'] for i in worker['indices'])
        if kind == 'encoding':
            lane = 'encoding'
            tier = 0 if early else 1 if context else 2 if primary_job else 3
        elif kind in {'geometry-panel', 'compare-geometry-panel'}:
            lane = 'geometry' if j['resource'].startswith('gpu-') else 'cpu'
            # A comparison inherits primary status from its geometry parents.
            parents = [jobs[d]['options'] for d in j['dependencies']] if kind == 'compare-geometry-panel' else [o]
            tier = 0 if all(is_primary(p) or p.get('views') == ['cooccurrence'] for p in parents) else 3
        elif j['resource'].startswith('cpu-'):
            lane = 'cpu'
            tier = -1 if kind == 'semantic-coverage' else 0 if early else 2
        else:
            lane = 'context' if context else 'prediction'
            tier = 0 if early or context else 2 if primary_job else 3
        # Favor completed comparisons over a backlog of new setup, while ensuring
        # preparation is eligible first when its dependents cannot yet run.
        stage = {'decoder': 0, 'select-decoder': 1, 'prepare-decoder': 2,
                 'compare-geometry-panel': 0, 'geometry-panel': 1}.get(kind, 0)
        system = o.get('subject', o.get('model'))
        system_rank = systems.index(system) if system in systems else 0
        fold_rank = (EARLY_FOLDS.index(fold) if fold in EARLY_FOLDS else int(fold) + 2 if fold.isdigit()
                     else 0 if 'story_01' in fold else 1)
        rows.append({'worker': index, 'lane': lane, 'tier': tier, 'stage': stage,
                     'system_rank': system_rank, 'fold_rank': fold_rank,
                     'early_panel': early, 'resource': worker['resource']})
    dependencies = {d for w in m['workers'] if any(m['jobs'][i]['id'] in early_ids for i in w['indices']) for d in w['dependencies']}
    if not dependencies <= early_ids:
        raise ValueError('Early scientific panel is not dependency complete.')
    early_workers = [r for r in rows if r['early_panel']]

    # Select complete scientific comparisons from the qualified inventory. No
    # score-dependent choice, new fit definition, or split of a seed worker.
    native = {j['id'] for j in m['jobs'] if j['kind'] == 'geometry-panel' and
              set(j['options']['views']) <= {'native', 'encoding-implied', 'cooccurrence'} and
              (is_primary(j['options']) or j['options']['views'] == ['cooccurrence'])}
    grounding = {j['id'] for j in m['jobs'] if j['kind'] == 'geometry-panel' and
                 'grounding' in j['options']['views'] and is_primary(j['options']) and
                 j['options'].get('subject', j['options'].get('model')) in GROUNDING_SYSTEMS}
    comparisons = {j['id'] for j in m['jobs'] if j['kind'] == 'compare-geometry-panel' and
                   set(j['dependencies']) <= native | grounding}
    selected = early_ids | native | grounding | comparisons | {
        j['id'] for j in m['jobs'] if j['kind'] == 'semantic-coverage'}
    while True:
        missing = {d for identity in selected for d in jobs[identity]['dependencies']} - selected
        if not missing:
            break
        selected.update(missing)
    selected_workers = []
    for row in rows:
        w = m['workers'][row['worker']]
        ids = {m['jobs'][i]['id'] for i in w['indices']}
        row['selected'] = bool(ids & selected)
        if row['selected']:
            if not ids <= selected or not set(w['dependencies']) <= selected:
                raise ValueError('Milestone would split a worker or omit a dependency.')
            selected_workers.append(row['worker'])
    milestone = {'name': MILESTONE, 'workers': len(selected_workers),
                 'logical_items': len(selected), 'worker_indices': selected_workers,
                 'job_ids': sorted(selected),
                 'counts': dict(Counter(j['kind'] for j in m['jobs'] if j['id'] in selected)),
                 'grounding_systems': list(GROUNDING_SYSTEMS),
                 'selection_rule': 'Fixed IDs, audit reuse and model-family coverage; not effect-size ranking.',
                 'on_complete': 'Stop and report; do not expand to the remaining development inventory.'}
    milestone['selection_sha256'] = digest({'manifest': MANIFEST, 'name': MILESTONE,
                                          'workers': selected_workers, 'jobs': milestone['job_ids']})
    return {'format_version': 2, 'manifest_hash': MANIFEST, 'code_root': CODE_ROOT,
            'default_max_active_gpus': DEFAULT_GPUS, 'maximum_authorized_gpus': 6,
            'early_folds': list(EARLY_FOLDS), 'primary_layers': primary,
            'early_panel': {'workers': len(early_workers), 'logical_items': len(early_ids),
                            'counts': dict(Counter(j['kind'] for j in m['jobs'] if j['id'] in early_ids))},
            'milestone': milestone,
            'policy': 'Dispatch only the dependency-complete first-day release; stop when complete. Remaining inventory retained but not submitted.',
            'workers': rows}


def milestone_progress(plan, available, completed):
    selected = set(plan['milestone']['job_ids'])
    return {'name': plan['milestone']['name'],
            'selection_sha256': plan['milestone']['selection_sha256'],
            'expected_workers': plan['milestone']['workers'],
            'expected_logical_items': len(selected),
            'available_including_qualified_reuse': len(selected & set(available)),
            'current_execution_completed': len(selected & set(completed)),
            'evidence_complete': selected <= set(available),
            'execution_complete': selected <= set(completed),
            'missing_evidence_job_ids': sorted(selected - set(available)),
            'missing_execution_job_ids': sorted(selected - set(completed))}


def lane_weights(limit):
    # Soft shares, not reservations. At six: three prediction, one context,
    # one encoding and one geometry; ready work borrows otherwise idle capacity.
    return {'prediction': 3, 'context': 1, 'encoding': 1, 'geometry': 1}


def choose_wave(m, plan, completed, occupied, active_resources, active_lanes,
                other_gpu=0, other_cpu=0, max_gpus=DEFAULT_GPUS, admitted=None):
    if not 1 <= max_gpus <= plan['maximum_authorized_gpus']:
        raise ValueError('GPU ceiling must be between one and six.')
    profiles = m['resources']['resources']
    counts, lanes = Counter(active_resources), Counter(active_lanes)
    admitted = admitted or {}
    pending = defaultdict(list)
    for row in plan['workers']:
        if not row['selected']:
            continue
        index = row['worker']
        w = m['workers'][index]
        if index in occupied or all(m['jobs'][i]['id'] in completed for i in w['indices']):
            continue
        if set(w['dependencies']) <= completed:
            pending[row['lane']].append(row)
    for lane in pending:
        pending[lane].sort(key=lambda r: (r['tier'], r['stage'], r['fold_rank'], r['system_rank'], r['worker']))
    gpu_used = sum(counts[k] * profiles[k]['gpus'] for k in counts) + other_gpu
    cpu_used = sum(counts[k] for k in counts if not profiles[k]['gpus']) + other_cpu
    picked = []
    weights = lane_weights(max_gpus)
    while pending:
        candidates = []
        for lane, rows in pending.items():
            eligible = next((r for r in rows if counts[r['resource']] < profiles[r['resource']]['concurrency']), None)
            if eligible is None:
                continue
            gpu = profiles[eligible['resource']]['gpus']
            if (gpu and gpu_used >= max_gpus) or (not gpu and cpu_used >= m['resources']['execution']['max_active_cpu_workers']):
                continue
            # Prior/coverage CPU jobs do not compete for GPU shares. Admission
            # history breaks ties fairly, including when the ceiling is one.
            weight = weights.get(lane, 1)
            candidates.append(((0 if not gpu else 1, lanes[lane] / weight,
                                admitted.get(lane, 0) / weight, eligible['tier']), eligible))
        if not candidates:
            break
        _, row = min(candidates, key=lambda p: p[0])
        picked.append(row['worker'])
        pending[row['lane']].remove(row)
        if not pending[row['lane']]:
            del pending[row['lane']]
        counts[row['resource']] += 1
        lanes[row['lane']] += 1
        if profiles[row['resource']]['gpus']:
            gpu_used += 1
        else:
            cpu_used += 1
    return picked
