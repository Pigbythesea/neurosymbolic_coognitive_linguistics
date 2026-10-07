"""Heldout query-conditioned evidence replacement; no anatomical labels."""
from collections import defaultdict
import hashlib
import math
import numpy as np
import torch

from .analysis_runs import decoder_metrics, write_report
from .io import read_json, object_hash
from .runtime import deadline
from .decoder_batch import SourceProgram, choice_targets, choice_losses
from .decoder_minibatch import observations


def faithfulness(model, examples, windows, sources, *, fraction, seed, checkpoint=None):
    from .decoder_fit import device_windows, mismatch_map
    device = next(model.parameters()).device
    windows = device_windows(windows, device)
    donor = mismatch_map(sources)
    groups = defaultdict(list)
    for e in examples:
        groups[e['source_id']].append(e)
    sites = np.flatnonzero(model.site_mask.cpu().numpy())
    if len(sites) < 2 or not 0 < fraction < 1:
        raise ValueError('Faithfulness needs multiple observed sites and a valid fraction.')
    count = max(1, min(len(sites) - 1, math.ceil(fraction * len(sites))))
    rows, masks = {'selected': [], 'control': []}, []
    identity = object_hash({'fraction': fraction, 'seed': seed, 'queries': [e['query_id'] for e in examples]})
    completed = []
    if checkpoint and checkpoint.exists():
        saved = read_json(checkpoint)
        if saved['identity'] != identity:
            raise ValueError('Faithfulness checkpoint support changed.')
        rows, masks, completed = saved['rows'], saved['masks'], saved['completed']
    model.eval()
    with torch.no_grad():
        for source, queries in groups.items():
            for repeat, values in enumerate(windows[source]):
                if [source, repeat] in completed:
                    continue
                observed = model.encode_observation(values, checked=True)
                model.descriptors.begin_source()
                pending, altered = [], []
                for e in queries:
                    _, trace = model.answer(observed, e['inputs'], capture=True)
                    if not trace['primitive']:
                        raise ValueError('Structured query has no primitive grounding evidence.')
                    # Only public-query routes, never the correct answer, select sites.
                    route = torch.stack([p['routing'] for p in trace['primitive']]).mean(0).cpu().numpy()
                    selected = sites[np.argsort(-route[sites], kind='stable')[:count]]
                    random_seed = int(hashlib.sha256(f"{seed}:{e['query_id']}:{repeat}".encode()).hexdigest()[:16], 16)
                    control = np.random.default_rng(random_seed).choice(sites, count, replace=False)
                    replacement = windows[donor[source]][repeat % len(windows[donor[source]])]
                    masks.append({'query_id': e['query_id'], 'repeat': repeat, 'selected': selected.tolist(),
                                  'control': control.tolist(), 'donor_source': donor[source]})
                    for label, chosen in [('selected', selected), ('control', control)]:
                        modified = values.clone()
                        modified[torch.as_tensor(chosen, device=device)] = replacement[torch.as_tensor(chosen, device=device)]
                        altered.append(modified); pending.append((label, e))
                model.descriptors.end_source(); model.end_observation()
                for start in range(0, len(pending), 32):
                    current = pending[start:start+32]
                    program = SourceProgram(model, [e['inputs'] for _, e in current], list(range(len(current))))
                    logits = program.execute(model, observations(model, torch.stack(altered[start:start+32])))
                    targets = choice_targets([e for _, e in current], device)
                    losses = choice_losses(logits, targets)
                    correct = ((logits.argmax(-1)[:, None] == targets['indices']) & targets['valid']).any(-1)
                    measurements = torch.stack((correct.to(logits.dtype), losses), -1).cpu().numpy()
                    for index, (label, e) in enumerate(current):
                        row = {k: e[k] for k in ['query_id','source_id','story_id','family','weight']}
                        row.update(repeat=repeat, correct=float(measurements[index,0]), nll=float(measurements[index,1]),
                                   chance=len(e['acceptable_indices'])/len(e['inputs']['candidates']))
                        rows[label].append(row)
                completed.append([source, repeat])
                deadline.advance()
                if checkpoint and deadline.due():
                    write_report(checkpoint, {'identity': identity, 'rows': rows, 'masks': masks, 'completed': completed})
                    deadline.check()
    return {'selected': decoder_metrics(rows['selected']), 'control': decoder_metrics(rows['control']),
            'site_fraction': fraction, 'sites_replaced': count, 'selection_uses_answers': False, 'masks': masks,
            'interpretation': 'Query-conditioned mean primitive-route site replacement versus equal-count seeded random sites, using the same nonoverlapping donor at the same site coordinates. Measures this decoder\'s reliance; replacement changes joint activity and is not a biological intervention or proof of unique semantic localization.'}
