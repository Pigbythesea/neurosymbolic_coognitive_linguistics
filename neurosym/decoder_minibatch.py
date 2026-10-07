"""Packed multi-observation execution with exact candidate and source weights.

Packing is parameter-free. Batches are fixed by source IDs and a declared seed;
their order is shuffled each epoch. AdamW steps once per batch. This is a new
optimization protocol, not a promise to reproduce per-source AdamW trajectories.
"""
from collections import defaultdict, OrderedDict
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from .decoder_batch import SourceProgram, choice_targets, choice_losses
from .compute import atomic_torch_save
from .runtime import deadline
from .io import object_hash


# Only immutable query programs are shared. Observations, targets, losses,
# parameters, gradients and optimizer states never enter this resident cache.
_PROGRAMS, _PROGRAM_BYTES = OrderedDict(), 0


def packed_program(model, rows, owners):
    global _PROGRAM_BYTES
    namespace = getattr(model, 'program_namespace', None)
    def build():
        return SourceProgram(model, [r[0]['inputs'] for r in rows], owners)
    if namespace is None:
        return build()
    key = (namespace, model.family, str(next(model.parameters()).device),
           str(next(model.parameters()).dtype), object_hash(model.descriptors.vocabulary),
           tuple((r[0]['query_id'], owner) for r, owner in zip(rows, owners, strict=True)))
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key]
    program = build()
    limit = model.source_program_budget // 2
    if program.bytes <= limit:
        while _PROGRAMS and _PROGRAM_BYTES + program.bytes > limit:
            _, removed = _PROGRAMS.popitem(last=False)
            _PROGRAM_BYTES -= removed.bytes
        _PROGRAMS[key] = program
        _PROGRAM_BYTES += program.bytes
    return program


def observations(model, values):
    if model.family == 'prior':
        return None
    if model.family in {'linear', 'mlp'}:
        return model.global_projection(values.flatten(1))
    local = F.gelu(torch.einsum('bpf,pfh->bph', values.flatten(2), model.local_weight))
    return model.local_shared(local) * model.site_mask[None, :, None]


class PackedBatches:
    def __init__(self, model, examples, windows, *, size, pairing=None, packing_seed=1729):
        device = next(model.parameters()).device
        groups = defaultdict(list)
        if not isinstance(size, int) or size < 1 or not examples:
            raise ValueError('Packed execution requires positive batch size and actual queries.')
        for example in examples:
            groups[example['source_id']].append(example)
        counts = defaultdict(int)
        for queries in groups.values():
            counts[queries[0]['story_id']] += 1
        order = np.random.default_rng(packing_seed).permutation(sorted(groups)).tolist()
        self.source_count, self.query_count = len(groups), len(examples)
        self.batches, self.bytes = [], 0
        steps = (len(order) + size - 1) // size
        for start in range(0, len(order), size):
            members = order[start:start + size]
            rows, owners, values, scale = [], [], [], []
            for source in members:
                current = groups[source]
                observed_source = pairing[source] if pairing else source
                repeats = current[0].get('observation_repeats', 1) if windows is None else len(windows[observed_source])
                query_mass = sum(e['weight'] for e in current)
                if query_mass <= 0 or repeats < 1:
                    raise ValueError('A source needs positive query mass and observed repeats.')
                for repeat in range(repeats):
                    slot = len(values)
                    values.append(None if windows is None else windows[observed_source][repeat])
                    for example in current:
                        rows.append((example, repeat, observed_source))
                        owners.append(slot)
                        # Mean over uniformly shuffled batches equals the
                        # equal-story, equal-source, query-weighted objective,
                        # including the shorter last batch and repeated scans.
                        scale.append(steps / (len(counts) * counts[example['story_id']] * repeats * query_mass))
            program = packed_program(model, rows, owners)
            targets = choice_targets([r[0] for r in rows], device)
            weights = targets['weights'] * torch.tensor(scale, device=device)
            self.bytes += program.bytes + sum(v.numel() * v.element_size() for v in targets.values())
            self.batches.append((program, targets, weights, values, rows))
        if self.bytes > model.source_program_budget:
            raise MemoryError(f'Packed query working set needs {self.bytes} bytes; configured budget is {model.source_program_budget}. Increase decoder.program_cache_mib on an appropriately sized allocation.')
        model._source_program_bytes = self.bytes

    def logits(self, model, index):
        program, targets, weights, values, rows = self.batches[index]
        values = None if model.family == 'prior' else torch.stack(values)
        return program.execute(model, observations(model, values)), targets, weights, rows

    def validation_loss(self, model):
        total = torch.zeros((), device=next(model.parameters()).device)
        with torch.no_grad():
            for index in range(len(self.batches)):
                logits, targets, weights, _ = self.logits(model, index)
                total += (choice_losses(logits, targets) * weights).sum()
        return float((total / len(self.batches)).cpu())


def evaluate_batched(model, examples, windows, *, pairing=None, probabilities=True, size=16):
    packed = PackedBatches(model, examples, windows, size=size, pairing=pairing)
    rows = []
    with torch.no_grad():
        for index in range(len(packed.batches)):
            logits, targets, _, metadata = packed.logits(model, index)
            losses = choice_losses(logits, targets)
            correct = ((logits.argmax(-1)[:, None] == targets['indices']) & targets['valid']).any(-1)
            metrics = torch.stack((correct.to(logits.dtype), losses), -1).cpu().numpy()
            predictions = logits.softmax(-1).cpu().numpy() if probabilities else None
            for i, (example, repeat, source) in enumerate(metadata):
                count = len(example['inputs']['candidates'])
                row = {k: example[k] for k in ('query_id', 'source_id', 'story_id', 'family', 'weight', 'annotation_review_status')}
                row.update(repeat=repeat, observation_source=source, acceptable_indices=example['acceptable_indices'],
                           chance=len(example['acceptable_indices']) / count, correct=float(metrics[i, 0]), nll=float(metrics[i, 1]))
                if probabilities:
                    row['probabilities'] = predictions[i, :count].tolist()
                rows.append(row)
    return rows


def train_minibatches(data, model, examples, windows, *, identity, learning_rate, epochs,
                     seed, validation=None, pairing=None, patience=None, checkpoint=None):
    from .decoder_fit import device_windows
    cfg, device = data.config['decoder'], next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg['weight_decay'])
    rng, history = np.random.default_rng(seed), []
    best, best_state, best_epoch, stale, completed, pending = float('inf'), None, 0, 0, 0, None
    if checkpoint and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved.get('executor') != 'cross-source-v1':
            raise ValueError('Checkpoint belongs to different inputs or training protocol.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, best, best_state, best_epoch, stale, completed, pending = [saved[k] for k in
            ('history', 'best', 'best_state', 'best_epoch', 'stale', 'completed', 'pending')]
        print('DECODER RESUME completed_epochs=' + str(completed), flush=True)

    # A finished checkpoint is also a stage boundary. Restoring it must not
    # compile the entire training and validation program again for export.
    if pending is None and (completed >= epochs or (validation and patience is not None and stale >= patience)):
        if validation:
            if best_state is None:
                raise ValueError('Finished selection checkpoint has no selected parameters.')
            model.load_state_dict(best_state)
        return {'best_epoch': best_epoch if validation else epochs,
                'best_validation_nll': best if validation else None,
                'history': history, 'protocol': 'cross-source-v1', 'packing_seconds': 0.,
                'restored_finished_checkpoint': True}
    began = time.perf_counter()
    windows = None if model.family == 'prior' else device_windows(windows, device)
    train = PackedBatches(model, examples, windows, size=cfg['source_batch_size'], pairing=pairing)
    valid = None
    if validation:
        valid = PackedBatches(model, validation[0], None if model.family == 'prior' else device_windows(validation[1], device), size=cfg['source_batch_size'])
    packed_bytes = train.bytes + (valid.bytes if valid else 0)
    if packed_bytes > model.source_program_budget:
        raise MemoryError('Training plus validation query working set exceeds decoder.program_cache_mib.')
    packing_seconds = time.perf_counter() - began

    def save(pending=None):
        if checkpoint:
            atomic_torch_save({'identity': identity, 'executor': 'cross-source-v1', 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                'history': history, 'best': best, 'best_state': best_state, 'best_epoch': best_epoch,
                'stale': stale, 'completed': completed, 'pending': pending}, checkpoint)

    for epoch in range(completed + 1, epochs + 1):
        if valid is not None and patience is not None and stale >= patience:
            break
        started = time.perf_counter()
        model.train()
        if pending:
            if pending['epoch'] != epoch or sorted(pending['order']) != list(range(len(train.batches))):
                raise ValueError('Checkpoint batch inventory changed.')
            order, cursor, previous = pending['order'], pending['cursor'], pending['seconds']
            losses = [torch.tensor(v, device=device) for v in pending['losses']]
            pending = None
        else:
            order, cursor, previous, losses = rng.permutation(len(train.batches)).tolist(), 0, 0., []
        for position in range(cursor, len(order)):
            optimizer.zero_grad(set_to_none=True)
            logits, targets, weights, _ = train.logits(model, order[position])
            loss = (choice_losses(logits, targets) * weights).sum()
            loss.backward()
            # Fail before publishing invalid weights; one check per minibatch,
            # rather than a host synchronization for every individual source.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
            deadline.advance()
            if checkpoint and deadline.due():
                save({'epoch': epoch, 'order': order, 'cursor': position + 1, 'seconds': previous + time.perf_counter() - started,
                      'losses': torch.stack(losses).cpu().tolist()})
                deadline.check()
        record = {'epoch': epoch, 'train_story_macro_nll': float(torch.stack(losses).mean().cpu()),
                  'training_seconds': previous + time.perf_counter() - started, 'query_execution': 'cross-source-v1',
                  'train_sources': train.source_count, 'train_queries': train.query_count,
                  'optimizer_steps': len(train.batches), 'source_batch_size': cfg['source_batch_size'],
                  'packed_program_bytes': packed_bytes, 'packing_seconds': packing_seconds}
        model.eval()
        validation_started = time.perf_counter()
        if valid:
            value = valid.validation_loss(model)
            if not np.isfinite(value):
                raise ValueError('Nonfinite validation loss.')
            record['validation_story_macro_nll'] = value
            if value < best:
                best, best_epoch, stale = value, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        record['validation_seconds'] = time.perf_counter() - validation_started
        history.append(record); completed = epoch
        checkpoint_started = time.perf_counter(); save()
        record['checkpoint_seconds'] = time.perf_counter() - checkpoint_started
        record['wall_seconds'] = previous + time.perf_counter() - started
        print('DECODER EPOCH', json.dumps(record), flush=True)
        deadline.check()
    if valid:
        if best_state is None:
            raise ValueError('No finite selectable epoch.')
        model.load_state_dict(best_state)
    return {'best_epoch': best_epoch if valid else epochs, 'best_validation_nll': best if valid else None,
            'history': history, 'protocol': 'cross-source-v1', 'packing_seconds': packing_seconds}
