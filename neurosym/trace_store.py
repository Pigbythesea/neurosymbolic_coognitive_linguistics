"""Source-table trace storage, public query membership, and legacy read support.

Version 2 stores arrays once per source/repeat, not thousands of HDF objects per
query. Targets occur only in prediction records, never in compiled execution.
All old fields remain available through QueryTrace; no precision reduction.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .decoders import acceptable_loss, descriptor_key
from .runtime import deadline
from .storage import require_free_space


FORMAT = 2


def json_array(value):
    return np.frombuffer(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), dtype=np.uint8)


def read_json_array(value):
    return json.loads(bytes(value[()]).decode('utf-8'))


class Array:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value if key == () else self.value[key]


class Fields(dict):
    def __init__(self, arrays=(), **attributes):
        super().__init__((k, Array(v)) for k, v in dict(arrays).items())
        self.attrs = attributes


class QueryTrace(dict):
    """Small views of shared in-memory arrays, compatible with legacy readers."""
    def __init__(self, arrays, metadata, index, prediction):
        self.attrs = {k: prediction[k] for k in ('query_id', 'source_id', 'story_id', 'family')}
        self.attrs['prediction_complete'] = True
        self.attrs['public_input_query_id'] = prediction['query_id']
        self.attrs['public_input_origin'] = 'compiled query and candidate catalogs pinned by complete.json'
        entry = metadata['queries'][index]
        super().__init__(site_mask=Array(arrays['site_mask']), latent=Array(arrays['latent'][index]))
        self['primitive'] = {str(i): Fields({'scores': arrays['scores'][p], 'routing': arrays['routing'][p]},
            description=json.dumps(metadata['primitive'][p], ensure_ascii=False))
            for i, p in sorted(enumerate(entry['primitive']), key=lambda row: str(row[0]))}
        rank = arrays['left_factor'].shape[-1]
        self['relations'] = {str(i): Fields({'left_factor': arrays['left_factor'][p], 'right_factor': arrays['right_factor'][p]},
            operation=json.dumps(metadata['relations'][p], ensure_ascii=False), scale=json.dumps(math.sqrt(rank)))
            for i, p in sorted(enumerate(entry['relations']), key=lambda row: str(row[0]))}
        # Keep step order, including repeated operations.
        self['intermediate'] = {str(i): Fields({'routing': arrays['intermediate'][entry['path'], i]},
            operation=json.dumps(entry['step_descriptors'][i], ensure_ascii=False))
            for i in sorted(range(entry['steps']), key=str)}


def block_queries(group):
    arrays = {k: v[()] for k, v in group.items() if k not in {'metadata', 'predictions'}}
    metadata, predictions = read_json_array(group['metadata']), read_json_array(group['predictions'])
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i]['query_id']):
        row = predictions[index]
        yield row['query_id'], str(row['repeat']), QueryTrace(arrays, metadata, index, row)


def iter_sources(file):
    """Yield source ID and a query iterator; compact arrays live one block only.

    Legacy sources are indexed from attributes once, then read in the same
    query/repeat lexical order used by the original geometry implementation.
    """
    if int(file.attrs.get('trace_format', 1)) == FORMAT:
        for source, repeats in file['sources'].items():
            def queries(repeats=repeats):
                # All repeats for a source are needed for query-repeat means.
                values = []
                for repeat, group in repeats.items():
                    if not group.attrs.get('complete', False):
                        raise ValueError('Incomplete source trace block.')
                    values.extend(block_queries(group))
                yield from sorted(values, key=lambda row: (row[0], row[1]))
            yield source, queries()
    else:
        sources = defaultdict(list)
        for query, repeats in file.items():
            for repeat, group in repeats.items():
                sources[group.attrs['source_id']].append((query, repeat))
        for source, keys in sources.items():
            yield source, ((query, repeat, file[query][repeat]) for query, repeat in keys)


@torch.no_grad()
def capture_source(model, queries, window):
    """Execute legacy arithmetic/order, sharing captured tensors before transfer.

    Do not replace this with SourceProgram: close scores can collapse distinct
    floating signatures and change the existing exact-value geometry weights.
    Stacking tensors only copies values; no batched arithmetic is introduced.
    """
    model.descriptors.begin_source()
    observed = model.encode_observation(window, checked=True)
    primitives, relations, memberships, latents, intermediates = [], [], [], [], []
    pids, rids, metrics, probabilities = {}, {}, [], []
    metadata = {'primitive': [], 'relations': [], 'queries': memberships}
    used_bytes = 0
    def tensor_key(value):
        return value.data_ptr(), tuple(value.shape), tuple(value.stride())
    try:
        for example in queries:
            logits, trace = model.answer(observed, example['inputs'], capture=True)
            ps, rs = [], []
            for item in trace['primitive']:
                key = (descriptor_key(item['description'], representation='repr'), tensor_key(item['scores']))
                if key not in pids:
                    pids[key] = len(primitives)
                    primitives.append((item['scores'], item['routing']))
                    metadata['primitive'].append(item['description'])
                    used_bytes += sum(v.numel() * v.element_size() for v in primitives[-1])
                ps.append(pids[key])
            for item in trace['relations']:
                key = (descriptor_key(item['operation'], representation='repr'),
                       tensor_key(item['left_factor']), tensor_key(item['right_factor']))
                if key not in rids:
                    rids[key] = len(relations)
                    relations.append((item['left_factor'], item['right_factor']))
                    metadata['relations'].append(item['operation'])
                    used_bytes += sum(v.numel() * v.element_size() for v in relations[-1])
                rs.append(rids[key])
            memberships.append({'primitive': ps, 'relations': rs, 'path': len(latents),
                'steps': len(trace['intermediate']), 'step_descriptors': [v['operation'] for v in trace['intermediate']]})
            latents.append(trace['latent'])
            intermediates.append([v['routing'] for v in trace['intermediate']])
            accepted = torch.tensor(example['acceptable_indices'], device=logits.device)
            metrics.append(torch.stack(((logits.argmax() == accepted).any().to(logits.dtype),
                                        acceptable_loss(logits, example['acceptable_indices']))))
            probabilities.append(logits.softmax(-1))
            used_bytes += (trace['latent'].numel() + sum(v.numel() for v in intermediates[-1]) + logits.numel() + 2) * logits.element_size()
            # Includes room for stacked copies and the bulk host staging copy.
            if 3 * used_bytes > model.source_program_budget:
                raise MemoryError('Source trace staging exceeds decoder.program_cache_mib; no truncated traces are published.')
        sites = len(model.site_mask)
        depth = max(map(len, intermediates), default=0)
        padded_bytes = len(queries) * depth * sites * observed.element_size()
        if 3 * used_bytes + 2 * padded_bytes > model.source_program_budget:
            raise MemoryError('Padded composition trace staging exceeds decoder.program_cache_mib.')
        intermediate = observed.new_zeros((len(queries), depth, sites))
        for i, values in enumerate(intermediates):
            if values:
                intermediate[i, :len(values)] = torch.stack(values)
        empty = observed.new_empty((0, sites, model.rank))
        arrays = {'scores': torch.stack([v[0] for v in primitives]),
            'routing': torch.stack([v[1] for v in primitives]), 'site_mask': model.site_mask,
            'left_factor': torch.stack([v[0] for v in relations]) if relations else empty,
            'right_factor': torch.stack([v[1] for v in relations]) if relations else empty,
            'latent': torch.stack(latents), 'intermediate': intermediate}
        return arrays, metadata, torch.stack(metrics), torch.cat(probabilities)
    finally:
        model.descriptors.end_source()
        model.end_observation()


def export(model, examples, windows, file, *, probabilities=True, free_reserve_bytes=0):
    """Lossless source tables with independently committed source/repeat blocks."""
    from .decoder_fit import device_windows
    if len(file) and int(file.attrs.get('trace_format', 1)) != FORMAT:
        raise ValueError('Legacy partial traces require explicit import; never overwrite them in place.')
    file.attrs['trace_format'] = FORMAT
    sources = file.require_group('sources')
    windows = device_windows(windows, next(model.parameters()).device)
    groups = defaultdict(list)
    for example in examples:
        groups[example['source_id']].append(example)
    if set(sources) - set(groups):
        raise ValueError('Partial trace contains sources outside the current evaluation.')
    def saved_rows(source, repeat, queries):
        rows = read_json_array(sources[source][str(repeat)]['predictions'])
        expected = [e['query_id'] for e in queries]
        if ([r['query_id'] for r in rows] != expected or
                any(r['repeat'] != repeat or r['source_id'] != source for r in rows)):
            raise ValueError('Committed trace query/repetition inventory changed.')
        return rows
    rows = []
    model.eval()
    cumulative = json.loads(file.attrs.get('phase_seconds', '{}'))
    with torch.no_grad():
        for source, queries in groups.items():
            pending = [r for r in range(len(windows[source])) if
                       str(r) not in sources.get(source, {}) or not sources[source][str(r)].attrs.get('complete', False)]
            if not pending:
                for repeat in range(len(windows[source])):
                    rows.extend(saved_rows(source, repeat, queries))
                continue
            deadline.check()
            if free_reserve_bytes:
                require_free_space(Path(file.filename).parent, 0, free_reserve_bytes)
            for repeat in range(len(windows[source])):
                parent = sources.require_group(source)
                key = str(repeat)
                if key in parent and parent[key].attrs.get('complete', False):
                    rows.extend(saved_rows(source, repeat, queries)); continue
                if key in parent:
                    del parent[key]
                deadline.check()
                if windows[source].is_cuda:
                    torch.cuda.synchronize(windows[source].device)
                started = time.perf_counter()
                captured, metadata, metrics, flat = capture_source(model, queries, windows[source][repeat])
                if windows[source].is_cuda:
                    torch.cuda.synchronize(windows[source].device)
                cumulative['compute'] = cumulative.get('compute', 0.) + time.perf_counter() - started
                started = time.perf_counter()
                arrays = {k: value.detach().cpu().numpy() for k, value in captured.items()}
                measured = metrics.cpu().numpy()
                predicted = flat.cpu().numpy() if probabilities else None
                cumulative['transfer'] = cumulative.get('transfer', 0.) + time.perf_counter() - started
                records = []
                offset = 0
                for i, example in enumerate(queries):
                    row = {k: example[k] for k in ('query_id', 'source_id', 'story_id', 'family', 'weight', 'annotation_review_status')}
                    count = len(example['inputs']['candidates'])
                    row.update(repeat=repeat, observation_source=source, acceptable_indices=example['acceptable_indices'],
                               chance=len(example['acceptable_indices']) / count, correct=float(measured[i, 0]), nll=float(measured[i, 1]))
                    if probabilities:
                        row['probabilities'] = predicted[offset:offset + count].tolist()
                    offset += count
                    records.append(row)
                started = time.perf_counter()
                metadata_bytes, record_bytes = json_array(metadata), json_array(records)
                if free_reserve_bytes:
                    require_free_space(Path(file.filename).parent,
                        sum(v.nbytes for v in arrays.values()) + metadata_bytes.nbytes + record_bytes.nbytes + 65536,
                        free_reserve_bytes)
                group = parent.create_group(key)
                for name, value in arrays.items():
                    group.create_dataset(name, data=value, compression='lzf' if value.ndim else None)
                group.create_dataset('metadata', data=metadata_bytes, compression='lzf')
                group.create_dataset('predictions', data=record_bytes, compression='lzf')
                group.attrs['complete'] = True
                file.flush()
                cumulative['write'] = cumulative.get('write', 0.) + time.perf_counter() - started
                file.attrs['phase_seconds'] = json.dumps(cumulative)
                file.flush()
                rows.extend(records)
                deadline.advance()
                print('TRACE SOURCE', json.dumps({'source': source, 'repeat': repeat, 'queries': len(queries),
                      'cumulative_phase_seconds': cumulative}), flush=True)
                deadline.check()
    return rows
