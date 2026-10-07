"""One source pass for all requested grounding kinds/scopes, with exact weights."""
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import time

import h5py
import numpy as np

from .io import object_hash
from .runtime import deadline, analysis_lock
from .storage import require_free_space, ResidentCache
from .trace_store import iter_sources, json_array, read_json_array


def build_tables(parent, path, panels, *, reserve_bytes=0):
    """Reduce the immutable parent once; commit each source independently.

    Exact floating-value signature deduplication and query/repeat lexical order
    are unchanged. Pooled/scoped items share calculation only when every input
    to trace_signature is equal. No inference from prediction correctness.
    """
    from .geometry import trace_signature, prepare_trace, vector_digest
    parent, path = Path(parent), Path(path)
    by_source = defaultdict(dict)
    for records in panels:
        for record in records:
            key = (record['source_id'], record['item_id'])
            previous = by_source[key[0]].get(key[1])
            if previous is not None and previous != record:
                raise ValueError('Conflicting source/item definitions in one geometry panel.')
            by_source[key[0]][key[1]] = record
    with h5py.File(parent / 'traces.h5', 'r') as original, analysis_lock(path.with_suffix('.lock')):
        if not original.attrs.get('complete', False):
            raise ValueError('Geometry needs complete parent traces.')
        identity = object_hash({'parent': original.attrs['run_identity'], 'records': dict(by_source),
            'format': 1})
        with h5py.File(path, 'a') as target:
            if len(target) and target.attrs.get('identity') != identity:
                raise ValueError('Grounding source tables belong to another parent/panel.')
            target.attrs['identity'] = identity
            if target.attrs.get('complete', False):
                return path
            for source, groups in iter_sources(original):
                if not by_source[source] or source in target and target[source].attrs.get('complete', False):
                    continue
                deadline.check()
                require_free_space(path.parent, 0, reserve_bytes)
                started = time.perf_counter()
                definitions, assignments = {}, {}
                for item, record in by_source[source].items():
                    signature = object_hash({k: record.get(k) for k in
                        ('kind', 'label', 'grounding_query', 'context_selectors')})
                    definitions.setdefault(signature, record)
                    assignments[item] = signature
                collected = defaultdict(dict)
                shared = ResidentCache(128 * 2**20)
                for _, _, group in groups:
                    prepared = prepare_trace(group, shared=shared) if 'site_mask' in group else None
                    for signature, record in definitions.items():
                        value = trace_signature(group, record, 'grounding', prepared=prepared)
                        if value is not None:
                            collected[signature][vector_digest(value)] = value
                means = {key: np.mean(list(values.values()), axis=0) for key, values in collected.items()}
                require_free_space(path.parent, sum(v.nbytes for v in means.values()) + 65536, reserve_bytes)
                # Store each numerical vector once, with explicit item references.
                widths = defaultdict(list)
                for key, value in means.items():
                    widths[(len(value), value.dtype.str)].append(key)
                references = {}
                if source in target:
                    del target[source]
                group = target.create_group(source)
                for column, keys in enumerate(widths.values()):
                    name = str(column)
                    group.create_dataset(name, data=np.stack([means[k] for k in keys]), compression='lzf')
                    references.update({key: [name, index, len(collected[key])] for index, key in enumerate(keys)})
                group.create_dataset('items', data=json_array({item: references[key] for item, key in assignments.items() if key in references}), compression='lzf')
                group.attrs['complete'] = True
                group.attrs['seconds'] = time.perf_counter() - started
                target.flush()
                deadline.advance()
                print('GEOMETRY SOURCE', source, 'items=' + str(len(assignments)),
                      'seconds=' + str(group.attrs['seconds']), flush=True)
                deadline.check()
            target.attrs['complete'] = True
            target.flush()
    return path


class VectorTable(Mapping):
    """Read only the vectors requested by one panel; one source stays resident."""
    def __init__(self, file, records):
        self.file, self.index, self.loaded, self.source = file, {}, {}, None
        if not file.attrs.get('complete', False):
            raise ValueError('Grounding source tables are incomplete.')
        requested = defaultdict(set)
        for record in records:
            requested[record['source_id']].add(record['item_id'])
        for source, items in requested.items():
            if source in file:
                refs = read_json_array(file[source]['items'])
                self.index.update({(source, item): refs[item] for item in items if item in refs})

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    def __contains__(self, key):
        return key in self.index

    def __getitem__(self, key):
        source, _ = key
        column, index, _ = self.index[key]
        if self.source != source:
            self.loaded = {}
            self.source = source
        if column not in self.loaded:
            self.loaded[column] = self.file[source][column][()]
        return self.loaded[column][index]
