"""Committed fitted weights are independent of resumable downstream export."""
from pathlib import Path

import torch

from .analysis_data import SiteProjector
from .analysis_runs import write_report
from .compute import atomic_torch_save, file_hash
from .decoders import SemanticDecoder
from .io import object_hash, read_json


def load_fitted(directory, identity, device, config):
    directory = Path(directory)
    path = directory / 'fitted.json'
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt['run_identity'] != object_hash(identity):
        raise ValueError('Fitted stage belongs to another run.')
    for name, expected in receipt['files'].items():
        if file_hash(directory / name) != expected:
            raise ValueError('Committed fitted stage changed: ' + name)
    d = read_json(directory / 'decoder.json')
    model = SemanticDecoder(d['family'], d['shape'], d['vocabulary'],
        hidden=d['hidden'], site_mask=d['site_mask']).to(device)
    model.load_state_dict(torch.load(directory / 'weights.pt', map_location=device, weights_only=True))
    model.source_batch_size = config['source_batch_size']
    model.source_program_budget = int(config['program_cache_mib'] * 2**20)
    model.program_namespace = identity['semantic_build_hash']
    projector = None if d['family'] == 'prior' else SiteProjector.load(directory / 'projector')
    if projector is not None and receipt.get('projector_cache_identity') is not None:
        projector.cache_identity = receipt['projector_cache_identity']
    print('DECODER STAGE: committed fitted weights restored; training preparation skipped', flush=True)
    return model, projector, receipt['training_pairing']


def commit_fitted(directory, identity, model, projector, description, pairing):
    directory = Path(directory)
    if projector is not None:
        projector.save(directory / 'projector')
    atomic_torch_save(model.state_dict(), directory / 'weights.pt')
    write_report(directory / 'decoder.json', description)
    files = ['weights.pt', 'decoder.json', 'selection.json']
    if projector is not None:
        files += ['projector/projector.json', 'projector/projector.npz']
    write_report(directory / 'fitted.json', {'run_identity': object_hash(identity),
        'files': {name: file_hash(directory / name) for name in files}, 'training_pairing': pairing,
        'projector_cache_identity': getattr(projector, 'cache_identity', None)})


def attach_prepared_projector(data, fitted, options, train):
    """Imported fitted projectors must use their identical prepared window cache."""
    if fitted is None or hasattr(fitted, 'cache_identity'):
        return
    import numpy as np
    prepared = data.projector(options['modality'], train,
        **{key: options.get(key) for key in ('subject', 'model', 'layer')})
    if (not np.array_equal(prepared.assignment, fitted.assignment) or
            prepared.training_stories != fitted.training_stories or
            prepared.components != fitted.components or len(prepared.parameters) != len(fitted.parameters) or
            any(not np.array_equal(a, b) for first, second in zip(prepared.parameters, fitted.parameters, strict=True)
                for a, b in zip(first, second, strict=True))):
        raise ValueError('Prepared observation projector differs from the imported fitted projector.')
    fitted.cache_identity = prepared.cache_identity


def prediction_stage(directory, name, identity, build):
    """Publish one complete prediction file atomically; interrupted work retries."""
    import gzip
    import json
    directory = Path(directory)
    target = directory / (name + '.jsonl.gz')
    receipt_path = directory / (name + '.complete.json')
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['run_identity'] != object_hash(identity) or file_hash(target) != receipt['sha256']:
            raise ValueError('Prediction stage identity/content mismatch.')
        with gzip.open(target, 'rt', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    rows = build()
    temporary = target.with_suffix('.writing')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(target)
    write_report(receipt_path, {'run_identity': object_hash(identity), 'rows': len(rows), 'sha256': file_hash(target)})
    return rows
