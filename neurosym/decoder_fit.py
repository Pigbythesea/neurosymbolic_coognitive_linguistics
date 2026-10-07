"""Actual answer supervision, nested story selection and auditable heldout traces."""
from collections import defaultdict
import json
import hashlib
import gzip
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .analysis_runs import composition_partition, decoder_metrics, partition, run_directory, write_report
from .compute import FoldCache, Timings, atomic_torch_save, file_hash
from .decoders import SemanticDecoder, acceptable_loss, detached_trace, query_vocabulary
from .decoder_batch import choice_targets, choice_losses
from .runtime import analysis_lock as exclusive_run, deadline
from .io import object_hash, read_json
from .storage import require_free_space
from .protocol import selection_partition, selection_options


def observation_options(options):
    return {key: options.get(key) for key in ("subject", "model", "layer")}


def examples_and_windows(data, stories, projector, options):
    examples, windows, sources = [], {}, {}
    for story in stories:
        current = list(data.examples(story, reviewed_only=options.get("reviewed_only", False)))
        sources.update({s["id"]: s for s in data.semantics.records(story, "sources")})
        if options['family'] == 'prior':
            repeats = data.reader.contract['subjects'][options['subject']][story]['repeats'] if options['modality'] == 'brain' else 1
            current = [{**e, 'observation_repeats': repeats} for e in current
                       if sources[e['source_id']]['decoder_timing_eligible']]
        else:
            windows.update(data.windows(options["modality"], story, projector, **observation_options(options)))
            current = [e for e in current if e['source_id'] in windows]
        examples.extend(current)
    if set(stories) - {e["story_id"] for e in examples}:
        raise ValueError("Every selected story needs actual eligible labelled observations; no empty-fold substitution.")
    used = {e["source_id"] for e in examples}
    return examples, (None if options['family'] == 'prior' else {s: windows[s] for s in used}), {s: sources[s] for s in used}


def mismatch_map(sources, *, cross_story=False):
    """Deterministic temporal mismatch, retaining the observed arrays unchanged.

    Training null: rotate whole stories and match relative endpoint rank.
    Test mismatch: circular endpoint rotation with nonoverlapping delayed TRs.
    Model endpoints still share earlier story prefixes in the latter control.
    """
    by_story = defaultdict(list)
    for key, source in sources.items():
        by_story[source["story_id"]].append(key)
    for keys in by_story.values():
        keys.sort(key=lambda k: sources[k]["unit_index"])
    stories = sorted(by_story)
    result = {}
    if cross_story:
        if len(stories) < 2:
            raise ValueError("Retrained null needs at least two training stories.")
        for index, story in enumerate(stories):
            first, second = by_story[story], by_story[stories[(index + 1) % len(stories)]]
            for rank, key in enumerate(first):
                result[key] = second[min(len(second) - 1, int((rank + .5) * len(second) / len(first)))]
    else:
        for story, keys in by_story.items():
            # Preserve endpoint order/autocorrelation under a circular shift.
            offsets = sorted(range(1, len(keys)), key=lambda n: abs(n - len(keys) / 2))
            chosen = None
            for offset in offsets:
                pairs = [(key, keys[(i + offset) % len(keys)]) for i, key in enumerate(keys)]
                if all(min(abs(a - b) for a in sources[k]["decoder_trimmed_response_rows"]
                           for b in sources[v]["decoder_trimmed_response_rows"]) > 4 for k, v in pairs):
                    chosen = pairs
                    break
            if chosen is None:
                raise ValueError(f"No valid nonoverlapping circular mismatch for {story}; report unavailable control.")
            result.update(chosen)
    return result


def make_decoder(data, examples, windows, projector, options, device):
    torch.manual_seed(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options["seed"])
    shape = None if options['family'] == 'prior' else next(iter(windows.values())).shape[1:]
    key = ('decoder-vocabulary', object_hash([e['query_id'] for e in examples]))
    vocabulary = data.host_cache.get(key, lambda: query_vocabulary(examples))
    model = SemanticDecoder(options["family"], shape, vocabulary,
                            hidden=data.config["decoder"]["hidden"],
                            site_mask=None if projector is None else projector.site_mask).to(device)
    model.source_program_budget = data.config['decoder'].get('program_cache_mib', 1024) * 2**20
    model.source_batch_size = data.config['decoder']['source_batch_size']
    model.program_namespace = data.semantics.build_hash
    return model, vocabulary


class DeviceWindows(dict):
    """Actual observations checked once before staging, never fabricated inputs."""
    def __init__(self, windows, device):
        super().__init__()
        self.device = torch.device(device)
        for key, values in windows.items():
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite actual observations: ' + key)
            self[key] = torch.as_tensor(np.ascontiguousarray(values), device=device)


def device_windows(windows, device):
    if windows is None:
        return None
    if isinstance(windows, DeviceWindows):
        if windows.device != torch.device(device):
            raise ValueError('Staged observations belong to another device.')
        return windows
    return DeviceWindows(windows, device)


def evaluate(model, examples, windows, *, pairing=None, trace_file=None, probabilities=True, free_reserve_bytes=0,
             trace_format='legacy'):
    model.eval()
    device = next(model.parameters()).device
    windows = None if model.family == 'prior' else device_windows(windows, device)
    if trace_file is not None and trace_format == 'compact':
        if pairing is not None or model.family != 'structured':
            raise ValueError('Compact grounding traces require matched structured observations.')
        from .trace_store import export
        return export(model, examples, windows, trace_file, probabilities=probabilities, free_reserve_bytes=free_reserve_bytes)
    if trace_format not in {'legacy', 'compact'}:
        raise ValueError('Unknown trace format.')
    if trace_file is None and model.source_batched:
        from .decoder_minibatch import evaluate_batched
        return evaluate_batched(model, examples, windows, pairing=pairing, probabilities=probabilities,
                                size=getattr(model, 'source_batch_size', 16))
    by_source, rows = defaultdict(list), []
    for example in examples:
        by_source[example["source_id"]].append(example)
    with torch.no_grad():
        for source_index, (source, queries) in enumerate(by_source.items()):
            if trace_file is not None and free_reserve_bytes and source_index % 32 == 0:
                require_free_space(Path(trace_file.filename).parent, 0, free_reserve_bytes)
            observed_source = pairing[source] if pairing else source
            repeats = queries[0].get('observation_repeats', 1) if windows is None else len(windows[observed_source])
            for repeat in range(repeats):
                if trace_file is not None:
                    paths = [e['query_id'] + f'/{repeat}' for e in queries]
                    if all(p in trace_file and trace_file[p].attrs.get('prediction_complete', False) for p in paths):
                        rows.extend(json.loads(bytes(trace_file[p]['prediction_record'][()]).decode('utf-8')) for p in paths)
                        continue
                    # A yielded source is committed as a unit. A partial source
                    # from an interrupted write is recomputed, never trusted.
                    for p in paths:
                        if p in trace_file:
                            del trace_file[p]
                trace_arrays = {}  # identical arrays share HDF storage within this source/repeat
                model.descriptors.begin_source()
                observed = model.encode_observation(None if windows is None else windows[observed_source][repeat], checked=True)
                if model.source_batched and trace_file is None:
                    targets = choice_targets(queries, device)
                    logits = model.answer_many(observed, [e['inputs'] for e in queries])
                    losses = choice_losses(logits, targets)
                    correct = ((logits.argmax(-1)[:, None] == targets['indices']) & targets['valid']).any(-1)
                    measured = torch.stack((correct.to(logits.dtype), losses), dim=-1).cpu().numpy()
                    predicted = logits.softmax(-1).cpu().numpy() if probabilities else None
                    for index, example in enumerate(queries):
                        row = {k: example[k] for k in ('query_id', 'source_id', 'story_id', 'family', 'weight', 'annotation_review_status')}
                        count = len(example['inputs']['candidates'])
                        row.update(repeat=repeat, observation_source=observed_source,
                                   acceptable_indices=example['acceptable_indices'],
                                   chance=len(example['acceptable_indices']) / count,
                                   correct=float(measured[index, 0]), nll=float(measured[index, 1]))
                        if probabilities:
                            row['probabilities'] = predicted[index, :count].tolist()
                        rows.append(row)
                    model.descriptors.end_source()
                    model.end_observation()
                    continue
                pending, metrics, probability_blocks = [], [], []
                for example in queries:
                    logits, trace = model.answer(observed, example["inputs"], capture=trace_file is not None)
                    acceptable = torch.tensor(example['acceptable_indices'], device=device)
                    correct = (logits.argmax() == acceptable).any().to(logits.dtype)
                    metrics.append(torch.stack([correct, acceptable_loss(logits, example['acceptable_indices'])]))
                    if probabilities:
                        probability_blocks.append(logits.softmax(-1))
                    row = {k: example[k] for k in ("query_id", "source_id", "story_id", "family", "weight", "annotation_review_status")}
                    row.update(repeat=repeat, observation_source=observed_source,
                               acceptable_indices=example["acceptable_indices"],
                               chance=len(example["acceptable_indices"]) / len(logits))
                    if trace_file is not None:
                        group = trace_file.create_group(example["query_id"] + f"/{repeat}")
                        group.attrs.update(source_id=source, story_id=example["story_id"], family=example["family"],
                                           public_input_query_id=example['query_id'],
                                           public_input_origin='compiled query and candidate catalogs pinned by complete.json')
                        write_trace(group, detached_trace(trace), shared_arrays=trace_arrays)
                    pending.append(row)
                # Validation transfers only scalar metrics, once per source/repeat.
                measured = torch.stack(metrics).cpu().numpy()
                flat = torch.cat(probability_blocks).cpu().numpy() if probabilities else None
                offset = 0
                for index, row in enumerate(pending):
                    row.update(correct=float(measured[index, 0]), nll=float(measured[index, 1]))
                    if probabilities:
                        count = len(probability_blocks[index])
                        row['probabilities'] = flat[offset:offset + count].tolist()
                        offset += count
                    rows.append(row)
                    if trace_file is not None:
                        group = trace_file[row['query_id'] + f'/{repeat}']
                        payload = np.frombuffer(json.dumps(row, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
                        group.create_dataset('prediction_record', data=payload, compression='lzf')
                        group.attrs['prediction_complete'] = True
                model.descriptors.end_source()
                model.end_observation()
                if trace_file is not None:
                    trace_file.flush()
                    deadline.advance()
                    deadline.check()
    return rows


def write_trace(group, trace, *, shared_arrays=None):
    for key, value in trace.items():
        if isinstance(value, np.ndarray):
            identity = (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).digest())
            if shared_arrays is not None and identity in shared_arrays:
                group[key] = group.file[shared_arrays[identity]]
            else:
                dataset = group.create_dataset(key, data=value, compression="lzf" if value.ndim else None)
                if shared_arrays is not None:
                    shared_arrays[identity] = dataset.name
        elif isinstance(value, list):
            children = group.create_group(key)
            for index, item in enumerate(value):
                write_trace(children.create_group(str(index)), item, shared_arrays=shared_arrays)
        else:
            group.attrs[key] = json.dumps(value, ensure_ascii=False)


def fit_identity(data, model, examples, windows, validation, **settings):
    def describe(examples, windows):
        records = [{k: e.get(k) for k in ('query_id', 'source_id', 'story_id', 'family', 'weight',
                                          'acceptable_indices', 'observation_repeats')} for e in examples]
        observations = None if model.family == 'prior' else {
            k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k, v in windows.items()}
        return {'queries': object_hash(records), 'observations': observations}
    return {'semantic_build': data.semantics.build_hash, 'family': model.family, 'shape': list(model.shape),
            'vocabulary': model.descriptors.vocabulary, 'hidden': model.hidden, 'torch': str(torch.__version__),
            'device': str(next(model.parameters()).device), 'decoder_config': data.config['decoder'],
            'train': describe(examples, windows),
            'validation': describe(*validation) if validation else None, 'settings': settings,
            'code': {n: file_hash(Path(__file__).parent / n) for n in ('decoder_fit.py', 'decoders.py', 'decoder_batch.py', 'decoder_minibatch.py', 'protocol.py', 'compute.py')}}


def train_decoder(data, model, examples, windows, *, learning_rate, epochs, seed,
                  validation=None, pairing=None, patience=None, checkpoint=None, reuse_prior=True):
    cfg = data.config["decoder"]
    identity = fit_identity(data, model, examples, windows, validation, learning_rate=learning_rate,
                            epochs=epochs, seed=seed, pairing=pairing, patience=patience)
    if model.family == 'prior' and reuse_prior:
        def build(path):
            result = train_decoder(data, model, examples, windows, learning_rate=learning_rate, epochs=epochs,
                                   seed=seed, validation=validation, pairing=pairing, patience=patience,
                                   checkpoint=checkpoint, reuse_prior=False)
            np.savez(path / 'weights.npz', **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
            write_report(path / 'fit.json', result)
        cache = FoldCache(data.cache.root, policy=data.cache.policy)
        path = cache.entry('prior-fit', identity, build)
        with np.load(path / 'weights.npz', allow_pickle=False) as file:
            model.load_state_dict({k: torch.from_numpy(file[k]) for k in file.files})
        return {**read_json(path / 'fit.json'), 'shared_prior_fit': path.name, 'cache_hit': bool(cache.hits)}
    if cfg.get('source_batch_size', 1) > 1:
        from .decoder_minibatch import train_minibatches
        return train_minibatches(data, model, examples, windows, identity=identity,
            learning_rate=learning_rate, epochs=epochs, seed=seed, validation=validation,
            pairing=pairing, patience=patience, checkpoint=checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg["weight_decay"])
    device, groups = next(model.parameters()).device, defaultdict(list)
    windows = None if model.family == 'prior' else device_windows(windows, device)
    if validation:
        validation = (validation[0], None if model.family == 'prior' else device_windows(validation[1], device))
    for example in examples:
        groups[example["source_id"]].append(example)
    supervision = {source: choice_targets(queries, device) for source, queries in groups.items()} if model.source_batched else {}
    story_counts = defaultdict(int)
    for queries in groups.values():
        story_counts[queries[0]["story_id"]] += 1
    rng, history, best, best_state, stale = np.random.default_rng(seed), [], float("inf"), None, 0
    pending = None
    best_epoch, completed = 0, 0
    if checkpoint and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint inputs, precision, code or optimizer settings changed.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, best, best_state, stale, best_epoch, completed = [saved[k] for k in
            ('history', 'best', 'best_state', 'stale', 'best_epoch', 'completed')]
        pending = saved.get('pending_epoch')
        print('DECODER RESUME completed_epochs=' + str(completed), flush=True)

    def save_progress(pending_epoch=None):
        if checkpoint:
            atomic_torch_save({'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                'history': history, 'best': best, 'best_state': best_state, 'stale': stale,
                'best_epoch': best_epoch, 'completed': completed, 'pending_epoch': pending_epoch}, checkpoint)
    for epoch in range(completed + 1, epochs + 1):
        if validation is not None and patience is not None and stale >= patience:
            break
        started = time.perf_counter()
        previous_seconds = 0.
        model.train()
        if pending is not None:
            if pending['epoch'] != epoch or set(pending['order']) != set(groups):
                raise ValueError('Interrupted epoch source order does not match this fit.')
            order, cursor = pending['order'], pending['cursor']
            previous_seconds = pending.get('elapsed_seconds', 0.)
            losses = [torch.as_tensor(v, device=device) for v in pending['losses']]
            pending = None
        else:
            order, cursor, losses = rng.permutation(list(groups)).tolist(), 0, []
        for source_index in range(cursor, len(order)):
            source = order[source_index]
            queries = groups[source]
            values = (range(queries[0].get('observation_repeats', 1)) if windows is None else
                      windows[pairing[source] if pairing else source])
            optimizer.zero_grad(set_to_none=True)
            model.descriptors.begin_source()
            source_loss = torch.zeros((), device=device)
            for x in values:
                observed = model.encode_observation(None if windows is None else x, checked=True)
                if model.source_batched:
                    logits = model.answer_many(observed, [example['inputs'] for example in queries])
                    targets = supervision[source]
                    source_loss = source_loss + (targets['weights'] * choice_losses(logits, targets)).sum() / len(values)
                else:
                    for example in queries:
                        logits, _ = model.answer(observed, example["inputs"])
                        source_loss = source_loss + example["weight"] * acceptable_loss(logits, example["acceptable_indices"]) / len(values)
            # Equal expected total contribution per story across an epoch.
            weight = len(groups) / (len(story_counts) * story_counts[queries[0]["story_id"]])
            if not torch.isfinite(source_loss):
                raise ValueError("Nonfinite decoder loss; stopping the actual fit.")
            (source_loss * weight).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            model.descriptors.end_source()
            model.end_observation()
            losses.append(source_loss.detach())
            deadline.advance()
            if deadline.due() and checkpoint:
                save_progress({'epoch': epoch, 'order': order, 'cursor': source_index + 1,
                               'losses': torch.stack(losses).cpu().tolist(),
                               'elapsed_seconds': previous_seconds + time.perf_counter() - started})
                deadline.check()
        record = {"epoch": epoch, "train_source_mean_nll": float(torch.stack(losses).cpu().numpy().astype(np.float64).mean())}
        record.update(training_seconds=previous_seconds + time.perf_counter() - started,
                      train_sources=len(groups), train_queries=len(examples),
                      query_execution='source-batched' if model.source_batched else 'scalar-reference',
                      source_program_cache_bytes=model._source_program_bytes)
        validation_started = time.perf_counter()
        if validation is not None:
            val_examples, val_windows = validation
            value = decoder_metrics(evaluate(model, val_examples, val_windows, probabilities=False))["story_macro"]["nll"]
            record["validation_story_macro_nll"] = value
            if value < best:
                best, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                stale += 1
        record['validation_seconds'] = time.perf_counter() - validation_started if validation is not None else 0.
        record['wall_seconds'] = previous_seconds + time.perf_counter() - started
        history.append(record)
        completed = epoch
        save_progress()
        if checkpoint:
            deadline.check()
        print("DECODER EPOCH", json.dumps(record), flush=True)
        if validation is not None and patience is not None and stale >= patience:
            break
    if validation is not None:
        if best_state is None:
            raise ValueError("Validation produced no selectable model.")
        model.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "best_validation_nll": best, "history": history}
    return {"best_epoch": epochs, "history": history}


def select_decoder(data, options):
    options = selection_options(data, options)
    split = selection_partition(data, options)
    data.validate_partition(split['train'], test=split['test'])
    cfg, device = data.config['decoder'], options.get('device', 'cpu')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for decoder selection but unavailable.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    directory, identity = run_directory(data, 'decoder-selection', options)
    timings = Timings(device)
    with exclusive_run(directory / 'RUNNING.lock'):
        if (directory / 'complete.json').exists():
            return directory
        if options['family'] == 'linear':
            result = {'selected_learning_rate': cfg['linear_policy']['learning_rate'],
                      'refit_epochs': cfg['linear_policy']['epochs'], 'selection': 'fixed-linear-protocol'}
            write_report(directory / 'complete.json', {'identity': identity, 'partition': split, **result})
            return directory
        scores = [[] for _ in cfg["learning_rates"]]
        for index, fold in enumerate(split["inner"]):
            selection_path = directory / f"inner-{index}.json"
            if selection_path.exists():
                results = read_json(selection_path)
            else:
                with timings.phase('observation_preparation'):
                    projector = (None if options['family'] == 'prior' else
                                 data.projector(options["modality"], fold["train"], **observation_options(options)))
                    train, windows, sources = examples_and_windows(data, fold["train"], projector, options)
                    validation, val_windows, _ = examples_and_windows(data, fold["validation"], projector, options)
                pairing = mismatch_map(sources, cross_story=True) if options.get("retrain_null") else None
                results = []
                for lr_index, learning_rate in enumerate(cfg["learning_rates"]):
                    print(f"DECODER SELECT inner={index + 1}/{len(split['inner'])} lr={learning_rate}", flush=True)
                    model, _ = make_decoder(data, train, windows, projector, options, device)
                    with timings.phase('inner_training_and_validation'):
                        results.append(train_decoder(data, model, train, windows, learning_rate=learning_rate,
                                       epochs=cfg["epochs"], seed=options["seed"], validation=(validation, val_windows),
                                       pairing=pairing, patience=cfg["patience"],
                                       checkpoint=directory / f'inner-{index}-lr-{lr_index}.pt'))
                write_report(selection_path, results)
                for lr_index in range(len(cfg['learning_rates'])):
                    (directory / f'inner-{index}-lr-{lr_index}.pt').unlink(missing_ok=True)
            for group, result in zip(scores, results, strict=True):
                group.append(result)
        mean_losses = [np.average([v['best_validation_nll'] for v in s],
                                 weights=[len(f['validation']) for f in split['inner']]) for s in scores]
        selected = int(np.argmin(mean_losses))
        epochs = max(1, int(round(np.median([v["best_epoch"] for v in scores[selected]]))))
        learning_rate = cfg["learning_rates"][selected]
        write_report(directory / 'complete.json', {'identity': identity, 'partition': split,
            'mean_inner_nll': mean_losses, 'selected_learning_rate': learning_rate,
            'refit_epochs': epochs, 'selection_seed': cfg['selection_seed'],
            'epoch_policy': 'median selected inner-fold epoch; no test inspection',
            'shared_across_evaluation_seeds': True, 'runtime': timings.report()})
    return directory


def run_decoder(data, options):
    timings = Timings(options.get('device', 'cpu'))
    split = selection_partition(data, options)
    data.validate_partition(split["train"], test=split["test"])
    if options["modality"] == "model" and options.get("subject"):
        raise ValueError("Model-only decoder runs must not be duplicated by participant.")
    device = options.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no silent device substitution.")
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    directory, identity = run_directory(data, "decoder", options)
    free_reserve = data.compute_config['storage']['minimum_free_gib'] * 2**30
    require_free_space(directory, 0, free_reserve)
    cfg = data.config["decoder"]
    with exclusive_run(directory / "RUNNING.lock"), timings.attempts(directory):
        if (directory / "complete.json").exists():
            return directory
        from .decoder_stages import load_fitted, commit_fitted, prediction_stage, attach_prepared_projector
        restored = load_fitted(directory, identity, device, cfg)
        if restored is None:
            selected_path = select_decoder(data, options)
            selected = read_json(selected_path / 'complete.json')
            learning_rate, epochs = selected['selected_learning_rate'], selected['refit_epochs']
            write_report(directory / 'selection.json', {'parent': str(selected_path),
                'selected_learning_rate': learning_rate, 'refit_epochs': epochs,
                'null_policy': 'matched-settings training-correspondence ablation' if options.get('retrain_null') else None})
            with timings.phase('training_observation_preparation'):
                projector = (None if options['family'] == 'prior' else
                             data.projector(options['modality'], split['train'], **observation_options(options)))
                train, windows, sources = examples_and_windows(data, split['train'], projector, options)
            model, vocabulary = make_decoder(data, train, windows, projector, options, device)
            pairing = mismatch_map(sources, cross_story=True) if options.get('retrain_null') else None
            with timings.phase('refit'):
                fit = train_decoder(data, model, train, windows, learning_rate=learning_rate, epochs=epochs,
                                    seed=options['seed'], pairing=pairing, checkpoint=directory / 'refit.pt')
            commit_fitted(directory, identity, model, projector, {
                'family': model.family, 'shape': model.shape, 'vocabulary': vocabulary,
                'hidden': model.hidden, 'parameters': sum(p.numel() for p in model.parameters()),
                'training_stories': split['train'], 'training_source_ids': sorted(sources),
                'site_mask': model.site_mask.cpu().numpy(), 'fit': fit,
                'site_coordinates': ('none: query-only prior' if options['family'] == 'prior' else
                    'Schaefer parcels' if options['modality'] == 'brain' else 'seeded model-coordinate groups, not anatomical regions')}, pairing)
            del train, windows
        else:
            model, projector, pairing = restored
            attach_prepared_projector(data, projector, options, split['train'])
        with timings.phase('observation_preparation'):
            test, test_windows, test_sources = examples_and_windows(data, split["test"], projector, options)
        if "test_query_ids" in split:
            query_ids = set(split["test_query_ids"])
            test = [e for e in test if e["query_id"] in query_ids]
            if not test:
                raise ValueError("No scoring-eligible composition queries after ambiguity/review exclusions.")
            # Recompute source/family weights after selecting exact heldout events.
            groups = defaultdict(lambda: defaultdict(list))
            for example in test:
                groups[example["source_id"]][example["family"]].append(example)
            for families in groups.values():
                for examples in families.values():
                    for example in examples:
                        example["weight"] = 1 / (len(families) * len(examples))
        if options.get('export_traces', options['family'] == 'structured' and not options.get('retrain_null')):
            with h5py.File(directory / 'traces.h5', 'a') as file:
                if len(file) and file.attrs.get('run_identity') != object_hash(identity):
                    raise ValueError('Partial trace file belongs to another fit.')
                file.attrs.update(complete=False, semantic_build_hash=data.semantics.build_hash)
                file.attrs['run_identity'] = object_hash(identity)
                with timings.phase('heldout_predictions_and_traces'):
                    rows = evaluate(model, test, test_windows, trace_file=file, free_reserve_bytes=free_reserve,
                                    trace_format='compact')
                file.attrs['complete'] = True
        else:
            with timings.phase('heldout_predictions'):
                rows = evaluate(model, test, test_windows)
        rows = prediction_stage(directory, 'predictions', identity, lambda: rows)
        mismatch = mismatch_map(test_sources)
        with timings.phase('mismatched_predictions'):
            disrupted = prediction_stage(directory, 'mismatched-predictions', identity,
                lambda: evaluate(model, test, test_windows, pairing=mismatch))
        if options.get('faithfulness'):
            from .grounding_checks import faithfulness
            with timings.phase('grounding_faithfulness'):
                write_report(directory / 'faithfulness.json', faithfulness(model, test, test_windows, test_sources,
                    fraction=cfg['faithfulness_fraction'], seed=options['seed'], checkpoint=directory / 'faithfulness-progress.json'))
        report = {"identity": identity, "partition": split, "matched": decoder_metrics(rows),
                  'evaluation_support_hash': object_hash(sorted([
                      {k: e[k] for k in ['query_id', 'source_id', 'story_id', 'family', 'weight', 'acceptable_indices']}
                      for e in test], key=lambda e: e['query_id'])),
                  "test_mismatch": decoder_metrics(disrupted), "test_pairing": mismatch,
                  'prediction_files': ['predictions.jsonl.gz', 'mismatched-predictions.jsonl.gz'],
                  'traces_exported': bool(options.get('export_traces', options['family'] == 'structured' and not options.get('retrain_null'))),
                  'faithfulness_file': 'faithfulness.json' if options.get('faithfulness') else None,
                  "training_pairing": pairing, "torch_version": torch.__version__, "device": str(device),
                  "interpretation": "Answer-supervised decoder evidence; no anatomical grounding labels or biological-necessity claim.",
                  "mismatch_scope": "Within-story nonoverlapping delayed windows. Model states retain shared earlier prefixes; retrained cross-story null is a separate run.",
                  "trace_scope": "Low-rank ordered-pair factors reconstruct all raw pair scores; invalid sites must be masked. Latents are comparable only within a fitted coordinate system."}
        write_report(directory / "complete.json", report)
        (directory / 'refit.pt').unlink(missing_ok=True)
        (directory / 'faithfulness-progress.json').unlink(missing_ok=True)
    return directory


def prepare_decoder(data, options):
    """Create real training-fitted dependencies with the configured PCA backend."""
    split = selection_partition(data, options)
    data.validate_partition(split['train'], test=split['test'])
    timings = Timings(data.compute_config['preparation']['device'])
    prepared = []
    inner = [] if options.get('outer_only') else split['inner']
    for fold in [*inner, {'train': split['train'], 'validation': split['test']}]:
        deadline.check()
        with timings.phase('projectors_and_windows'):
            projector = data.projector(options['modality'], fold['train'], **observation_options(options))
            for story in fold['train'] + fold['validation']:
                data.windows(options['modality'], story, projector, **observation_options(options))
        prepared.append({'train': fold['train'], 'transformed': fold['train'] + fold['validation']})
    directory, identity = run_directory(data, 'preparation', options)
    write_report(directory / 'complete.json', {'identity': identity, 'prepared': prepared,
        'runtime': timings.report(), 'cache_hits': data.cache.hits, 'cache_misses': data.cache.misses,
        'scientific_fits_executed': False, 'scope': 'Training-only local PCA and actual observation windows.'})
    return directory
