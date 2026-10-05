"""Actual answer supervision, nested story selection and auditable heldout traces."""
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .analysis_runs import composition_partition, decoder_metrics, partition, run_directory, write_report
from .decoders import SemanticDecoder, acceptable_loss, detached_trace, query_vocabulary
from .extraction import exclusive_run
from .io import object_hash, read_json


def observation_options(options):
    return {key: options.get(key) for key in ("subject", "model", "layer")}


def examples_and_windows(data, stories, projector, options):
    examples, windows, sources = [], {}, {}
    for story in stories:
        examples.extend(data.examples(story, reviewed_only=options.get("reviewed_only", False)))
        windows.update(data.windows(options["modality"], story, projector, **observation_options(options)))
        sources.update({s["id"]: s for s in data.semantics.records(story, "sources")})
    examples = [e for e in examples if e["source_id"] in windows]
    if set(stories) - {e["story_id"] for e in examples}:
        raise ValueError("Every selected story needs actual eligible labelled observations; no empty-fold substitution.")
    used = {e["source_id"] for e in examples}
    return examples, {s: windows[s] for s in used}, {s: sources[s] for s in used}


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
    shape = next(iter(windows.values())).shape[1:]
    vocabulary = query_vocabulary(examples)
    model = SemanticDecoder(options["family"], shape, vocabulary,
                            hidden=data.config["decoder"]["hidden"], site_mask=projector.site_mask).to(device)
    return model, vocabulary


def evaluate(model, examples, windows, *, pairing=None, trace_file=None):
    model.eval()
    device = next(model.parameters()).device
    by_source, rows = defaultdict(list), []
    for example in examples:
        by_source[example["source_id"]].append(example)
    with torch.no_grad():
        for source, queries in by_source.items():
            observed_source = pairing[source] if pairing else source
            for repeat, x in enumerate(windows[observed_source]):
                model.descriptors.begin_source()
                observed = model.encode_observation(torch.as_tensor(x, device=device))
                for example in queries:
                    logits, trace = model.answer(observed, example["inputs"], capture=trace_file is not None)
                    probabilities = logits.softmax(-1).cpu().numpy()
                    row = {k: example[k] for k in ("query_id", "source_id", "story_id", "family", "weight", "annotation_review_status")}
                    row.update(repeat=repeat, observation_source=observed_source,
                               probabilities=probabilities.tolist(), acceptable_indices=example["acceptable_indices"],
                               correct=float(int(probabilities.argmax()) in example["acceptable_indices"]),
                               nll=float(acceptable_loss(logits, example["acceptable_indices"])),
                               chance=len(example["acceptable_indices"]) / len(probabilities))
                    if trace_file is not None:
                        group = trace_file.create_group(example["query_id"] + f"/{repeat}")
                        group.attrs.update(source_id=source, story_id=example["story_id"], family=example["family"],
                                           query_json=json.dumps(example["inputs"], ensure_ascii=False))
                        write_trace(group, detached_trace(trace))
                    rows.append(row)
                model.descriptors.end_source()
    return rows


def write_trace(group, trace):
    for key, value in trace.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression="lzf" if value.ndim else None)
        elif isinstance(value, list):
            children = group.create_group(key)
            for index, item in enumerate(value):
                write_trace(children.create_group(str(index)), item)
        else:
            group.attrs[key] = json.dumps(value, ensure_ascii=False)


def train_decoder(data, model, examples, windows, *, learning_rate, epochs, seed,
                  validation=None, pairing=None, patience=None):
    cfg = data.config["decoder"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg["weight_decay"])
    device, groups = next(model.parameters()).device, defaultdict(list)
    for example in examples:
        groups[example["source_id"]].append(example)
    story_counts = defaultdict(int)
    for queries in groups.values():
        story_counts[queries[0]["story_id"]] += 1
    rng, history, best, best_state, stale = np.random.default_rng(seed), [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for source in rng.permutation(list(groups)):
            queries = groups[source]
            values = windows[pairing[source] if pairing else source]
            optimizer.zero_grad(set_to_none=True)
            model.descriptors.begin_source()
            source_loss = torch.zeros((), device=device)
            for x in values:
                observed = model.encode_observation(torch.as_tensor(x, device=device))
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
            losses.append(float(source_loss.detach()))
        record = {"epoch": epoch, "train_source_mean_nll": float(np.mean(losses))}
        if validation is not None:
            val_examples, val_windows = validation
            value = decoder_metrics(evaluate(model, val_examples, val_windows))["story_macro"]["nll"]
            record["validation_story_macro_nll"] = value
            if value < best:
                best, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                stale += 1
        history.append(record)
        print("DECODER EPOCH", json.dumps(record), flush=True)
        if validation is not None and patience is not None and stale >= patience:
            break
    if validation is not None:
        if best_state is None:
            raise ValueError("Validation produced no selectable model.")
        model.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "best_validation_nll": best, "history": history}
    return {"best_epoch": epochs, "history": history}


def run_decoder(data, options):
    split = partition(data.semantics, options["fold"])
    if options.get("composition_keys"):
        split = composition_partition(data.semantics, split, options["composition_keys"])
    data.validate_partition(split["train"], test=split["test"])
    if options["modality"] == "model" and options.get("subject"):
        raise ValueError("Model-only decoder runs must not be duplicated by participant.")
    device = options.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no silent device substitution.")
    directory, identity = run_directory(data, "decoder", options)
    cfg = data.config["decoder"]
    with exclusive_run(directory / "RUNNING.lock"):
        if (directory / "complete.json").exists():
            return directory
        scores = [[] for _ in cfg["learning_rates"]]
        for index, fold in enumerate(split["inner"]):
            selection_path = directory / f"inner-{index}.json"
            if selection_path.exists():
                results = read_json(selection_path)
            else:
                projector = data.projector(options["modality"], fold["train"], **observation_options(options))
                train, windows, sources = examples_and_windows(data, fold["train"], projector, options)
                validation, val_windows, _ = examples_and_windows(data, fold["validation"], projector, options)
                pairing = mismatch_map(sources, cross_story=True) if options.get("retrain_null") else None
                results = []
                for learning_rate in cfg["learning_rates"]:
                    print(f"DECODER SELECT inner={index + 1}/{len(split['inner'])} lr={learning_rate}", flush=True)
                    model, _ = make_decoder(data, train, windows, projector, options, device)
                    results.append(train_decoder(data, model, train, windows, learning_rate=learning_rate,
                                   epochs=cfg["epochs"], seed=options["seed"], validation=(validation, val_windows),
                                   pairing=pairing, patience=cfg["patience"]))
                write_report(selection_path, results)
            for group, result in zip(scores, results, strict=True):
                group.append(result)
        mean_losses = [np.mean([v["best_validation_nll"] for v in s]) for s in scores]
        selected = int(np.argmin(mean_losses))
        epochs = max(1, int(round(np.median([v["best_epoch"] for v in scores[selected]]))))
        learning_rate = cfg["learning_rates"][selected]
        write_report(directory / "selection.json", {"mean_inner_nll": mean_losses, "selected_learning_rate": learning_rate,
                     "refit_epochs": epochs, "epoch_policy": "median selected inner-fold epoch; no test inspection"})
        projector = data.projector(options["modality"], split["train"], **observation_options(options))
        train, windows, sources = examples_and_windows(data, split["train"], projector, options)
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
        model, vocabulary = make_decoder(data, train, windows, projector, options, device)
        pairing = mismatch_map(sources, cross_story=True) if options.get("retrain_null") else None
        fit = train_decoder(data, model, train, windows, learning_rate=learning_rate, epochs=epochs,
                            seed=options["seed"], pairing=pairing)
        projector.save(directory / "projector")
        torch.save(model.state_dict(), directory / "weights.pt")
        write_report(directory / "decoder.json", {"family": model.family, "shape": model.shape, "vocabulary": vocabulary,
                     "hidden": model.hidden, "parameters": sum(p.numel() for p in model.parameters()),
                     "training_stories": split["train"], "training_source_ids": sorted(windows),
                     "site_mask": model.site_mask.cpu().numpy(), "fit": fit,
                     "site_coordinates": "Schaefer parcels" if options["modality"] == "brain" else "seeded model-coordinate groups, not anatomical regions"})
        with h5py.File(directory / "traces.h5", "w") as file:
            file.attrs.update(complete=False, semantic_build_hash=data.semantics.build_hash)
            rows = evaluate(model, test, test_windows, trace_file=file)
            file.attrs["complete"] = True
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        mismatch = mismatch_map(test_sources)
        disrupted = evaluate(model, test, test_windows, pairing=mismatch)
        with (directory / "mismatched-predictions.jsonl").open("w", encoding="utf-8") as file:
            for row in disrupted:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {"identity": identity, "partition": split, "matched": decoder_metrics(rows),
                  "test_mismatch": decoder_metrics(disrupted), "test_pairing": mismatch,
                  "training_pairing": pairing, "torch_version": torch.__version__, "device": str(device),
                  "interpretation": "Answer-supervised decoder evidence; no anatomical grounding labels or biological-necessity claim.",
                  "mismatch_scope": "Within-story nonoverlapping delayed windows. Model states retain shared earlier prefixes; retrained cross-story null is a separate run.",
                  "trace_scope": "Low-rank ordered-pair factors reconstruct all raw pair scores; invalid sites must be masked. Latents are comparable only within a fitted coordinate system."}
        write_report(directory / "complete.json", report)
    return directory
