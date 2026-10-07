"""Resolve the scientific execution definitions without running or submitting fits."""
from .analysis_runs import partition
from .protocol import selection_partition
from .encoding_support import comparison_spec
from .io import object_hash, read_json


def experiment_plan(data):
    plan = read_json(data.root / "configs/experiments.json")
    if plan["format_version"] != 1 or plan["primary_layer"] != "final":
        raise ValueError("Unsupported experiment-definition version/layer policy.")
    if plan.get('protocol_version') != 2 or data.config.get('protocol_version') != 2:
        raise ValueError('Scientific definitions and analysis must both use protocol 2.')
    cfg = data.config['decoder']
    if cfg['selection_seed'] not in cfg['seeds'] or cfg['source_batch_size'] < 2:
        raise ValueError('A declared tuning seed and multi-source execution are required.')
    if plan['decoder']['primary_families'] != ['linear', 'mlp', 'structured'] or plan['decoder']['descriptive_families'] != ['linear']:
        raise ValueError('Primary and descriptive readout policies differ from protocol 2.')
    if cfg['selection_folds'] != 3 or data.config['encoding']['selection_folds'] != 3:
        raise ValueError('Protocol 2 uses three whole-story selection folds.')
    if set(plan["subjects"]) != set(data.reader.contract["subjects"]):
        raise ValueError("Experiment participant panel differs from the data contract.")
    if plan["seeds"] != data.config["decoder"]["seeds"]:
        raise ValueError("Decoder and experiment seed definitions differ.")
    if plan["development_folds"] != [str(i) for i in range(len(data.semantics.splits["outer_folds"]))]:
        raise ValueError("Experiment definitions omit or duplicate development folds.")
    if plan["final_fold"] != "final":
        raise ValueError("The final evaluation must retain the registered holdout.")
    partitions = {fold: selection_partition(data, {'fold': fold}) for fold in
                  plan["development_folds"] + [plan["final_fold"]] + plan["geometry"]["context_folds"]}
    if any("story_11" in s["train"] for s in partitions.values()):
        raise ValueError("Final story entered a training partition.")
    for contrast in plan["encoding_contrasts"]:
        specs = [comparison_spec(data.config, {"comparison": contrast["comparison"], "groups": contrast[side]})
                 for side in ("baseline", "augmented")]
        if specs[0] != specs[1] or not set(contrast["baseline"]) < set(contrast["augmented"]):
            raise ValueError("Encoding contrast is not nested on identical declared support.")
    extraction = read_json(data.root / "configs/extraction.json")
    lock = read_json(data.root / extraction["model_lock"])
    if lock["content_hash"] != object_hash({k: v for k, v in lock.items() if k != "content_hash"}):
        raise ValueError("Frozen-model lock changed.")
    models = {m["id"]: m for m in lock["models"]}
    if plan["models"] != [m["id"] for m in extraction["models"]] or set(models) != set(plan["models"]):
        raise ValueError("Scientific model panel differs from the extraction configuration/lock.")
    resolved = []
    for name in plan["models"]:
        model = models[name]
        depth = model["num_hidden_layers"]
        fractions = plan["descriptive_layer_fractions"]
        if any(not 0 < f <= 1 for f in fractions):
            raise ValueError("Layer fractions must lie in (0,1].")
        resolved.append({"id": name, "repo": model["repo"], "revision": model["revision"],
                         "primary_layer": depth, "descriptive_layers": sorted({max(1, round(depth * f)) for f in fractions})})
        for groups in plan["model_encoding"]["conditions"]:
            options = {"comparison": plan["model_encoding"]["comparison"], "groups": groups,
                       "mask_models": [name + ":" + str(depth)]}
            if "model" in groups:
                options.update(model=name, layer=depth)
            comparison_spec(data.config, options)
    if plan["decoder"]["families"] != ["prior", "linear", "mlp", "structured"]:
        raise ValueError("Experiment definitions must retain the four comparator readouts.")
    return {"status": "resolved_not_executed", "semantic_build_hash": data.semantics.build_hash,
            "definition_hash": object_hash(plan), "analysis_config_hash": object_hash(data.config),
            "model_lock_hash": lock["content_hash"], "definitions": plan, "models": resolved,
            "partitions": partitions, "scientific_fits_executed": False,
            "layer_indexing": "0 is the embedding output; N is the final block output for an N-block model.",
            "execution": "Definitions and dependencies only. No remote access, job submission, or final-story release is performed."}
