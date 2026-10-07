"""Versioned selection rules, distinct from the outer scientific evaluations."""
from .analysis_runs import partition, composition_partition


def selection_partition(data, options, *, decoder=True):
    split = partition(data.semantics, options['fold'])
    if options.get('composition_keys'):
        split = composition_partition(data.semantics, split, options['composition_keys'])
    count = data.config['decoder' if decoder else 'encoding']['selection_folds']
    stories = sorted(split['train'])
    if count < 2 or count > len(stories):
        raise ValueError('Selection requires 2..number-of-training-stories folds.')
    # Whole stories, deterministic and independent of responses or effect sizes.
    validation = [stories[i::count] for i in range(count)]
    inner = [{'train': [s for s in stories if s not in held], 'validation': held}
             for held in validation]
    if any(len(f['train']) < 2 for f in inner):
        raise ValueError('Insufficient whole-story training support for selection.')
    return {**split, 'inner': inner, 'selection_policy': 'sorted-story-round-robin-v1'}


def selection_options(data, options):
    # Nulls intentionally use the matched fit's training settings. The null is
    # a fixed-procedure ablation, not a separately optimized null competitor.
    return {k: v for k, v in {**options, 'seed': data.config['decoder']['selection_seed']}.items()
            if k not in {'retrain_null', 'export_traces', 'faithfulness', 'require_prepared'}}


def encoding_mixtures(config, groups):
    import numpy as np
    weights = [np.ones(len(groups)) / len(groups)]
    if len(groups) > 1:
        for seed in config['search_seeds']:
            weights.extend(np.random.default_rng(seed).dirichlet(np.ones(len(groups)), config['group_mixtures']))
    return weights
