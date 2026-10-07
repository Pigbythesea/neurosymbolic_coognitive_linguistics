"""Claim-directed experiment graph. No Cartesian expansion of secondary layers."""


def populate_jobs(data, resolved, resources, phase, encoding_device, decoder_device, add):
    plan = resolved['definitions']
    folds = plan['development_folds'] if phase == 'development' else [plan['final_fold']]
    contexts = plan['geometry']['context_folds'] if phase == 'development' else []
    humans = [{'modality': 'brain', 'subject': s} for s in plan['subjects']]
    models = [{'modality': 'model', 'model': m['id'], 'layer': m['primary_layer']} for m in resolved['models']]
    depth = [{'modality': 'model', 'model': m['id'], 'layer': layer}
             for m in resolved['models'] for layer in m['descriptive_layers'] if layer != m['primary_layer']]
    primary, observations = humans + models, humans + models + depth
    prep_device = resources['preparation']['device']
    prep_resource = 'gpu-prepare' if prep_device == 'cuda' else 'prepare'
    dec_resource = ('gpu-' if decoder_device == 'cuda' else 'cpu-') + 'decoder'
    enc_resource = ('gpu-' if encoding_device == 'cuda' else 'cpu-') + 'encoding'
    geo_device = encoding_device
    geo_resource = 'gpu-geometry' if geo_device == 'cuda' else 'cpu-report'
    seed = data.config['decoder']['selection_seed']
    decoder_parents, encoding_parents, all_fits = {}, {}, []
    def key(obs):
        return tuple(sorted(obs.items()))

    # Same public prior for every observation system; retained repeats belong
    # to the real first participant. Aggregate answers are repeat-invariant.
    for fold in folds:
        base = {**humans[0], 'fold': fold, 'seed': seed, 'family': 'prior', 'device': 'cpu'}
        selection = add('select-decoder', base, 'cpu-prior')
        for s in plan['seeds']:
            all_fits.append(add('decoder', {**base, 'seed': s, 'export_traces': False}, 'cpu-prior', [selection], purpose='shared query-only baseline'))

    for obs in observations:
        is_primary = obs in primary
        for fold in folds + (contexts if is_primary else []):
            context = fold in contexts
            prep = add('prepare-decoder', {**obs, 'fold': fold, 'seed': seed, 'device': prep_device,
                                          'outer_only': not is_primary}, prep_resource)
            families = ([plan['geometry']['primary_decoder_parent']] if context else
                        plan['decoder']['primary_families'] if is_primary else plan['decoder']['descriptive_families'])
            seeds = plan['seeds'] if is_primary else plan['decoder']['descriptive_seeds']
            for family in families:
                base = {**obs, 'fold': fold, 'seed': seed, 'family': family, 'device': decoder_device, 'require_prepared': True}
                selection = add('select-decoder', base, dec_resource, [prep])
                for s in seeds:
                    opts = {**base, 'seed': s, 'export_traces': family == 'structured' and context,
                            'faithfulness': is_primary and family == 'structured' and not context}
                    fitted = add('decoder', opts, dec_resource, [prep, selection])
                    all_fits.append(fitted)
                    if context:
                        decoder_parents[(key(obs), fold, s)] = fitted
                    if not context and is_primary and family in plan['decoder']['retrained_null_families']:
                        all_fits.append(add('decoder', {**opts, 'retrain_null': True, 'export_traces': False, 'faithfulness': False}, dec_resource, [prep, selection]))

    # One deterministic candidate-search union and one selected fit, not three
    # randomized searches masquerading as three independent encoding repeats.
    for subject in plan['subjects']:
        for fold in folds + contexts:
            for contrast in plan['encoding_contrasts']:
                if fold in contexts and contrast['comparison'] != 'matched-binding':
                    continue
                for side in (['augmented'] if fold in contexts else ['baseline', 'augmented']):
                    opts = {'subject': subject, 'fold': fold, 'seed': seed, 'groups': contrast[side],
                            'comparison': contrast['comparison'], 'device': encoding_device}
                    fitted = add('encoding', opts, enc_resource)
                    all_fits.append(fitted)
                    if fold in contexts:
                        encoding_parents[(subject, fold)] = fitted
            if fold in contexts:
                continue
            for obs in models + depth:
                conditions = plan['model_encoding']['conditions' if obs in models else 'descriptive_conditions']
                for groups in conditions:
                    opts = {'subject': subject, 'fold': fold, 'seed': seed, 'groups': groups,
                            'comparison': plan['model_encoding']['comparison'], 'device': encoding_device,
                            'mask_models': [obs['model'] + ':' + str(obs['layer'])]}
                    if 'model' in groups:
                        opts.update(model=obs['model'], layer=obs['layer'])
                    all_fits.append(add('encoding', opts, enc_resource))

    geometry, comparisons = {}, []
    for fold in contexts:
        for obs in observations:
            for residual in [False, True]:
                opts = {**obs, 'fold': fold, 'device': geo_device, 'views': ['native'], 'residualize_presentation': residual}
                geometry[('native', key(obs), fold, residual)] = add('geometry-panel', opts, geo_resource)
            if obs in primary:
                for s in plan['seeds']:
                    parent = decoder_parents[(key(obs), fold, s)]
                    opts = {**obs, 'fold': fold, 'seed': s, 'device': geo_device,
                            'views': ['grounding', 'latent', 'latent-passage'], 'parent_job': parent}
                    geometry[('derived', key(obs), fold, s)] = add('geometry-panel', opts, geo_resource, [parent])
            if obs['modality'] == 'brain':
                parent = encoding_parents[(obs['subject'], fold)]
                opts = {**obs, 'fold': fold, 'device': geo_device, 'views': ['encoding-implied'], 'parent_job': parent}
                geometry[('implied', key(obs), fold, 0)] = add('geometry-panel', opts, geo_resource, [parent])
        cooc = add('geometry-panel', {'fold': fold, 'device': 'cpu', 'modality': 'brain',
                    'views': ['cooccurrence']}, 'cpu-report')
        for k, first in list(geometry.items()):
            if k[2] != fold:
                continue
            comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': cooc,
                'device': geo_device, 'mode': 'rdm', 'purpose': 'cooccurrence'}, geo_resource, [first, cooc]))
        for human in humans:
            implied = geometry[('implied', key(human), fold, 0)]
            native = geometry[('native', key(human), fold, True)]
            comparisons.append(add('compare-geometry-panel', {'first_job': implied, 'second_job': native,
                'device': geo_device, 'mode': 'rdm', 'purpose': 'encoding-implied-native', 'cross_views': True},
                geo_resource, [implied, native]))
            for model in models + depth:
                for residual in [False, True]:
                    first, second = [geometry[('native', key(o), fold, residual)] for o in [human, model]]
                    comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': second,
                        'device': geo_device, 'mode': 'rdm', 'purpose': 'human-model-native'}, geo_resource, [first, second]))
                if model in models:
                    for s in plan['seeds']:
                        first, second = [geometry[('derived', key(o), fold, s)] for o in [human, model]]
                        comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': second,
                            'device': geo_device, 'mode': 'rdm', 'purpose': 'human-model-derived'}, geo_resource, [first, second]))
            first = geometry[('derived', key(human), fold, plan['seeds'][0])]
            for s in plan['seeds'][1:]:
                second = geometry[('derived', key(human), fold, s)]
                comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': second,
                    'device': geo_device, 'mode': 'maps', 'purpose': 'seed-stability'}, geo_resource, [first, second]))
        for i, human in enumerate(humans):
            for other in humans[i+1:]:
                for s in plan['seeds']:
                    first, second = [geometry[('derived', key(o), fold, s)] for o in [human, other]]
                    comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': second,
                        'device': geo_device, 'mode': 'maps', 'purpose': 'participant-stability'}, geo_resource, [first, second]))
    if len(contexts) == 2:
        for obs in humans:
            for s in plan['seeds']:
                first, second = [geometry[('derived', key(obs), fold, s)] for fold in contexts]
                comparisons.append(add('compare-geometry-panel', {'first_job': first, 'second_job': second,
                    'device': geo_device, 'mode': 'maps', 'purpose': 'context-stability'}, geo_resource, [first, second]))
    coverage = add('semantic-coverage', {'phase': phase, 'device': 'cpu'}, 'cpu-report')
    add('study-report', {'phase': phase, 'device': 'cpu'}, 'cpu-report', list(dict.fromkeys(all_fits + list(geometry.values()) + comparisons + [coverage])), purpose='complete inventory, paired effects, coverage and declared comparison families')
