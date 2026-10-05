"""Versioned, reversible semantic-interface declarations, not model features.

No training, neural data, embeddings, truth flattening or legacy schema conversion.
Only explicitly listed lexical aliases are proposed; original payloads survive.
"""
from collections import defaultdict
from copy import deepcopy

VERSION = 'reviewed-semantic-interface-v1'

PREDICATE_GROUPS = [
    ('buy', ['buy','purchase'], {'agent':'buyer','buyer':'buyer','theme':'purchased_object','object':'purchased_object'},
     'Commercial acquisition. Intended beneficiary, actual recipient and seller location stay distinct.'),
    ('be_age', ['be_age','be_aged'], {'theme':'age_bearer','person':'age_bearer','age':'age_value','value':'age_value'},
     'Age predication; exact/range/approximate values and time remain separate.'),
    ('obtain_job', ['get_job','obtain_job','obtain_employment'], {'agent':'job_obtainer','person':'job_obtainer','theme':'job','job':'job'},
     'Acquiring employment, distinct from holding a job or working.'),
    ('be_able', ['be_able','be_able_to'], {'agent':'capacity_bearer','person':'capacity_bearer','bearer':'capacity_bearer','experiencer':'capacity_bearer','action':'capacity_content','activity':'capacity_content','content':'capacity_content'},
     'Capacity predication. Polarity, conditional/question scope and realization remain independent. Ambiguous can/permission is excluded.'),
    ('receive_phone_call', ['receive_call','receive_phone_call'], {'caller':'caller','source':'caller','recipient':'recipient','theme':'call'},
     'Receiving a telephone call; unresolved caller stays null and no call node is invented.'),
    ('be_naked', ['naked','be_naked','be_nude'], {},
     'Physical unclothed state in reviewed examples. Retain lexical choice, degree, figurative qualifiers and all contexts.'),
    ('be_okay', ['be_ok','be_okay'], {},
     'Orthographic variants of OK/alright state. This does not identify health, permission, assent or life-success senses.'),
    ('be_afraid', ['afraid','be_afraid'], {},
     'Adjectival fear state. Content and stimulus roles are retained separately; no universal role collapse.'),
    ('reside', ['reside','live_at','live_in'], {'theme':'resident','resident':'resident','place':'residence'},
     'Residence predication. Bare live (being alive or living life) is excluded.'),
    ('be_certain', ['certain','be_certain'], {},
     'Adjectival certainty state; topic and propositional content remain distinct.'),
    ('be_different', ['different','be_different'], {'reference':'comparison_standard','comparison':'comparison_standard'},
     'Difference comparison; compared participants, comparison domains and degree remain explicit.'),
    ('be_disgusting', ['disgusting','be_disgusting'], {'theme':'stimulus'},
     'Disgust evaluation, retaining experiencer, reported insult scope and sense rather than inferring physical disgust.'),
    ('be_excited', ['excited','be_excited'], {},
     'Excitement state, preserving stimulus versus propositional content and conjunction with other emotions.'),
    ('be_fortunate', ['fortunate','be_fortunate'], {},
     'Fortune evaluation; beneficiary and experiencer are not globally equated and null remains unknown.'),
    ('be_frightening', ['frightening','be_frightening'], {'theme':'stimulus'},
     'Fear-inducing evaluation, distinct from being afraid; retain experiencer and evaluation scope.'),
    ('be_fun', ['fun','be_fun'], {},
     'Enjoyable activity/experience evaluation, not the experiencer actually enjoying it.'),
    ('be_great', ['great','be_great'], {'theme':'evaluated'},
     'Positive evaluation with source-specific standard, degree and sense retained; no numerical quality scale.'),
    ('be_happy', ['happy','be_happy'], {},
     'Happiness state. Trait, comparison, momentary state and attributed appearance remain qualified distinctions.'),
    ('be_interesting', ['interesting','be_interesting'], {'theme':'stimulus'},
     'Interest-eliciting evaluation; retain appearance/belief scope and experiencer if supplied.'),
    ('be_nervous', ['nervous','be_nervous'], {},
     'Nervousness state. Cause, degree, comparison and time remain separate.'),
    ('say_farewell', ['bid_farewell','say_farewell','farewell'], {},
     'Closing speech act with speaker/recipient; no physical departure is inferred.'),
]

# Aliases name a common operator family, never discard original kinds or payloads.
CONTEXT_ALIASES = {
    'possibility':'possible', 'capacity':'ability', 'appearance':'apparent',
    'belief':'believed', 'required':'obligatory',
}
RELATION_ALIASES = {
    'overlaps':'overlap',
    'contrasts':'contrast','contrasts_with':'contrast','discourse_contrast':'contrast',
    'explains':'explanation','explanation_for':'explanation',
    'discourse_elaboration':'elaboration','elaborated_by':'elaboration',
    'answers':'answer_to','continues':'continuation_of',
    'reiterates':'restates','exemplifies':'example_of',
}
# The same raw label has both directions. These are enumerated corpus bindings,
# not an inference from chronology (a later assertion can elaborate older text).
RESTATEMENT_ORIENTATION = {
    'story_05:u0042_rel2':True,'story_05:u0049_rel2':True,
    'story_05:u0057_rel1':True,'story_05:u0088_rel0':True,
    'story_07:u0059_rel0':False,'story_07:u0079_rel0':False,
    'story_09:u0053_rel0':True,'story_09:u0053_rel1':True,
}
RELATION_REVERSE_IDS = {'story_01:u0048_rel2'}


def relation_alias(item):
    canonical=RELATION_ALIASES.get(item['type'],item['type'])
    reverse=item['id'] in RELATION_REVERSE_IDS
    if reverse:
        if item['type']!='exemplification': raise ValueError('Reviewed example orientation changed')
        canonical='example_of'
    if item['id'] in RESTATEMENT_ORIENTATION and item['type']!='restatement':
        raise ValueError('Reviewed restatement orientation changed')
    if item['type']=='restatement' and item['id'] in RESTATEMENT_ORIENTATION:
        canonical='restates'; reverse=RESTATEMENT_ORIENTATION[item['id']]
    return {'canonical_type':canonical,'canonical_source':item['target'] if reverse else item['source'],
            'canonical_target':item['source'] if reverse else item['target'],'endpoints_reversed':reverse}


def relation_definitions():
    return {
        'restatement_orientation_by_id':RESTATEMENT_ORIENTATION,
        'other_reversed_ids':sorted(RELATION_REVERSE_IDS),
        'direction':{
            'restates':'source = restating expression; target = earlier content. This is not recurrence of a physical event.',
            'elaboration':'source = general/base content; target = elaborating detail. Chronological order may differ.',
            'example_of':'source = illustrative case; target = illustrated class, property or pattern. Not automatic taxonomic subclassing.',
            'explanation':'source = explaining proposition; target = explained proposition. Do not silently turn justification into objective causation.',
            'answer_to':'source = answer/content or answering act; target = question/content or questioning act. Preserve endpoint types and as-if scope.',
            'continuation_of':'source = continuing event/content; target = earlier event/content. Do not infer event identity from continuation.',
            'contrast':'Preserve supplied source/target presentation order; this is not temporal precedence or causal direction.',
        },
        'examples':{
            'opposite_restatement_directions':['story_05:u0042_rel2','story_07:u0059_rel0','story_08:u0153_rel0'],
            'example_direction':['story_01:u0048_rel2','story_04:u0031_rel3'],
            'contrast_spelling':['story_08:u0167_rel3','story_11:u0109_rel1'],
            'explanation':['story_01:u0039_rel3','story_08:u0151_rel1'],
        },
        'preserve_distinctions':['concession and concessive_contrast/despite retain force and direction; no blanket merge with contrast',
            'contrast_in_framing concerns alternative descriptions, not generic discourse opposition',
            'repeat_of denotes recurrence; restates denotes repeated content',
            'discourse_continuation/continuation retain their original forward presentation orientation',
            'coincident_with episode overlap is not automatically coincides_with calendar identity'],
        'unknown_restatement':'An unlisted restatement ID keeps its raw type and orientation; new instances require explicit inspection.',
        'reversibility':'Raw entire record remains in original; canonical endpoint reversal is explicit and does not mutate the graph.',
    }


def relation_orientation_bindings(records):
    selected=set(RESTATEMENT_ORIENTATION)|RELATION_REVERSE_IDS
    for row in records:
        for relation in row['graph']['relations']:
            if relation['id'] not in selected: continue
            yield {'case_id':relation['id']+':orientation','unit_id':row['unit_id'],'story_id':row['story_id'],
                   'available_at_token':row['available_at_token'],'available_at_seconds':row['available_at_seconds'],
                   'kind':'relation_orientation','target':relation['id'],
                   'affected_ids':[relation['id'],relation['source'],relation['target']],
                   'evidence_refs':[relation['id']],'source_evidence':[relation['evidence']],
                   'canonical_relation':relation_alias(relation),
                   'interpretation':'Explicit optional interface orientation; raw relation and its contexts remain unchanged. Interpret according to relation_definitions, never infer direction from label similarity or chronology.'}

# Explicitly inspected negative-modal encodings. The event polarity is the sign
# of the modal claim, not a second negation of its complement. IDs make this
# convention reviewable and prevent an unbounded string-based rewrite.
MODAL_DENIALS = {
    'not_possible':[
        'story_01:u0022_qua0','story_01:u0022_qua1','story_03:u0050_qua0'],
    'not_able':[
        'story_01:u0069_qua0','story_01:u0089_qua0','story_01:u0105_qua0',
        'story_01:u0107_qua1','story_01:u0122_qua1','story_01:u0122_qua2',
        'story_02:u0041_qua0','story_02:u0092_qua0',
        'story_04:u0022_qua4','story_04:u0031_qua3','story_04:u0040_qua0',
        'story_04:u0044_qua0','story_04:u0050_qua2',
        'story_07:u0021_qua2','story_07:u0028_qua0','story_07:u0036_qua0',
        'story_07:u0087_qua11','story_09:u0012_qua0','story_11:u0062_qua0'],
    'forbidden':['story_07:u0062_qua3','story_07:u0069_qua1'],
    'refusal':['story_02:u0088_qua0','story_02:u0089_qua0','story_02:u0090_qua0','story_04:u0066_qua0'],
    'unresolved_prohibition_or_inability':['story_02:u0010_qua0'],
    'not_can_force_unspecified':['story_11:u0101_qua0'],
}


def modal_bindings(records):
    """Concrete scope definitions for reviewed redundant negative modal cues."""
    selected={ident:operator for operator,ids in MODAL_DENIALS.items() for ident in ids}
    events={e['id']:e for row in records for e in row['graph']['events']}
    found=set()
    for row in records:
        for q in row['graph']['qualifiers']:
            if q['id'] not in selected: continue
            found.add(q['id']); event=events[q['target']]
            if event['polarity']!='negative' or q['dimension']!='modality':
                raise ValueError('Reviewed modal case changed: '+q['id'])
            yield {'case_id':q['id']+':modal_scope','unit_id':row['unit_id'],
                   'story_id':row['story_id'],'available_at_token':row['available_at_token'],
                   'available_at_seconds':row['available_at_seconds'],
                   'affected_ids':[event['id'],q['id']],'evidence_refs':[q['id']],
                   'source_evidence':[q['evidence'],event['evidence']],
                   'kind':'negative_modal_encoding','target':event['id'],
                   'operator':selected[q['id']],'complement':{'op':'qualified_atom','id':event['id']},
                   'consume_fields':[{'id':event['id'],'field':'polarity'},{'id':q['id'],'field':'value'}],
                   'interpretation':'The supplied operator already contains the denial/refusal. Apply it once to positive predicate content with nonconsumed qualification. Preserve all outer contexts, temporal restrictions and uncertain force; never invert to a denied negative complement.'}
    # Partial inventories are allowed for tooling tests; full export verifies IDs.


def composition_rules():
    return {
        'modal_denial_instances':MODAL_DENIALS,
        'modal_operators':{
            'not_possible':'NOT POSSIBLE(positive qualified content)',
            'not_able':'NOT ABLE(positive qualified content)',
            'forbidden':'FORBIDDEN(positive qualified content), preserving social/practical/deontic force',
            'refusal':'REFUSE(positive qualified content); retain whether reported, future, or unwillingness is uncertain',
            'unresolved_prohibition_or_inability':'Alternatives FORBIDDEN(content) and NOT ABLE(content), without selecting one',
            'not_can_force_unspecified':'NOT CAN(content), retaining explicitly unknown practical/permission force',
        },
        'negative_frequency':'For never/zero occasions with negative event, bind the supplied relevant time domain over one denied proposition. If a modal binding exists, quantify that modal denial; never introduce a second NOT from the word never.',
        'negative_existential':'For zero/negative_existential/universal_negative plus negative event, bind the stated domain as no satisfying positive instances, retaining restrictions. Redundant signs do not cancel.',
        'distributive_universal_negative':'Universal/distributive plus negative event is EACH domain member satisfies the denial; preserve exclusions and explicit role. This differs from NOT ALL members satisfy the positive predicate.',
        'quantifier_under_negation':'Explicit existential or many under negation stays inside that negation: NOT EXISTS or NOT MANY, never EXISTS NOT or MANY NOT.',
        'object_all':'A universal object under a denied ability need not distribute the denied ability: cannot appreciate it all does not mean cannot appreciate any part. Preserve target/domain/role and the modal outer scope.',
        'question_confirmation':'A confirmation-question qualifier over a negative proposition questions that negative proposition; matching question context is the same question, not polarity cancellation.',
        'qualifier_target_scope':'A qualifier is attached to its target expression and cannot project that expression outside its target contexts. Explicit qualifier contexts further constrain the attachment; an empty list is no license for projection. Example: story_08:u0128_qua0 qualifies scoped story_08:u0128_cool.',
        'unknown_scope':'If domain, role, operator order or negative-cue identity is not specified by the graph or reviewed cases, preserve it symbolically and report unresolved scope instead of guessing.',
    }


def declarations():
    return {
        'version':VERSION,
        'status':'reviewed interface specification; human validation pending',
        'scope':'Symbolic annotation interface only; not a fitted analysis vocabulary.',
        'application':'Optional explicit aliases in a parallel view; retain source graphs and raw labels as provenance, not infallible semantic authority.',
        'predicate_groups':[{'canonical':c,'labels':labels,'roles':roles,'reason':reason}
                            for c,labels,roles,reason in PREDICATE_GROUPS],
        'context_aliases':CONTEXT_ALIASES,
        'context_alias_limit':'Operator-family spelling variants only. Attribution predicate, polarity, holder, parents and qualifier scope determine interpretation; believed can host remembered/discovered content.',
        'relation_aliases':RELATION_ALIASES,
        'relation_definitions':relation_definitions(),
        'composition_rules':composition_rules(),
        'unlisted_labels':'Preserve under their typed original key. No string-stemming, role unification, synonym guessing or closed-world inference.',
        'heldout_policy':'These are post-review linguistic interface declarations. All-story label inventories are diagnostic, not fit inputs. Downstream vocabularies, pruning, thresholds and empirical transforms must be fit within training folds; story_11 remains held out.',
        'preserve':['raw label and sense','ordered argument list including repeated roles','nulls and alternatives','evidence spans','unit availability','context DAG','qualified proposition endpoints','support and uncertainty','entity versus event identity','all literal/qualifier value structure'],
        'do_not_merge':[
            ['work_with','work_for','colleagues and employment hierarchy differ'],
            ['be_able','be_allowed_or_able_to','capacity and permission ambiguity differ'],
            ['know','believe','knowledge, subjective conviction and belief require sense/attribution'],
            ['same_event','same_episode_as','event identity differs from participation in one episode'],
            ['realization_of','same_event','a later actualization does not make earlier intended content actual'],
            ['part_of','member_of','mereology differs from group membership'],
            ['overlap','before','overlap is not causal or strict precedence'],
            ['recipient','intended_beneficiary','actual receipt differs from intended benefit'],
            ['somebody','generic_person','existential and generic quantification depend on scope, not node labels'],
        ],
    }


def alias_view(records):
    """Sidecar, one row per actual record; no graph mutation or inferred links."""
    pmap={label:(canonical,roles) for canonical,labels,roles,_ in PREDICATE_GROUPS for label in labels}
    for row in records:
        for family in sorted(row['graph']):
            items=row['graph'][family]
            if not isinstance(items,list): continue
            for item in items:
                if not isinstance(item,dict) or 'id' not in item: continue
                value={'story_id':row['story_id'],'unit_id':row['unit_id'],
                       'available_at_token':row['available_at_token'],
                       'available_at_seconds':row['available_at_seconds'],
                       'record_type':family,'id':item['id'],'source_graph_hash':row['graph_hash'],
                       'original':deepcopy(item),'interface_version':VERSION}
                if family=='events':
                    canonical,roles=pmap.get(item['predicate'],(item['predicate'],{}))
                    value['canonical_predicate']=canonical
                    value['argument_aliases']=[{'argument_index':i,'original_role':a['role'],
                                               'canonical_role':roles.get(a['role'],a['role'])}
                                              for i,a in enumerate(item['arguments'])]
                elif family=='contexts': value['canonical_kind']=CONTEXT_ALIASES.get(item['kind'],item['kind'])
                elif family=='relations': value.update(relation_alias(item))
                yield value


def inventory(records, splits):
    """Diagnostic counts/examples with held-out counts separated, never fitted."""
    data=defaultdict(lambda:defaultdict(lambda:{'count':0,'training_count':0,'heldout_count':0,'examples':[]}))
    fields={'entities':'concept','events':'predicate','contexts':'kind','relations':'type',
            'properties':'attribute','qualifiers':'dimension','mentions':'form'}
    for row in records:
        for family,field in fields.items():
            for item in row['graph'][family]:
                keys=[(family,item[field])]
                if family=='events': keys += [('predicate_roles',item['predicate']+' / '+a['role']) for a in item['arguments']]
                for category,label in keys:
                    entry=data[category][label]; entry['count']+=1
                    entry['heldout_count' if row['story_id']=='story_11' else 'training_count']+=1
                    if len(entry['examples'])<3: entry['examples'].append(item['id'])
    return {'purpose':'diagnostic_only_not_fit_vocabulary','splits':splits,
            'labels':{c:dict(sorted(v.items())) for c,v in data.items()}}


def paths_to_root(context_id, index, stack=()):
    if context_id in stack: raise ValueError('Cyclic context')
    context=index[context_id]
    if not context['parents']: return [[context_id]]
    return [path+[context_id] for p in context['parents'] for path in paths_to_root(p,index,stack+(context_id,))]


def scope_bindings(records):
    """Lossless availability-tagged attachments; deliberately no truth collapse."""
    contexts={}; kinds={}; items_by_id={}
    for row in records:
        graph=row['graph']
        for family,items in graph.items():
            if isinstance(items,list):
                for item in items:
                    if isinstance(item,dict) and 'id' in item:
                        kinds[item['id']]=family; items_by_id[item['id']]=item
        contexts.update({x['id']:x for x in graph['contexts']})
        for family in ('events','relations','properties','qualifiers','identity_links'):
            for item in graph[family]:
                target_id=item.get('target'); target=items_by_id.get(target_id,{})
                target_paths=([p for c in target.get('contexts',[]) for p in paths_to_root(c,contexts)]
                              if family=='qualifiers' else [])
                if family=='qualifiers' and target_id in contexts: target_paths=paths_to_root(target_id,contexts)
                yield {'id':item['id'],'record_type':family,'story_id':row['story_id'],
                       'unit_id':row['unit_id'],'available_at_token':row['available_at_token'],
                       'available_at_seconds':row['available_at_seconds'],
                       'context_paths_outer_to_inner':[p for c in item.get('contexts',[]) for p in paths_to_root(c,contexts)],
                       'local_polarity':item.get('polarity'),
                       'target_id':target_id,'target_kind':kinds.get(target_id),
                       'qualifier_target_context_paths_outer_to_inner':target_paths,
                       'rule':'All listed paths constrain the statement; multiple paths are not independent root assertions. Apply explicit scope overrides before any operator simplification.'}
