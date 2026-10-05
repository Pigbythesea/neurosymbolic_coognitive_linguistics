"""Gated source-first review and explicit surgical patches; no semantic inference.

The gate controls the supported workflow, not filesystem/model access. Reviewers
must declare any other exposure. Original exports are read only.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import direct_annotations as d

ROOT = d.ROOT
OUT = ROOT / 'data/annotations/deniz-independent-review-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    d.immutable_bytes(Path(path), (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def initialize():
    if (OUT/'baseline.json').exists(): return read(OUT/'baseline.json')
    latest = read(d.OUTPUT/'latest.json')
    base = ROOT/latest['directory']
    files = ['author-corrected/annotations.jsonl', 'source-only.jsonl', 'checksums.json']
    result = {'review_protocol':'independent-prefix-review-v1', 'created_utc':stamp(),
              'baseline':latest, 'baseline_view':'author-corrected',
              'file_sha256':{p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in files},
              'model_family':'GPT-6-based Codex', 'exact_model_revision':None,
              'human_validation':'pending', 'semantic_accuracy_measured':False,
              'access_limit':'Gate and declarations document process; no filesystem sandbox or pretraining-exposure claim.',
              'scope':'Annotation review, versioned corrections and interface definitions only; no neural data or performance used.'}
    save(OUT/'baseline.json',result)
    return result


def baseline():
    manifest = initialize()
    return ROOT/manifest['baseline']['directory']


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def inputs(sid):
    base = baseline()
    source = [r for r in rows(base/'source-only.jsonl') if r['story_id']==sid]
    annotations = [r for r in rows(base/'author-corrected/annotations.jsonl') if r['story_id']==sid]
    if not source or len(source)!=len(annotations): raise ValueError('Unknown/incomplete story')
    return source, annotations


def folder(sid):
    if sid not in {f'story_{i:02}' for i in range(1,12)}: raise ValueError('Invalid story')
    return OUT/'stories'/sid


def reviewed(sid):
    return [read(p) for p in sorted((folder(sid)/'decisions').glob('*.json'))]


def trace(sid, kind, **data):
    path=folder(sid)/'trace.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'utc':stamp(),'kind':kind,**data},ensure_ascii=False)+'\n')


def register(sid, actor, exposure):
    path=folder(sid)/'actor.json'
    declaration={'story_id':sid,'actor':actor,'exposure':exposure,
                 'procedure':'One current source unit; commit interpretation; reveal corresponding graph; commit comparison/corrections; then next source.',
                 'no_neural_results':True,'no_future_target_source_or_graphs':True}
    save(path,declaration)
    return next_source(sid)


def next_source(sid):
    if not (folder(sid)/'actor.json').exists(): raise ValueError('Register first')
    sources,_=inputs(sid); n=len(reviewed(sid))
    if n==len(sources): return {'story_id':sid,'complete':True,'units':n}
    source=sources[n]; uid=source['unit_id']
    path=folder(sid)/'exposures'/(uid+'.json')
    if not path.exists():
        save(path,{'unit_id':uid,'source_hash':d.object_hash(source),'exposed_utc':stamp(),
                   'prior_decision_hash':d.object_hash(reviewed(sid)[-1]) if n else None})
        trace(sid,'source_exposed',unit_id=uid)
    return {'unit_id':uid,'text':source['text'],
            'start_token':source['unit']['start_token'],'end_token':source['unit']['end_token'],
            'available_at_token':source['available_at_token'],
            'words':[[w['index'],w['text']] for w in source['words']]}


def unanchor(graph):
    raw=deepcopy(graph)
    for family in d.COLLECTIONS:
        for item in raw[family]:
            for key in ('evidence','trigger'):
                if key in item:
                    ev=item[key]
                    item[key]={'quote':ev['quote'],'start':ev['start_token']}
    return raw


def interpret(sid, submission):
    sources,annotations=inputs(sid); n=len(reviewed(sid))
    uid=sources[n]['unit_id']
    if submission['unit_id']!=uid: raise ValueError('Only the current source can be interpreted')
    if not (folder(sid)/'exposures'/(uid+'.json')).exists(): raise ValueError('Read source first')
    if len(submission['interpretation'].strip())<12: raise ValueError('Record a substantive independent interpretation')
    path=folder(sid)/'interpretations'/(uid+'.json')
    if not path.exists():
        save(path,{'unit_id':uid,'interpretation':submission['interpretation'],
                   'source_hash':d.object_hash(sources[n]),'committed_utc':stamp(),
                   'actor':read(folder(sid)/'actor.json')['actor']})
        trace(sid,'interpretation_committed',unit_id=uid,interpretation_hash=d.object_hash(read(path)))
    elif read(path)['interpretation']!=submission['interpretation']:
        raise ValueError('Independent interpretation is immutable once graph is exposed')
    graph=annotations[n]['graph']
    exposure=folder(sid)/'graph-exposures'/(uid+'.json')
    if not exposure.exists():
        save(exposure,{'unit_id':uid,'baseline_graph_hash':annotations[n]['graph_hash'],
                       'interpretation_hash':d.object_hash(read(path)),'exposed_utc':stamp()})
        trace(sid,'graph_exposed',unit_id=uid)
    return unanchor(graph)


def apply_operations(graph, operations):
    result=deepcopy(graph)
    for op in operations:
        family=op['collection']
        if family not in d.COLLECTIONS: raise ValueError('Patch only explicit semantic collections')
        if not op.get('reason') or not op.get('evidence'): raise ValueError('Every patch requires source evidence and reason')
        matches=[i for i,x in enumerate(result[family]) if x['id']==op['id']]
        if op['op']=='add':
            if matches or op['value']['id']!=op['id']: raise ValueError('Invalid addition ID')
            result[family].append(deepcopy(op['value']))
        elif op['op']=='remove':
            if len(matches)!=1: raise ValueError('Missing removal ID')
            before=result[family][matches[0]]
            if before!=op['before']: raise ValueError('Removal precondition differs')
            del result[family][matches[0]]
        elif op['op']=='set':
            if len(matches)!=1: raise ValueError('Missing patch ID')
            obj=result[family][matches[0]]
            keys=op['field'].split('.')
            for key in keys[:-1]: obj=obj[int(key)] if isinstance(obj,list) else obj[key]
            key=int(keys[-1]) if isinstance(obj,list) else keys[-1]
            if obj[key]!=op['before']: raise ValueError('Patch precondition differs: '+op['id']+' '+op['field'])
            obj[key]=deepcopy(op['value'])
        else: raise ValueError('Unknown patch operation')
    return result


def commit(sid, submission):
    sources,annotations=inputs(sid); previous=reviewed(sid); n=len(previous)
    uid=sources[n]['unit_id']
    if submission['unit_id']!=uid: raise ValueError('Only current unit can be committed')
    ip=folder(sid)/'interpretations'/(uid+'.json')
    if not ip.exists() or not (folder(sid)/'graph-exposures'/(uid+'.json')).exists(): raise ValueError('Interpret/reveal graph first')
    if len(submission.get('assessment','').strip())<12: raise ValueError('Record the source/graph comparison')
    operations=submission.get('operations',[])
    unresolved=submission.get('unresolved',[])
    raw=apply_operations(unanchor(annotations[n]['graph']),operations)
    story=read(baseline()/'sources'/(sid+'.json'))
    unit=story['units'][n]
    prior=[{'graph':r['graph']} for r in previous]
    anchored=d.validate_graph(raw,story,unit,prior)
    # Preserve all original evidence metadata when the selected span is unchanged.
    # A replay adds explicit starts, which must not become unrelated data churn.
    for family in d.COLLECTIONS:
        for i,item in enumerate(anchored[family]):
            original=next((x for x in annotations[n]['graph'][family] if x['id']==item['id']),None)
            if original:
                for key in ('evidence','trigger'):
                    if key in item and key in original:
                        old=original[key]; new=item[key]
                        if (old['quote'],old['start_token'],old['end_token']) == (new['quote'],new['start_token'],new['end_token']):
                            item[key]=deepcopy(old)
    for op in operations:
        d.anchor(op['evidence'],story,unit)
    for issue in unresolved:
        if not issue.get('reason') or not issue.get('evidence'): raise ValueError('Unresolved cases need evidence and reason')
        if not isinstance(issue.get('affected_ids'),list) or not issue['affected_ids']:
            raise ValueError('Unresolved cases need explicit affected_ids')
        known={item['id'] for row in prior+[{'graph':anchored}] for family in d.COLLECTIONS for item in row['graph'][family]}
        if not set(issue['affected_ids'])<=known: raise ValueError('Unresolved case references an unavailable target')
        d.anchor(issue['evidence'],story,unit)
    value={'unit_id':uid,'story_id':sid,'actor':read(folder(sid)/'actor.json')['actor'],
           'baseline_graph_hash':annotations[n]['graph_hash'],'interpretation_hash':d.object_hash(read(ip)),
           'available_at_token':sources[n]['available_at_token'],'available_at_seconds':sources[n]['available_at_seconds'],
           'assessment':submission['assessment'],'operations':operations,'unresolved':unresolved,
           'status':'corrected' if operations else ('retained_with_open_question' if unresolved else 'retained'),
           'graph':anchored,'graph_hash':d.object_hash(anchored),'committed_utc':stamp()}
    save(folder(sid)/'decisions'/(uid+'.json'),value)
    trace(sid,'decision_committed',unit_id=uid,decision_hash=d.object_hash(value),patches=len(operations))
    return next_source(sid)


def status():
    source=rows(baseline()/'source-only.jsonl')
    return {'baseline':initialize()['baseline'], 'stories':[
        {'story_id':sid,'reviewed':len(reviewed(sid)),
         'total':sum(r['story_id']==sid for r in source),
         'patches':sum(len(r['operations']) for r in reviewed(sid))}
        for sid in sorted({r['story_id'] for r in source})]}


def ledger(sid):
    return [{'unit_id':r['unit_id'],'assessment':r['assessment'], 'operations':r['operations'],
             'unresolved':r['unresolved']} for r in reviewed(sid)]
