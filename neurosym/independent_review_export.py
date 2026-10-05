"""Immutable export and deterministic audit of independent semantic review."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import platform
import re

from . import direct_annotations as d
from . import independent_review as r
from . import review_semantic_interface as interface


def sha(data): return hashlib.sha256(data).hexdigest()
def json_bytes(value): return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def adoption_manifest():
    path=r.OUT/'adoption.json'
    return r.read(path) if path.exists() else {'protocol':'surgical-correction-adoption-v1','exclusions':[]}


def question_bindings():
    path=r.OUT/'question-bindings.json'
    return r.read(path) if path.exists() else {'protocol':'review-question-target-bindings-v1','bindings':[]}


def selected_operations(decision,adoption):
    excluded={x['id']:x for x in adoption['exclusions']}
    selected=[]; declined=[]
    for position,op in enumerate(decision['operations']):
        if op['id'] in excluded:
            rule=excluded[op['id']]
            selector=rule.get('operation_selector',{'op':'add','collection':'mentions'})
            if rule['unit_id']!=decision['unit_id'] or any(op.get(k)!=v for k,v in selector.items()):
                raise ValueError('Adoption exclusion is not the declared redundant operation')
            declined.append((position,op,rule))
        else: selected.append((position,op))
    return selected,declined


def preserve_anchors(replay,original):
    result=deepcopy(replay)
    for family in d.COLLECTIONS:
        old={x['id']:x for x in original[family]}
        for item in result[family]:
            if item['id'] not in old: continue
            for key in ('evidence','trigger'):
                if key in item and key in old[item['id']]:
                    before=old[item['id']][key]
                    if all(before[field]==item[key][field] for field in ('quote','start_token','end_token')):
                        item[key]=deepcopy(before)
    return result


def all_refs(value, wanted, path=''):
    if isinstance(value,dict):
        for key,v in value.items():
            if key not in {'id','evidence','trigger','summary','decisions','label','sense','explanation'}:
                yield from all_refs(v,wanted,path+'/'+str(key))
    elif isinstance(value,list):
        for i,v in enumerate(value): yield from all_refs(v,wanted,path+'/'+str(i))
    elif isinstance(value,str) and value in wanted: yield value,path


def embedded_references(value,path=''):
    if isinstance(value,dict):
        for key,v in value.items():
            if key not in {'id','evidence','trigger','summary','decisions','label','sense','explanation'}:
                yield from embedded_references(v,path+'/'+str(key))
    elif isinstance(value,list):
        for i,v in enumerate(value): yield from embedded_references(v,path+'/'+str(i))
    elif isinstance(value,str) and re.fullmatch(r'story_\d{2}:[A-Za-z0-9_:-]+',value):
        yield value,path


def baseline_integrity():
    base=r.baseline(); pinned=r.initialize()
    for name,digest in pinned['file_sha256'].items():
        if sha((base/name).read_bytes())!=digest: raise ValueError('Pinned baseline changed: '+name)
    checks=r.read(base/'checksums.json')
    # Production checksums are a direct path->SHA mapping.
    if 'files' in checks: checks=checks['files']
    for name,digest in checks.items():
        if sha((base/name).read_bytes())!=digest: raise ValueError('Original export changed: '+name)
    return {'baseline_build_hash':pinned['baseline']['build_hash'],'unchanged_files':len(checks),
            'pinned_hashes_unchanged':True}


def load_inputs(require_complete=True):
    base=r.baseline()
    source=r.rows(base/'source-only.jsonl')
    original=r.rows(base/'author-corrected/annotations.jsonl')
    sources={x['unit_id']:x for x in source}
    stories={sid:r.read(base/'sources'/(sid+'.json')) for sid in sorted({x['story_id'] for x in source})}
    decisions={x['unit_id']:x for sid in stories for x in r.reviewed(sid)}
    if require_complete and set(decisions)!=set(sources):
        raise ValueError(f'Incomplete review: {len(decisions)}/{len(sources)}')
    if not set(decisions)<=set(sources): raise ValueError('Review contains an unknown unit')
    return source,original,stories,decisions


def audit(require_complete=True):
    integrity=baseline_integrity()
    source,original,stories,decisions=load_inputs(require_complete)
    coverage=[]; changes=[]; excluded_proposals=[]; unresolved=[]; reviewed=[]; priors=defaultdict(list)
    proposal_priors=defaultdict(list); effective_ids=defaultdict(set); adoption=adoption_manifest()
    bindings=question_bindings(); binding_index={x['issue_id']:x for x in bindings['bindings']}
    previous_decisions={}; previous_times={}; graph_ids={}; unit_positions={}; reference_count=0
    for original_row,src in zip(original,source,strict=True):
        sid,uid=src['story_id'],src['unit_id']
        if uid!=original_row['unit_id']: raise ValueError('Source/graph order differs')
        if uid not in decisions: continue
        decision=decisions[uid]; folder=r.folder(sid)
        exposure=r.read(folder/'exposures'/(uid+'.json'))
        interpretation=r.read(folder/'interpretations'/(uid+'.json'))
        graph_exposure=r.read(folder/'graph-exposures'/(uid+'.json'))
        actor=r.read(folder/'actor.json')
        if interpretation['source_hash']!=d.object_hash(src) or exposure['source_hash']!=d.object_hash(src):
            raise ValueError('Source receipt hash differs: '+uid)
        if graph_exposure['baseline_graph_hash']!=original_row['graph_hash'] or decision['baseline_graph_hash']!=original_row['graph_hash']:
            raise ValueError('Wrong graph reviewed: '+uid)
        ih=d.object_hash(interpretation)
        if decision['interpretation_hash']!=ih or graph_exposure['interpretation_hash']!=ih:
            raise ValueError('Interpretation altered: '+uid)
        times=[exposure['exposed_utc'],interpretation['committed_utc'],graph_exposure['exposed_utc'],decision['committed_utc']]
        if times!=sorted(times) or previous_times.get(sid,'')>times[0]:
            raise ValueError('Source-first chronology failed: '+uid)
        if exposure['prior_decision_hash']!=previous_decisions.get(sid): raise ValueError('Review prefix chain failed: '+uid)
        if actor['actor']!=decision['actor'] or actor['actor']!=interpretation['actor']: raise ValueError('Actor receipt mismatch')
        previous_decisions[sid]=d.object_hash(decision); previous_times[sid]=times[-1]
        unit=stories[sid]['units'][len(priors[sid])]
        if unit['id']!=uid or src['unit']!=unit: raise ValueError('Noncontiguous review or source unit changed')
        if src['available_at_token']!=unit['end_token']: raise ValueError('Not full-unit availability')
        for key in ('available_at_token','available_at_seconds'):
            if decision[key]!=src[key] or original_row[key]!=src[key]: raise ValueError('Timing changed: '+uid)
        raw=r.apply_operations(r.unanchor(original_row['graph']),decision['operations'])
        replay=d.validate_graph(raw,stories[sid],unit,proposal_priors[sid])
        if r.unanchor(replay)!=r.unanchor(decision['graph']): raise ValueError('Patch replay differs: '+uid)
        if d.object_hash(decision['graph'])!=decision['graph_hash']: raise ValueError('Decision graph hash differs')
        # Check derived character coordinates/verbatim independently of optional raw anchor selectors.
        for family in d.COLLECTIONS:
            for actual,expected in zip(decision['graph'][family],replay[family],strict=True):
                for key in ('evidence','trigger'):
                    if key in actual:
                        for field in ('quote','start_token','end_token','char_start','char_end','verbatim'):
                            if actual[key][field]!=expected[key][field]: raise ValueError('Evidence span differs: '+actual['id'])
                if actual['id'] in graph_ids: raise ValueError('Duplicate ID')
                graph_ids[actual['id']]=(uid,src['available_at_token'])
        selected,declined=selected_operations(decision,adoption)
        effective_raw=r.apply_operations(r.unanchor(original_row['graph']),[op for _,op in selected])
        effective=preserve_anchors(d.validate_graph(effective_raw,stories[sid],unit,priors[sid]),original_row['graph'])
        if not selected and effective!=original_row['graph']: raise ValueError('Retained final graph has churn')
        effective_ids[sid].update(x['id'] for family in d.COLLECTIONS for x in effective[family])
        for ident,path in embedded_references(effective):
            reference_count+=1
            if ident not in effective_ids[sid]:
                raise ValueError('Unavailable embedded reference: '+uid+' '+path+' '+ident)
        if not decision['operations'] and decision['graph']!=original_row['graph']:
            raise ValueError('Unchanged decision has graph churn: '+uid)
        expected_status='corrected' if decision['operations'] else ('retained_with_open_question' if decision['unresolved'] else 'retained')
        if decision['status']!=expected_status: raise ValueError('Review status differs from decision: '+uid)
        row=deepcopy(original_row)
        row.update({'interpretation_view':'independently_reviewed_prefix_v1',
                    'baseline_record_hash':d.object_hash(original_row),'baseline_graph_hash':original_row['graph_hash'],
                    'baseline_compiled_parent_graph_hash':original_row['compiled_parent_graph_hash'],
                    'compiled_parent_graph_hash':priors[sid][-1]['graph_hash'] if priors[sid] else None,
                    'graph':effective,'graph_hash':d.object_hash(effective),
                    'proposed_review_graph_hash':decision['graph_hash'],
                    'adoption_manifest_hash':d.object_hash(adoption),
                    'review_actor':decision['actor'],'review_decision_hash':d.object_hash(decision),
                    'semantic_review':'independent_model_review_complete; human_review_pending'})
        reviewed.append(row); priors[sid].append(row); proposal_priors[sid].append({'graph':decision['graph']}); unit_positions[uid]=len(reviewed)-1
        provenance={'story_id':sid,'unit_id':uid,'available_at_token':src['available_at_token'],
                    'available_at_seconds':src['available_at_seconds'],'baseline_graph_hash':original_row['graph_hash'],
                    'reviewed_graph_hash':row['graph_hash'],'proposed_review_graph_hash':decision['graph_hash'],'decision_hash':d.object_hash(decision)}
        final_status='corrected' if selected else ('retained_with_open_question' if decision['unresolved'] else 'retained')
        coverage.append({**provenance,'actor':decision['actor'],'status':final_status,'proposal_status':decision['status'],
                         'source_hash':d.object_hash(src),'interpretation_hash':ih,
                         'interpretation':interpretation['interpretation'],'assessment':decision['assessment'],
                         'operation_count':len(selected),'proposed_operation_count':len(decision['operations']),
                         'excluded_proposal_count':len(declined),'new_open_questions':len(decision['unresolved']),
                         'chronology':dict(zip(['source_exposed','interpretation_committed','graph_exposed','decision_committed'],times)),
                         'trace_directory':'review/'+sid})
        for n,operation in selected:
            changes.append({**provenance,'change_id':uid+f'_change{n:02}',
                            'affected_ids':[operation['id']],'operation':deepcopy(operation),
                            'source_evidence':d.anchor(operation['evidence'],stories[sid],unit)})
        for n,operation,rule in declined:
            excluded_proposals.append({**provenance,'proposal_id':uid+f'_change{n:02}',
                                       'operation':operation,'adoption_decision':rule,
                                       'source_evidence':d.anchor(operation['evidence'],stories[sid],unit)})
        for n,issue in enumerate(decision['unresolved']):
            issue=deepcopy(issue); issue_id=uid+f'_issue{n:02}'
            if 'affected_ids' not in issue:
                binding=binding_index[issue_id]
                issue['affected_ids']=binding['affected_ids']; issue['target_binding_provenance']=binding
            for ident in issue['affected_ids']:
                if ident not in effective_ids[sid]: raise ValueError('Unavailable unresolved target: '+ident)
            unresolved.append({**provenance,'issue_id':issue_id,**issue,
                               'source_evidence':d.anchor(issue['evidence'],stories[sid],unit)})
    for sid,story in stories.items():
        expected=[]
        for row in priors[sid]:
            uid=row['unit_id']
            expected.extend((kind,uid) for kind in ('source_exposed','interpretation_committed','graph_exposed','decision_committed'))
        trace=r.rows(r.folder(sid)/'trace.jsonl') if (r.folder(sid)/'trace.jsonl').exists() else []
        actual=[(x['kind'],x['unit_id']) for x in trace]
        if actual[:len(expected)]!=expected: raise ValueError('Trace ordering differs: '+sid)
        if require_complete and actual!=expected: raise ValueError('Extra trace events: '+sid)
        for event in trace[:len(expected)]:
            uid=event['unit_id']
            if event['kind']=='interpretation_committed':
                if event['interpretation_hash']!=decisions[uid]['interpretation_hash']: raise ValueError('Trace interpretation hash differs')
            if event['kind']=='decision_committed':
                if event['decision_hash']!=d.object_hash(decisions[uid]) or event['patches']!=len(decisions[uid]['operations']):
                    raise ValueError('Trace decision hash differs')
    for change in changes:
        wanted=set(change['affected_ids']); later=[]
        for row in reviewed[unit_positions[change['unit_id']]+1:]:
            if row['story_id']!=change['story_id']: continue
            matches=list(all_refs(row['graph'],wanted))
            if matches: later.append({'unit_id':row['unit_id'],'references':[{'id':i,'path':p} for i,p in matches]})
        change['later_reference_check']={'status':'replayed_against_corrected_prefix','units':later,
                                          'later_references_count':sum(len(x['references']) for x in later)}
        if change['operation']['op']=='remove' and later: raise ValueError('Removed ID still referenced')
    if require_complete and {x['operation']['id'] for x in excluded_proposals}!={x['id'] for x in adoption['exclusions']}:
        raise ValueError('Adoption exclusion did not match a saved proposal')
    summary={'protocol':'independent-review-audit-v1','complete':len(reviewed)==len(source),
             'reviewed_units':len(reviewed),'total_units':len(source),
             'corrected_units':sum(bool(x['operation_count']) for x in coverage),
             'operations':len(changes),'new_open_questions':len(unresolved),
             'excluded_redundant_proposals':len(excluded_proposals),
             'coverage_by_story':[{'story_id':sid,'reviewed':len(priors[sid]),'total':len(stories[sid]['units']),
                                   'corrected':sum(x['story_id']==sid and x['operation_count']>0 for x in coverage)} for sid in stories],
             'checks':{'baseline':integrity,'source_first_receipts':True,'patch_preconditions':True,
                       'reference_and_context_replay':True,'evidence_and_trigger_coordinates':True,
                       'unit_availability_preserved':True,'unchanged_graphs_exact':True,
                       'later_references_replayed':True},
             'embedded_reference_strings_checked':reference_count,
             'human_validation':False,'semantic_accuracy_measured':False}
    return {'source':source,'baseline':original,'stories':stories,'decisions':decisions,
            'records':reviewed,'coverage':coverage,'changes':changes,'unresolved':unresolved,'summary':summary,
            'adoption':adoption,'excluded_proposals':excluded_proposals,'question_bindings':bindings}


def assessment_cases(value, id_units):
    """Bind a retrospective example to its latest referenced availability point.

    A multi-unit assessment must never be shown at its first unit: its prose can
    disclose later revelations before a human interprets that later source.
    """
    if isinstance(value,dict):
        units=[value['unit_id']] if isinstance(value.get('unit_id'),str) else []
        for key in ('units','unit_ids'):
            if isinstance(value.get(key),list):
                units.extend(x for x in value[key] if isinstance(x,str) and re.fullmatch(r'story_\d{2}_u\d{4}',x))
        for key in ('ids','affected_ids'):
            for ident in value.get(key,[]):
                if ident in id_units: units.append(id_units[ident])
        if units:
            # A note can name an additional later graph ID in explanatory prose.
            for ident in re.findall(r'story_\d{2}:[A-Za-z0-9_:-]+',json.dumps(value)):
                if ident in id_units: units.append(id_units[ident])
            if len({x.split('_u')[0] for x in units})!=1: raise ValueError('Cross-story assessment case')
            yield {'available_after_unit':max(units),'assessment':value}
        else:
            for v in value.values(): yield from assessment_cases(v,id_units)
    elif isinstance(value,list):
        for v in value: yield from assessment_cases(v,id_units)


def human_packet(audit_result, assessments, overrides):
    a=audit_result; records={x['unit_id']:x for x in a['records']}; baseline={x['unit_id']:x for x in a['baseline']}
    source={x['unit_id']:x for x in a['source']}; selected=defaultdict(set)
    id_units={item['id']:row['unit_id'] for row in a['baseline']+a['records'] for family in d.COLLECTIONS for item in row['graph'][family]}
    notes=defaultdict(list)
    last_units={sid:max(x['unit_id'] for x in a['records'] if x['story_id']==sid) for sid in a['stories']}
    for x in a['changes']: selected[x['unit_id']].add('correction')
    for x in a['unresolved']: selected[x['unit_id']].add('open_question')
    for x in a.get('excluded_proposals',[]): selected[x['unit_id']].add('excluded_redundant_proposal')
    for sid,assessment in assessments.items():
        for case in assessment_cases(assessment,id_units):
            uid=case['available_after_unit']
            if uid not in records: raise ValueError('Assessment references unavailable unit: '+uid)
            selected[uid].add('story_assessment_case'); notes[sid].append(case)
        # Retrospective prose can mention later outcomes without naming their IDs.
        # Never rely on ID extraction alone to prevent that disclosure.
        selected[last_units[sid]].add('retrospective_story_assessment')
    for x in overrides:
        selected[x['unit_id']].add('consequential_scope_definition')
    # Reproducible label-independent sampling among units with no review flag.
    seed='independent-review-v1-unflagged-2026-10'
    for sid in a['stories']:
        candidates=[x['unit_id'] for x in a['records'] if x['story_id']==sid and x['unit_id'] not in selected
                    and x['graph']['coverage']=='annotated' and not a['decisions'][x['unit_id']]['operations']
                    and not a['decisions'][x['unit_id']]['unresolved']]
        candidates.sort(key=lambda uid:sha((seed+'|'+uid).encode()))
        if len(candidates)<3: raise ValueError('Insufficient unflagged sample')
        for uid in candidates[:3]: selected[uid].add('unflagged_sample')
    packet=[]
    for uid in sorted(selected):
        src=source[uid]; sid=src['story_id']; unit=src['unit']
        packet.append({'unit_id':uid,'story_id':sid,'selection_reasons':sorted(selected[uid]),
                       'sampling_seed':seed if 'unflagged_sample' in selected[uid] else None,
                       'available_at_token':src['available_at_token'],'available_at_seconds':src['available_at_seconds'],
                       'prior_source':a['stories'][sid]['text'][:unit['char_start']], 'current_source':src['text'],
                       'independent_model_interpretation':next(x['interpretation'] for x in a['coverage'] if x['unit_id']==uid),
                       'baseline_graph':baseline[uid]['graph'],'reviewed_graph':records[uid]['graph'],
                       'comparison':a['decisions'][uid]['assessment'],
                       'story_assessment_notes':notes[sid] if uid==last_units[sid] else [],
                        'changes':[{k:v for k,v in x.items() if k!='later_reference_check'}
                                   for x in a['changes'] if x['unit_id']==uid],
                        'later_reference_checks':[{'change_id':x['change_id'],
                                                   'check':x['later_reference_check']}
                                                  for x in a['changes'] if x['story_id']==sid]
                                                 if uid==last_units[sid] else [],
                       'excluded_proposals':[x for x in a.get('excluded_proposals',[]) if x['unit_id']==uid],
                       'open_questions':[x for x in a['unresolved'] if x['unit_id']==uid],
                       'scope_definitions':[x for x in overrides if x['unit_id']==uid],
                       'human_response':None})
    return packet


def verify_packet(packet, a):
    """Verify selection completeness and the source availability of shown notes."""
    byunit={x['unit_id']:x for x in packet}; sources={x['unit_id']:x for x in a['source']}
    last_units={sid:max(x['unit_id'] for x in a['records'] if x['story_id']==sid) for sid in a['stories']}
    if len(byunit)!=len(packet): raise ValueError('Duplicate human card')
    for family in ('changes','unresolved','excluded_proposals'):
        if any(x['unit_id'] not in byunit for x in a.get(family,[])):
            raise ValueError('Human packet omitted a review flag: '+family)
    sample=Counter()
    for card in packet:
        src=sources[card['unit_id']]; story=a['stories'][card['story_id']]
        if card['current_source']!=src['text'] or card['prior_source']!=story['text'][:src['unit']['char_start']]:
            raise ValueError('Human packet source prefix differs')
        if (card['available_at_token'],card['available_at_seconds'])!=(src['available_at_token'],src['available_at_seconds']):
            raise ValueError('Human packet availability differs')
        for case in card['story_assessment_notes']:
            if card['unit_id']!=last_units[card['story_id']] or case['available_after_unit']>card['unit_id']:
                raise ValueError('Premature retrospective assessment disclosure')
        if any('later_reference_check' in x for x in card['changes']):
            raise ValueError('Future reference metadata included in a prefix correction')
        if card['later_reference_checks'] and card['unit_id']!=last_units[card['story_id']]:
            raise ValueError('Premature later-reference disclosure')
        if 'unflagged_sample' in card['selection_reasons']:
            if len(card['selection_reasons'])!=1 or card['changes'] or card['open_questions'] or card['scope_definitions']:
                raise ValueError('Flagged unit included in unflagged sample')
            sample[card['story_id']]+=1
    if dict(sample)!={sid:3 for sid in a['stories']}: raise ValueError('Incomplete unflagged human sample')
    return {'source_prefixes_checked':len(packet),'unflagged_per_story':3,'retrospective_notes_only_at_story_end':True,'human_validation':False}


def human_html(packet):
    # One increasing sequence per story, source interpretation required before graph reveal.
    payload=json.dumps(packet,ensure_ascii=False,sort_keys=True).replace('<','\\u003c')
    return ('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent review: human adjudication</title>
<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}textarea{display:block;width:100%;min-height:110px;margin:12px 0}button,select{padding:9px;margin:5px}#source{background:#edf2ff;padding:20px;white-space:pre-wrap}details{margin:12px 0} .note{color:#555}</style>
<h1>Human adjudication packet</h1><p>Review in story order. Write your source-based interpretation before revealing the model judgments. Prior source is available below; future source is not displayed. Whole-story assessments appear only after the final source interpretation, because their prose may describe later outcomes. This browser gate records procedure, not a security boundary. No response is submitted anywhere.</p>
<p class="note">Select a story you have not read ahead in. A fresh reviewer is needed for a fresh prefix judgment after later material has been seen. Export your notes before closing; notes are kept in memory only.</p>
<label>Story <select id="story"></select></label><button id="download">Export human notes</button><h2 id="title"></h2>
<details><summary>Prior source through the preceding unit</summary><pre id="prefix"></pre></details><div id="source"></div>
<label>Independent interpretation<textarea id="interpretation" placeholder="Meaning, scope, roles, reference and uncertainties"></textarea></label>
<button id="reveal">Save interpretation and reveal annotations</button><div id="graphs" hidden></div>
<div id="judgment" hidden><label>Human judgment<textarea id="decision" placeholder="Accept / revise / unresolved, affected IDs, reason and source evidence"></textarea></label><button id="next">Save judgment and next selected unit</button></div>
<script>const packetSha='''+json.dumps(sha(d.jsonl_bytes(packet)))+'''; const cards='''+payload+'''; const notes={}; const drafts={}; const progress={}; let current=null; let frozen=null;
const ids=[...new Set(cards.map(x=>x.story_id))]; const $=id=>document.getElementById(id);
for(const id of ids){const o=document.createElement('option');o.value=id;o.textContent=id;$('story').append(o);progress[id]=0;}
function show(){const sid=$('story').value;const list=cards.filter(x=>x.story_id===sid);current=list[progress[sid]];frozen=null;$('graphs').hidden=true;$('judgment').hidden=true;$('graphs').replaceChildren();$('interpretation').disabled=false;$('interpretation').value='';$('decision').value='';$('reveal').disabled=!current;
if(!current){$('title').textContent=sid+' complete';$('source').textContent='Export notes to preserve your work.';$('prefix').textContent='';return;}
$('title').textContent=current.unit_id+' ('+(progress[sid]+1)+'/'+list.length+')';$('prefix').textContent=current.prior_source;$('source').textContent=current.current_source;
if(drafts[current.unit_id]){frozen=drafts[current.unit_id];$('interpretation').value=frozen.interpretation;displayGraphs();}}
function displayGraphs(){$('interpretation').disabled=true;$('reveal').disabled=true;
for(const [label,value] of [['Baseline graph',current.baseline_graph],['Reviewed graph',current.reviewed_graph],['Model comparison and evidence',{interpretation:current.independent_model_interpretation,comparison:current.comparison,story_assessment_notes:current.story_assessment_notes,changes:current.changes,later_reference_checks:current.later_reference_checks,excluded_proposals:current.excluded_proposals,open_questions:current.open_questions,scope_definitions:current.scope_definitions,selection_reasons:current.selection_reasons}]]){const box=document.createElement('details');const heading=document.createElement('summary');heading.textContent=label;const pre=document.createElement('pre');pre.textContent=JSON.stringify(value,null,2);box.append(heading,pre);$('graphs').append(box);}$('graphs').hidden=false;$('judgment').hidden=false;}
$('story').onchange=show;$('reveal').onclick=()=>{if(!$('interpretation').value.trim()){alert('Write the independent interpretation first.');return;}frozen={interpretation:$('interpretation').value,committed_utc:new Date().toISOString()};drafts[current.unit_id]=frozen;displayGraphs();};
$('next').onclick=()=>{if(!$('decision').value.trim()){alert('Record accept, revise or unresolved with a reason.');return;}notes[current.unit_id]={unit_id:current.unit_id,...frozen,judgment:$('decision').value,judged_utc:new Date().toISOString(),later_selected_source_not_yet_displayed:true};progress[current.story_id]++;show();};
$('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({protocol:'human-prefix-adjudication-v1',packet_sha256:packetSha,responses:Object.values(notes)},null,2)],{type:'application/json'}));a.download='human-review-notes.json';a.click();URL.revokeObjectURL(a.href);};show();</script></html>''').encode('utf-8')


def enriched_scope_document(document, records, source):
    result=deepcopy(document)
    locations={item['id']:row for row in records for family in d.COLLECTIONS for item in row['graph'][family]}
    items={item['id']:item for row in records for family in d.COLLECTIONS for item in row['graph'][family]}
    sources={x['unit_id']:x for x in source}
    for case in result['cases']:
        src=sources[case['unit_id']]
        for ident in case['affected_ids']:
            if ident not in locations: raise ValueError('Unknown scope example: '+ident)
            if locations[ident]['story_id']!=src['story_id'] or locations[ident]['available_at_token']>src['available_at_token']:
                raise ValueError('Future scope reference: '+ident)
        if not set(case['evidence_refs'])<=set(case['affected_ids']): raise ValueError('Unlisted scope evidence')
        case.update({'available_at_token':src['available_at_token'],'available_at_seconds':src['available_at_seconds'],
                     'source_evidence':[items[ident]['evidence'] for ident in case['evidence_refs']],
                     'integration_provenance':'Retrospective compiler interpretation based on saved source-first review; raw graph unchanged.'})
    return result


def build():
    a=audit(True)
    assessments={sid:r.read(r.folder(sid)/'assessment.json') for sid in a['stories']}
    if any(value['story_id']!=sid for sid,value in assessments.items()): raise ValueError('Story assessment identity differs')
    scope_document=enriched_scope_document(r.read(d.ROOT/'docs/independent_review_scope.json'),a['records'],a['source'])
    overrides=scope_document['cases']
    modal_cases=list(interface.modal_bindings(a['records']))
    orientation_cases=list(interface.relation_orientation_bindings(a['records']))
    expected_modals={x for ids in interface.MODAL_DENIALS.values() for x in ids}
    if {x['evidence_refs'][0] for x in modal_cases}!=expected_modals:
        raise ValueError('Incomplete modal-scope case coverage')
    if {x['target'] for x in orientation_cases}!=set(interface.RESTATEMENT_ORIENTATION)|interface.RELATION_REVERSE_IDS:
        raise ValueError('Incomplete relation orientation coverage')
    packets={x['unit_id']:{'unit':x['unit'],'current_text':x['text']} for x in a['source']}
    splits={x['story_id']:x['split'] for x in a['source']}
    compiled=d.compile_records(a['records'],packets,a['stories'],splits)
    if compiled['source-only.jsonl']!=a['source']: raise ValueError('Source/timing flag compilation changed')
    files={name:d.jsonl_bytes(values) for name,values in compiled.items()}
    files['coverage.jsonl']=d.jsonl_bytes(a['coverage'])
    files['changes.jsonl']=d.jsonl_bytes(a['changes'])
    files['adoption.json']=json_bytes(a['adoption'])
    files['question-bindings.json']=json_bytes(a['question_bindings'])
    files['excluded-proposals.jsonl']=d.jsonl_bytes(a['excluded_proposals'])
    files['unresolved.jsonl']=d.jsonl_bytes(a['unresolved'])
    files['verification.json']=json_bytes(a['summary'])
    files['interface/definitions.json']=json_bytes(interface.declarations())
    files['interface/label-inventory.json']=json_bytes(interface.inventory(a['records'],splits))
    files['interface/normalized-records.jsonl']=d.jsonl_bytes(list(interface.alias_view(a['records'])))
    files['interface/scope-bindings.jsonl']=d.jsonl_bytes(list(interface.scope_bindings(a['records'])))
    files['interface/scope-cases.json']=json_bytes(scope_document)
    files['interface/modal-scope-bindings.jsonl']=d.jsonl_bytes(modal_cases)
    files['interface/relation-orientations.jsonl']=d.jsonl_bytes(orientation_cases)
    packet=human_packet(a,assessments,overrides+modal_cases+orientation_cases)
    packet_check=verify_packet(packet,a)
    files['human-review/packet.jsonl']=d.jsonl_bytes(packet)
    files['human-review/index.html']=human_html(packet)
    files['human-review/selection.json']=json_bytes({'units':len(packet),'sample_units':sum('unflagged_sample' in x['selection_reasons'] for x in packet),
                                                   'packet_sha256':sha(d.jsonl_bytes(packet)),
                                                   'by_story':dict(Counter(x['story_id'] for x in packet)),
                                                   'human_responses_received':0,'status':'pending','verification':packet_check})
    for sid,story in a['stories'].items():
        files['sources/'+sid+'.json']=json_bytes(story)
        for path in sorted(r.folder(sid).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.jsonl'}:
                files['review/'+sid+'/'+path.relative_to(r.folder(sid)).as_posix()]=path.read_bytes()
    for name in ('baseline.json','baseline-verification.json'):
        files[name]=(r.OUT/name).read_bytes()
    files['baseline/annotations.jsonl']=(r.baseline()/'author-corrected/annotations.jsonl').read_bytes()
    files['baseline/original-annotations.jsonl']=(r.baseline()/'annotations.jsonl').read_bytes()
    files['schema/accepted-graph.json']=json_bytes(d.anchored_graph_schema())
    for name in ('docs/independent_review_protocol.md','docs/independent_review_interface.md',
                 'docs/independent_review_scope.json','docs/independent_review_handoff.md',
                 'docs/PROJECT_HANDOFF.md','docs/direct_annotation.md','docs/direct_annotation_conventions.md','docs/annotation.md',
                 'neurosym/independent_review.py','neurosym/independent_review_export.py',
                 'neurosym/review_semantic_interface.py','neurosym/direct_annotations.py',
                 'neurosym/__init__.py','neurosym/corpus.py','neurosym/deniz.py','neurosym/io.py',
                 'scripts/independent_review.py','scripts/export_independent_review.py',
                 'scripts/package_independent_review.py',
                 'scripts/review_graph_slice.py','scripts/verify_independent_baseline.py',
                 'scripts/check_independent_review_tools.py','scripts/review_inventory.py',
                 'scripts/review_label_examples.py','scripts/review_scope_audit.py',
                 'scripts/review_inspect.py','scripts/review_changes.py','scripts/review_open_questions.py','scripts/check_review_embedded_refs.py',
                 'scripts/check_review_packet.js','scripts/preview_independent_review.py'):
        files['code/'+name]=(d.ROOT/name).read_bytes()
    import importlib.metadata
    files['runtime.json']=json_bytes({'python':platform.python_version(),
        'dependencies':{name:importlib.metadata.version(name) for name in ('pydantic','numpy','h5py','praatio')},
        'model_family':'GPT-6-based Codex','exact_model_revision':None,'export_calls_model':False})
    manifest={'protocol':'independent-semantic-review-export-v1','baseline':r.initialize()['baseline'],
              'recommended_annotation':'annotations.jsonl','interface':'interface/definitions.json',
              'scope_cases':'interface/scope-cases.json','coverage':'coverage.jsonl','changes':'changes.jsonl',
              'unresolved':'unresolved.jsonl','adoption':'adoption.json',
              'interpretation_constraints':['interface/definitions.json','interface/scope-cases.json','unresolved.jsonl'],
              'human_packet':'human-review/index.html','reviewed_units':len(a['records']),
              'corrected_units':a['summary']['corrected_units'],'operations':len(a['changes']),
              'excluded_redundant_proposals':len(a['excluded_proposals']),
              'new_open_questions':len(a['unresolved']),'human_validation':False,
              'semantic_accuracy_measured':False,'scientific_scope':'Concepts, relations, roles, reference and discourse for encoding, decoding and representational comparisons.',
              'heldout':['story_11'],'original_exports_modified':False,
              'review_limitations':'Fresh story contexts and immutable source-first receipts within one model system; no proof of no pretrained familiarity or filesystem-level isolation. Global interface review is retrospective and separate from prefix judgments.'}
    files['manifest.json']=json_bytes(manifest)
    files['README.md']=('Recommended input: `annotations.jsonl`. Baseline and original are preserved under `baseline/`.\n\n'
                        'Read `code/docs/independent_review_handoff.md` and the interface definitions before implementing analyses. '
                        'Full coverage: `coverage.jsonl`; surgical changes: `changes.jsonl`; open review notes: `unresolved.jsonl`; '
                        'existing uncertainty annotations remain in `uncertainties.jsonl`. '
                        'Adoption exclusions are explicit in `adoption.json` and `excluded-proposals.jsonl`. '
                        'Use graph and review uncertainties at their recorded availability points. '
                        'Human review is pending; start with `human-review/index.html`.\n').encode()
    checksums={name:sha(data) for name,data in sorted(files.items())}
    fingerprint=d.object_hash({'protocol':manifest['protocol'],'files':checksums})
    out=r.OUT/'exports'/fingerprint[:20]
    for name,data in files.items(): d.immutable_bytes(out/name,data)
    d.immutable_bytes(out/'checksums.json',json_bytes(checksums))
    d.immutable_bytes(out/'snapshot.json',json_bytes({'build_hash':fingerprint,'checksums_sha256':sha(json_bytes(checksums))}))
    result=verify_export(out)
    latest={'build_hash':fingerprint,'directory':out.relative_to(d.ROOT).as_posix(),
            'recommended_input':(out/'annotations.jsonl').relative_to(d.ROOT).as_posix(),
            'complete':True,'reviewed_units':len(a['records']),'verification':'passed'}
    (r.OUT/'latest.json').write_bytes(json_bytes(latest))
    return {'latest':latest,'verification':result}


def verify_export(out):
    out=Path(out)
    checks=r.read(out/'checksums.json'); snapshot=r.read(out/'snapshot.json')
    if snapshot['checksums_sha256']!=sha((out/'checksums.json').read_bytes()): raise ValueError('Checksum manifest mismatch')
    for name,digest in checks.items():
        if sha((out/name).read_bytes())!=digest: raise ValueError('Export checksum mismatch: '+name)
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if actual!=set(checks)|{'checksums.json','snapshot.json'}: raise ValueError('Unlisted export files')
    manifest=r.read(out/'manifest.json')
    if d.object_hash({'protocol':manifest['protocol'],'files':checks})!=snapshot['build_hash']: raise ValueError('Build hash mismatch')
    rows=r.rows(out/'annotations.jsonl'); source=r.rows(out/'source-only.jsonl')
    pinned=r.read(out/'baseline.json')['file_sha256']
    if sha((out/'source-only.jsonl').read_bytes())!=pinned['source-only.jsonl']:
        raise ValueError('Frozen source/timing flags changed')
    if sha((out/'baseline/annotations.jsonl').read_bytes())!=pinned['author-corrected/annotations.jsonl']:
        raise ValueError('Frozen baseline annotations changed')
    stories={sid:r.read(out/'sources'/(sid+'.json')) for sid in {x['story_id'] for x in rows}}
    packets={x['unit_id']:{'unit':x['unit'],'current_text':x['text']} for x in source}
    splits={x['story_id']:x['split'] for x in source}
    compiled=d.compile_records(rows,packets,stories,splits)
    for name,values in compiled.items():
        if d.jsonl_bytes(values)!=(out/name).read_bytes(): raise ValueError('Lossless compilation differs: '+name)
    if len(rows)!=1217 or len({x['unit_id'] for x in rows})!=1217 or len(stories)!=11: raise ValueError('Incomplete export coverage')
    if interface.declarations()!=r.read(out/'interface/definitions.json'): raise ValueError('Interface version differs')
    if d.jsonl_bytes(list(interface.alias_view(rows)))!=(out/'interface/normalized-records.jsonl').read_bytes(): raise ValueError('Alias view differs')
    if d.jsonl_bytes(list(interface.scope_bindings(rows)))!=(out/'interface/scope-bindings.jsonl').read_bytes(): raise ValueError('Scope bindings differ')
    if d.jsonl_bytes(list(interface.modal_bindings(rows)))!=(out/'interface/modal-scope-bindings.jsonl').read_bytes(): raise ValueError('Modal bindings differ')
    if d.jsonl_bytes(list(interface.relation_orientation_bindings(rows)))!=(out/'interface/relation-orientations.jsonl').read_bytes(): raise ValueError('Relation orientations differ')
    if interface.inventory(rows,splits)!=r.read(out/'interface/label-inventory.json'): raise ValueError('Label inventory differs')
    scope_document=enriched_scope_document(r.read(out/'code/docs/independent_review_scope.json'),rows,source)
    if scope_document!=r.read(out/'interface/scope-cases.json'): raise ValueError('Scope case evidence/availability differs')
    # Rebuild from archived receipts and archived baseline; independent of live review work.
    baseline={x['unit_id']:x for x in r.rows(out/'baseline/annotations.jsonl')}; priors=defaultdict(list)
    receipt_times={}; decision_hashes={}; known_ids=defaultdict(set); proposal_priors=defaultdict(list)
    adoption=r.read(out/'adoption.json')
    coverage=r.rows(out/'coverage.jsonl'); coverage_index={x['unit_id']:x for x in coverage}
    changes=r.rows(out/'changes.jsonl'); change_index={x['change_id']:x for x in changes}
    excluded=r.rows(out/'excluded-proposals.jsonl'); excluded_index={x['proposal_id']:x for x in excluded}
    archived_decisions={}; expected_changes=set(); expected_excluded=set()
    if len(coverage)!=len(rows) or len(coverage_index)!=len(rows): raise ValueError('Coverage record incomplete')
    for row,src in zip(rows,source,strict=True):
        uid=row['unit_id']; sid=row['story_id']; unit=src['unit']
        folder=out/'review'/sid
        decision=r.read(folder/'decisions'/(uid+'.json'))
        archived_decisions[uid]=decision
        exposure=r.read(folder/'exposures'/(uid+'.json'))
        interpretation=r.read(folder/'interpretations'/(uid+'.json'))
        graph_exposure=r.read(folder/'graph-exposures'/(uid+'.json'))
        if interpretation['source_hash']!=d.object_hash(src) or exposure['source_hash']!=d.object_hash(src): raise ValueError('Archived source receipt differs')
        if graph_exposure['baseline_graph_hash']!=baseline[uid]['graph_hash']: raise ValueError('Archived graph exposure differs')
        ih=d.object_hash(interpretation)
        if decision['interpretation_hash']!=ih or graph_exposure['interpretation_hash']!=ih: raise ValueError('Archived interpretation differs')
        times=[exposure['exposed_utc'],interpretation['committed_utc'],graph_exposure['exposed_utc'],decision['committed_utc']]
        if times!=sorted(times) or receipt_times.get(sid,'')>times[0]: raise ValueError('Archived source-first chronology failed')
        if exposure['prior_decision_hash']!=decision_hashes.get(sid): raise ValueError('Archived review chain failed')
        receipt_times[sid]=times[-1]; decision_hashes[sid]=d.object_hash(decision)
        if unit!=stories[sid]['units'][len(priors[sid])] or row['available_at_token']!=unit['end_token']:
            raise ValueError('Archived unit availability differs')
        if row['available_at_seconds']!=src['available_at_seconds'] or decision['available_at_seconds']!=src['available_at_seconds']:
            raise ValueError('Archived timing differs')
        raw=r.apply_operations(r.unanchor(baseline[uid]['graph']),decision['operations'])
        proposal=d.validate_graph(raw,stories[sid],unit,proposal_priors[sid])
        if r.unanchor(proposal)!=r.unanchor(decision['graph']): raise ValueError('Archived proposal replay differs')
        selected,declined=selected_operations(decision,adoption)
        for n,operation in selected:
            change_id=uid+f'_change{n:02}'; expected_changes.add(change_id); saved=change_index[change_id]
            if saved['operation']!=operation or saved['source_evidence']!=d.anchor(operation['evidence'],stories[sid],unit):
                raise ValueError('Archived changelog differs: '+change_id)
            if saved['reviewed_graph_hash']!=row['graph_hash']: raise ValueError('Changelog graph provenance differs')
        for n,operation,rule in declined:
            proposal_id=uid+f'_change{n:02}'; expected_excluded.add(proposal_id); saved=excluded_index[proposal_id]
            if saved['operation']!=operation or saved['adoption_decision']!=rule: raise ValueError('Excluded proposal log differs')
        covered=coverage_index[uid]
        if covered['operation_count']!=len(selected) or covered['excluded_proposal_count']!=len(declined):
            raise ValueError('Coverage operation count differs')
        if covered['interpretation']!=interpretation['interpretation'] or covered['interpretation_hash']!=ih:
            raise ValueError('Coverage interpretation differs')
        if covered['reviewed_graph_hash']!=row['graph_hash'] or covered['assessment']!=decision['assessment']:
            raise ValueError('Coverage graph/comparison differs')
        effective_raw=r.apply_operations(r.unanchor(baseline[uid]['graph']),[op for _,op in selected])
        replay=d.validate_graph(effective_raw,stories[sid],unit,priors[sid])
        if preserve_anchors(replay,baseline[uid]['graph'])!=row['graph']: raise ValueError('Archived adopted graph replay differs')
        if row['graph_hash']!=d.object_hash(row['graph']) or row['proposed_review_graph_hash']!=decision['graph_hash']:
            raise ValueError('Adopted/proposed graph hash differs')
        if d.object_hash(decision['graph'])!=decision['graph_hash']: raise ValueError('Saved proposal graph hash differs')
        if row['baseline_record_hash']!=d.object_hash(baseline[uid]): raise ValueError('Baseline row provenance differs')
        if row['adoption_manifest_hash']!=d.object_hash(adoption): raise ValueError('Adoption provenance differs')
        known_ids[sid].update(x['id'] for family in d.COLLECTIONS for x in row['graph'][family])
        for ident,path in embedded_references(row['graph']):
            if ident not in known_ids[sid]: raise ValueError('Archived embedded reference unavailable: '+ident)
        for family in d.COLLECTIONS:
            for actual_item,replay_item in zip(row['graph'][family],replay[family],strict=True):
                for key in ('evidence','trigger'):
                    if key in actual_item:
                        for field in ('quote','start_token','end_token','char_start','char_end','verbatim'):
                            if actual_item[key][field]!=replay_item[key][field]: raise ValueError('Archived evidence coordinates differ')
        if row['compiled_parent_graph_hash']!=(priors[sid][-1]['graph_hash'] if priors[sid] else None): raise ValueError('Compilation chain broken')
        if row['operational_parent_graph_hash']!=baseline[uid]['operational_parent_graph_hash']: raise ValueError('Operational history rewritten')
        if row['review_decision_hash']!=d.object_hash(decision): raise ValueError('Decision provenance broken')
        priors[sid].append(row)
        proposal_priors[sid].append({'graph':decision['graph']})
    if expected_changes!=set(change_index) or len(changes)!=len(expected_changes): raise ValueError('Changelog inventory differs')
    if expected_excluded!=set(excluded_index) or len(excluded)!=len(expected_excluded): raise ValueError('Excluded proposal inventory differs')
    if {x['operation']['id'] for x in excluded}!={x['id'] for x in adoption['exclusions']}: raise ValueError('Adoption inventory differs')
    if manifest['operations']!=len(changes) or manifest['excluded_redundant_proposals']!=len(excluded): raise ValueError('Manifest operation counts differ')
    archived={'records':rows,'baseline':list(baseline.values()),'source':source,'stories':stories,
              'decisions':archived_decisions,'coverage':coverage,'changes':changes,'excluded_proposals':excluded,
              'unresolved':r.rows(out/'unresolved.jsonl')}
    assessments={sid:r.read(out/'review'/sid/'assessment.json') for sid in stories}
    packet=human_packet(archived,assessments,scope_document['cases']+list(interface.modal_bindings(rows))+list(interface.relation_orientation_bindings(rows)))
    verify_packet(packet,archived)
    if d.jsonl_bytes(packet)!=(out/'human-review/packet.jsonl').read_bytes(): raise ValueError('Human packet reconstruction differs')
    if human_html(packet)!=(out/'human-review/index.html').read_bytes(): raise ValueError('Human page reconstruction differs')
    return {'verified':True,'files':len(checks),'units':len(rows),'stories':len(stories),
            'lossless_compilation':True,'archived_patch_replay':True,'interface_sidecar_replay':True,
            'coverage_and_changelog_replay':True,'human_packet_replay':True,
            'semantic_accuracy_measured':False,'human_validation':False}
