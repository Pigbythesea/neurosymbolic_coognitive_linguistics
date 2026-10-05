"""Render a clearly marked preview using only completed story reviews."""
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r
from neurosym import independent_review_export as e
from neurosym import review_semantic_interface as interface
from neurosym import direct_annotations as d

source,baseline,stories,decisions=e.load_inputs(False)
complete={sid for sid,story in stories.items() if len(r.reviewed(sid))==len(story['units']) and (r.folder(sid)/'assessment.json').exists()}
source=[x for x in source if x['story_id'] in complete]
baseline=[x for x in baseline if x['story_id'] in complete]
records=[]; changes=[]; excluded=[]; priors=defaultdict(list); adoption=e.adoption_manifest()
for row in baseline:
    uid=row['unit_id']; sid=row['story_id']; selected,declined=e.selected_operations(decisions[uid],adoption)
    raw=r.apply_operations(r.unanchor(row['graph']),[op for _,op in selected])
    graph=e.preserve_anchors(d.validate_graph(raw,stories[sid],stories[sid]['units'][len(priors[sid])],priors[sid]),row['graph'])
    records.append({**row,'graph':graph,'graph_hash':d.object_hash(graph)}); priors[sid].append(records[-1])
    changes.extend({'story_id':sid,'unit_id':uid,'change_id':uid+f'_change{n:02}',
                    'operation':op} for n,op in selected)
    excluded.extend({'unit_id':uid,'operation':op,'adoption_decision':rule} for _,op,rule in declined)
for change in changes:
    later=[]
    for row in records:
        if row['story_id']!=change['story_id'] or row['unit_id']<=change['unit_id']: continue
        matches=list(e.all_refs(row['graph'],{change['operation']['id']}))
        if matches: later.append({'unit_id':row['unit_id'],'references':[{'id':ident,'path':path} for ident,path in matches]})
    change['later_reference_check']={'status':'replayed_against_corrected_prefix','units':later,
                                    'later_references_count':sum(len(x['references']) for x in later)}
unresolved=[{'unit_id':uid,'issues':x['unresolved']} for uid,x in decisions.items() if x['story_id'] in complete and x['unresolved']]
coverage=[r.read(r.folder(x['story_id'])/'interpretations'/(x['unit_id']+'.json')) for x in records]
a={'source':source,'baseline':baseline,'stories':{sid:stories[sid] for sid in complete},'decisions':decisions,'records':records,'changes':changes,'excluded_proposals':excluded,'unresolved':unresolved,'coverage':coverage}
assessments={sid:r.read(r.folder(sid)/'assessment.json') for sid in complete}
cases=[x for x in r.read(r.ROOT/'docs/independent_review_scope.json')['cases'] if x['unit_id'].split('_u')[0] in complete]
cases+=list(interface.modal_bindings(records))
cases+=list(interface.relation_orientation_bindings(records))
packet=e.human_packet(a,assessments,cases)
verification=e.verify_packet(packet,a)
page=e.human_html(packet)
out=r.OUT/'work'/'review-preview.html'; out.write_bytes(page)
script=page.decode().split('<script>',1)[1].split('</script>',1)[0]
out.with_suffix('.js').write_text(script,encoding='utf-8')
print(json.dumps({'preview':out.relative_to(r.ROOT).as_posix(),'complete_stories':sorted(complete),'cards':len(packet),'verification':verification,'recommended_export':False}))
