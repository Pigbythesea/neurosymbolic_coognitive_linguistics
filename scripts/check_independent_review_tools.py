"""Meaningful positive and rejection checks for review/export support code."""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d
from neurosym import independent_review as r
from neurosym import independent_review_export as e
from neurosym import review_semantic_interface as i


def rejects(fn):
    try: fn()
    except (ValueError,KeyError): return
    raise AssertionError('Invalid input was accepted')


def main():
    source,baseline,stories,decisions=e.load_inputs(False)
    index={x['unit_id']:x for x in baseline}
    checks=[]
    decision=decisions['story_06_u0043']; original=index[decision['unit_id']]
    raw=r.unanchor(original['graph'])
    applied=r.apply_operations(raw,decision['operations'])
    assert not any(x['id']=='story_06:u0043_men4' for x in applied['mentions'])
    assert raw==r.unanchor(original['graph'])
    checks.append('removal is surgical and input remains unchanged')
    bad=deepcopy(decision['operations']); bad[0]['before']['target']='story_06:wrong'
    rejects(lambda:r.apply_operations(raw,bad)); checks.append('mismatched patch precondition rejected')
    rejects(lambda:r.apply_operations(applied,decision['operations'])); checks.append('duplicate removal rejected')
    adoption=e.adoption_manifest()
    decision=decisions['story_11_u0116']; original=index[decision['unit_id']]
    selected,declined=e.selected_operations(decision,adoption)
    assert not selected and len(declined)==10
    assert r.apply_operations(r.unanchor(original['graph']),[])==r.unanchor(original['graph'])
    assert all(op['id'] in {x['id'] for x in decision['graph']['mentions']} for _,op,_ in declined)
    checks.append('adoption excludes redundant additions without altering saved proposals')
    decision=decisions['story_11_u0107']; selected,declined=e.selected_operations(decision,adoption)
    assert len(selected)==1 and len(declined)==1 and selected[0][1]['field']=='arguments'
    bad=deepcopy(adoption); next(x for x in bad['exclusions'] if x['id']==declined[0][1]['id'])['unit_id']='story_11_u0000'
    rejects(lambda:e.selected_operations(decision,bad))
    checks.append('adoption retains substantive changes and rejects mismatched exclusions')
    selected,declined=e.selected_operations(decisions['story_08_u0128'],adoption)
    assert not selected and len(declined)==1
    bad=deepcopy(decisions['story_08_u0128']); bad['operations'][0]['value']=[]
    rejects(lambda:e.selected_operations(bad,adoption))
    checks.append('context-copy exclusion matches only the exact declared proposal')
    cases=list(e.assessment_cases({'examples':[{'units':['story_11_u0115','story_11_u0116'],'finding':'later reveal'},
        {'ids':['story_11:future'],'finding':'ID-only example'}]}, {'story_11:future':'story_11_u0120'}))
    assert [x['available_after_unit'] for x in cases]==['story_11_u0116','story_11_u0120']
    checks.append('multi-unit and ID-only human notes wait until their latest source availability')
    first=index['story_04_u0000']; draft=r.unanchor(first['graph'])
    broken=deepcopy(draft); broken['events'][0]['arguments'][0]['target']='story_04:future_unavailable'
    rejects(lambda:d.validate_graph(broken,stories['story_04'],stories['story_04']['units'][0],[]))
    checks.append('unavailable reference rejected')
    rejects(lambda:d.anchor({'quote':'this quote is absent'},stories['story_04'],stories['story_04']['units'][0]))
    checks.append('unsupported evidence rejected')
    context={'c':{'parents':['c']}}
    rejects(lambda:i.paths_to_root('c',context)); checks.append('cyclic scope rejected')
    selected=[index['story_01_u0013'],index['story_06_u0001']]
    normalized=list(i.alias_view(selected))
    byid={x['id']:x for x in normalized}
    for ident in ('story_01:u0013_buy','story_06:u0001_buy'):
        view=byid[ident]; assert view['canonical_predicate']=='buy'
        assert [x['canonical_role'] for x in view['argument_aliases']][:2]==['buyer','purchased_object']
    assert all(x['original']==next(item for row in selected for family in d.COLLECTIONS for item in row['graph'][family] if item['id']==x['id']) for x in normalized)
    checks.append('cross-story purchase roles align while all original records survive')
    reordered=deepcopy(selected)
    for row in reordered: row['graph']=dict(reversed(list(row['graph'].items())))
    assert d.jsonl_bytes(list(i.alias_view(selected)))==d.jsonl_bytes(list(i.alias_view(reordered)))
    checks.append('alias export order survives JSON dictionary reordering')
    relation_rows=list(i.alias_view([index['story_05_u0042'],index['story_07_u0059'],index['story_01_u0048']]))
    byid={x['id']:x for x in relation_rows}
    forward=byid['story_05:u0042_rel2']; backward=byid['story_07:u0059_rel0']; example=byid['story_01:u0048_rel2']
    assert forward['canonical_type']=='restates' and forward['endpoints_reversed']
    assert forward['canonical_source']==forward['original']['target']
    assert backward['canonical_type']=='restates' and not backward['endpoints_reversed']
    assert example['canonical_type']=='example_of' and example['endpoints_reversed']
    checks.append('mixed relation directions normalize explicitly while original endpoints survive')
    modal=list(i.modal_bindings(baseline))
    assert len(modal)==sum(len(v) for v in i.MODAL_DENIALS.values())
    assert next(x for x in modal if x['target']=='story_03:u0050_failure')['operator']=='not_possible'
    assert next(x for x in modal if x['target']=='story_04:u0031_tell')['operator']=='not_able'
    checks.append('negative-modal formulas cover all explicit cases without polarity inversion')
    scopes=r.read(d.ROOT/'docs/independent_review_scope.json')
    ids={item['id'] for row in baseline for family in d.COLLECTIONS for item in row['graph'][family]}
    for case in scopes['cases']:
        assert set(case['affected_ids'])<=ids,case['case_id']
        assert set(case['evidence_refs'])<=set(case['affected_ids'])
    checks.append('all explicit scope examples point to real corpus IDs')
    scoped=list(i.scope_bindings([x for x in baseline if x['story_id']=='story_08' and x['unit_id']<='story_08_u0128']))
    qualifier=next(x for x in scoped if x['id']=='story_08:u0128_qua0')
    assert qualifier['context_paths_outer_to_inner']==[]
    assert qualifier['target_id']=='story_08:u0128_cool' and qualifier['qualifier_target_context_paths_outer_to_inner']
    checks.append('empty qualifier contexts retain the scoped target dependency')
    minimal=[{'story_id':'story_04','unit_id':'story_04_u0000','prior_source':'','current_source':'Example </script> source',
              'baseline_graph':{},'reviewed_graph':{},'independent_model_interpretation':'Example',
              'comparison':'Example','changes':[],'open_questions':[],'scope_definitions':[],'selection_reasons':['test']}]
    page=e.human_html(minimal).decode()
    assert page.count('</script>')==1
    assert page==e.human_html(json.loads(json.dumps(minimal,sort_keys=True))).decode()
    assert 'fetch(' not in page and 'XMLHttpRequest' not in page
    checks.append('human packet escapes script content and has no submission endpoint')
    print(json.dumps({'passed':len(checks),'checks':checks},indent=2))


if __name__=='__main__': main()
