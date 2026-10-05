"""Check source links and IDs in author self-review; do not judge semantics."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d


def strings(value, trail=()):
    if isinstance(value,str):
        yield trail,value
    elif isinstance(value,dict):
        for key,child in value.items(): yield from strings(child,trail+(key,))
    elif isinstance(value,list):
        for index,child in enumerate(value): yield from strings(child,trail+(str(index),))


def main():
    issues, checked_files, pending = [], [], []
    link_count = node_count = unit_count = 0
    for entry in d.corpus_index()['stories']:
        sid = entry['id']; rs = d.records(sid)
        if len(rs) != entry['units']:
            pending.append(sid); continue
        unit_ids = {r['unit_id'] for r in rs}
        node_ids = {item['id'] for r in rs for family in d.COLLECTIONS for item in r['graph'][family]}
        folder = d.OUTPUT/'authoring'/sid
        paths = [p for p in (folder/'semantic_review.json',folder/'review.md',folder/'final_checkpoint.json') if p.exists()]
        if not any(p.name in {'semantic_review.json','review.md'} for p in paths):
            issues.append({'story_id':sid,'issue':'No author self-review saved'})
        for path in paths:
            checked_files.append(path.relative_to(d.ROOT).as_posix())
            text = path.read_text(encoding='utf-8')
            values = strings(json.loads(text)) if path.suffix=='.json' else [((),text)]
            for trail,value in values:
                if re.fullmatch(r'story_\d\d:[A-Za-z0-9_:-]+',value):
                    node_count += 1
                    if value not in node_ids:
                        issues.append({'file':path.name,'story_id':sid,'field':'.'.join(trail),'issue':'Unavailable record ID','value':value})
                normalized_unit = sid+'_'+value if re.fullmatch(r'u\d{4}',value) else value
                if re.fullmatch(r'story_\d\d_u\d{4}',normalized_unit):
                    unit_count += 1
                    if normalized_unit not in unit_ids:
                        issues.append({'file':path.name,'story_id':sid,'field':'.'.join(trail),'issue':'Unavailable unit ID','value':value})
                if path.name=='final_checkpoint.json' and trail and trail[-1] in {'last_graph_hash','final_graph_hash'} and value != rs[-1]['graph_hash']:
                    issues.append({'file':path.name,'story_id':sid,'issue':'Completion hash differs','value':value})
                # Repository-relative paths are used in JSON reviews. Fragment
                # pointers identify fields for inspection; they are not URLs.
                for match in re.finditer(r'data/annotations/deniz-direct-v2/[A-Za-z0-9_./-]+\.(?:json|py|md)',value):
                    if 'uNNNN' in match.group(): continue  # documented filename template
                    link_count += 1
                    target = d.ROOT/match.group()
                    if not target.is_file():
                        issues.append({'file':path.name,'story_id':sid,'issue':'Missing source-linked file','value':match.group()})
            if path.suffix=='.md':
                for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
                    if target.startswith(('http:','https:')): continue
                    link_count += 1
                    local = target.split('#',1)[0]
                    if not (path.parent/local).is_file():
                        issues.append({'file':path.name,'story_id':sid,'issue':'Missing Markdown link','value':target})
    report = {'check':'author self-review source links and IDs','checked_files':checked_files,
              'source_file_links_checked':link_count,'record_ids_checked':node_count,'unit_ids_checked':unit_count,
              'pending_story_reviews':pending,'issues':issues,'passed':not issues,
              'complete_self_review_coverage':not pending and not issues,
              'semantic_correctness_evaluated':False,'independent_human_review':'pending','accepted_records_modified':False}
    d.save_json(d.ROOT/'artifacts/direct-annotation-review-links.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if issues: raise SystemExit(1)


if __name__=='__main__': main()
