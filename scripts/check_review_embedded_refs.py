"""Audit exact story-ID strings even in open qualifier/uncertainty values."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r
from neurosym import direct_annotations as d
from neurosym.independent_review_export import embedded_references as references


if __name__=='__main__':
    known={}; errors=[]; count=0
    for row in r.rows(r.baseline()/'author-corrected/annotations.jsonl'):
        sid=row['story_id']; ids=known.setdefault(sid,set())
        ids.update(x['id'] for f in d.COLLECTIONS for x in row['graph'][f])
        for ident,path in references(row['graph']):
            count+=1
            if ident not in ids: errors.append({'unit_id':row['unit_id'],'id':ident,'path':path})
    print(json.dumps({'references_checked':count,'unavailable':errors},ensure_ascii=False,indent=2))
