"""Inspect concrete instances for semantic-interface decisions, after story review."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

query=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows=r.rows(r.baseline()/'author-corrected/annotations.jsonl')
for category,spec in query.items():
    field=spec['field']; labels=spec['labels']; matches={x:[] for x in labels}
    for row in rows:
        for item in row['graph'][category]:
            if item.get(field) in matches:
                matches[item[field]].append(item)
    for label,items in matches.items():
        if len(sys.argv)>2 and sys.argv[2]=='compact':
            items=[{k:v for k,v in x.items() if k in ['id','predicate','sense','arguments','type','source','target']} | {'quote':x['evidence']['quote']} for x in items]
        print(json.dumps({'label':label,'count':len(items),'examples':items[:spec.get('limit',2)]},ensure_ascii=False))
