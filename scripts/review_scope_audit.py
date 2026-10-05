"""Diagnostic redundancy candidates; never changes semantic annotations."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

rows=r.rows(r.baseline()/'author-corrected/annotations.jsonl')
events={x['id']:x for row in rows for x in row['graph']['events']}
for row in rows:
    for q in row['graph']['qualifiers']:
        event=events.get(q['target'])
        if event and event['polarity']=='negative' and q['dimension'] in {'modality','modal','logical_scope','polarity_confirmation','quantification','frequency'}:
            print(json.dumps({'unit_id':row['unit_id'],'qualifier':q['id'],'dimension':q['dimension'],'value':q['value'],
                              'event':event['id'],'predicate':event['predicate'],'quote':event['evidence']['quote'],
                              'qualifier_quote':q['evidence']['quote']},ensure_ascii=False))
