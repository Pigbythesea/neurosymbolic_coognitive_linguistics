"""Read-only vocabulary inventory, never a semantic labeler."""
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

inventory = defaultdict(lambda: defaultdict(list))
for row in r.rows(r.baseline()/'author-corrected/annotations.jsonl'):
    for family, field in [('entities','concept'),('events','predicate'),('contexts','kind'),('relations','type'),('properties','attribute'),('qualifiers','dimension'),('mentions','form')]:
        for x in row['graph'][family]:
            inventory[family][x[field]].append(x['id'])
            if family == 'events':
                for a in x['arguments']:
                    inventory['roles'][a['role']].append(x['id'])
                    inventory['predicate_roles'][x['predicate']+' / '+a['role']].append(x['id'])
result = {f:{k:{'count':len(v),'examples':v[:3]} for k,v in sorted(values.items())} for f,values in inventory.items()}
if len(sys.argv)>1:
    category=sys.argv[1]
    if len(sys.argv)>2 and sys.argv[2]=='keys':
        print(' | '.join(result[category]))
    else:
        print(json.dumps(result[category],ensure_ascii=False,indent=2))
else:
    print(json.dumps({k:len(v) for k,v in result.items()}))
