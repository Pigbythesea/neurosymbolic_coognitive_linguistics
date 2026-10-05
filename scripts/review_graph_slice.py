"""Read only an already exposed baseline graph, in bounded slices for long units."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

sid, uid, family = sys.argv[1:4]
if not (r.folder(sid)/'graph-exposures'/(uid+'.json')).exists():
    raise ValueError('Cannot inspect an unexposed graph')
_, annotations = r.inputs(sid)
graph = r.unanchor(next(x['graph'] for x in annotations if x['unit_id']==uid))
value = graph[family]
if isinstance(value,list):
    start = int(sys.argv[4]) if len(sys.argv)>4 else 0
    stop = int(sys.argv[5]) if len(sys.argv)>5 else len(value)
    value = {'total':len(value),'start':start,'stop':stop,'records':value[start:stop]}
print(json.dumps(value,ensure_ascii=True,separators=(',',':')))
