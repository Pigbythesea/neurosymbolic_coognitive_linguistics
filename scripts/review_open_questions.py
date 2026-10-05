"""Inspect only committed review questions, never future source material."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

for sid in (f'story_{n:02}' for n in range(1,12)):
    for row in r.reviewed(sid):
        for question in row['unresolved']:
            print(json.dumps({'unit_id':row['unit_id'],**question},ensure_ascii=False))
