"""Compact view of already committed independent corrections."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

for decision in r.reviewed(sys.argv[1]):
    if decision['operations']:
        print(json.dumps({'unit_id':decision['unit_id'],'assessment':decision['assessment'],
                          'operations':decision['operations']},ensure_ascii=False))
