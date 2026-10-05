"""Read-only full replay of the pinned original export, without printing content."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r
result=r.d.verify_export(r.baseline())
r.save(r.OUT/'baseline-verification.json',result)
print(json.dumps(result,indent=2))
