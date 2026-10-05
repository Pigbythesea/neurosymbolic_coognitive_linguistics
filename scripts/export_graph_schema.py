"""Export the actual graph model as the Codex structured-output schema."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.graphs import graph_schema
from neurosym.annotation_repair import repair_schema
from neurosym.io import save_json

if __name__ == "__main__":
    path = ROOT / "schemas/graph-delta.schema.json"
    save_json(path, graph_schema())
    print("GRAPH SCHEMA:", path)
    repair_path = ROOT / "schemas/graph-repair.schema.json"
    save_json(repair_path, repair_schema())
    print("REPAIR SCHEMA:", repair_path)
