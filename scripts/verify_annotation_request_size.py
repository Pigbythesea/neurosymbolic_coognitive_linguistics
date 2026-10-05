"""Check shared catalogs against the actual request rejected by Codex.

The real failed grounding is used only to measure transport and candidate
preservation. Its semantic labels are not endorsed or saved as annotations.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import make_prompt, utcnow, verified_story
from neurosym.annotation_catalog import Catalog, bind_prompt, binding_schema, BINDING_TRANSPORT
from neurosym.annotation_catalog_jobs import CatalogStoryJob
from neurosym.annotation_grounding_checks import validate_grounding_spans
from neurosym.graphs import GraphDelta, GraphHistory
from neurosym.io import read_json, save_json
import json


config = read_json(ROOT / "configs/annotation-legacy-luna-v1.json")
output, corpus = ROOT / config["output"], ROOT / config["corpus"]
unit_id, story_id = "story_11_u0072", "story_11"
directory = output / "attempts/primary" / unit_id / "catalog"
old_prompt = (directory / "request-0001.prompt.txt").read_text(encoding="utf-8")
old_payload = json.loads(old_prompt.split("\nBINDING CATALOG:\n", 1)[1])
entry = next(e for e in read_json(corpus / "index.json")["stories"] if e["id"] == story_id)
story = verified_story(corpus, entry)
history = GraphHistory()
for unit in story["units"]:
    if unit["id"] == unit_id:
        break
    history.append(GraphDelta.model_validate(read_json(output / "primary" / story_id / (unit["id"] + ".json"))["graph"]), unit)
else:
    raise ValueError("The actual failed unit was not found.")
draft = read_json(directory / "request-0000.response.json")
catalog = Catalog(draft, history, unit)
try:
    validate_grounding_spans(catalog, story)
except ValueError as error:
    assert "actually covers 'get'" in str(error)
    assert "actually covers 'to'" in str(error)
else:
    raise AssertionError("The actual defective grounding was not rejected before binding.")
payload = catalog.payload()
assert old_payload["compiled_nodes"] == payload["compiled_nodes"]
assert old_payload["argument_targets"] == payload["argument_targets"]
assert old_payload["update_entities"] == payload["update_entities"]
assert [v["event_id"] for v in old_payload["relation_events"]] == payload["relation_events"]
for mention, old in old_payload["reference_anchors"].items():
    assert [v["mention_id"] for v in old] == [v for v in catalog.anchors[mention] if v != mention]
for event, old in old_payload["scope_parents"].items():
    assert [v["event_id"] for v in old] == [v for v in catalog.parents[event] if v != event]
base = make_prompt((ROOT / config["prompt"]).read_text(encoding="utf-8"), story, unit, history)
new_prompt = bind_prompt(base, catalog)
assert len(old_prompt) == 1_139_370
assert len(new_prompt) < 1_048_576


def no_request(*args, **kwargs):
    raise AssertionError("No model request may be made by this verification.")


job = CatalogStoryJob(output=output, corpus=corpus, config=config, index={}, identity={},
                     implementation_hash="offline-size-verification", instructions="", codex=None,
                     cli=None, workspace=None, graph_schema_path=None, repair_schema_path=None,
                     policy={}, progress=no_request)
try:
    job.request(directory, old_prompt, "offline-size-verification", None)
except RuntimeError as error:
    assert str(error).startswith("Local request-size check:")
else:
    raise AssertionError("The oversized real request was not rejected locally.")

report = {"recorded_utc": utcnow(), "unit": unit_id, "binding_transport": BINDING_TRANSPORT,
          "old_prompt_characters": len(old_prompt), "new_prompt_characters": len(new_prompt),
          "reduction_percent": round(100 * (1 - len(new_prompt) / len(old_prompt)), 1),
          "new_schema_characters": len(json.dumps(binding_schema(catalog), ensure_ascii=False)),
          "all_candidate_sets_preserved": True, "indexed_story_prefix_preserved": True,
          "local_oversize_guard_verified": True, "model_calls": 0,
          "actual_bad_grounding_rejected_before_binding": True,
          "semantic_status": "Failed draft is known to contain incorrect pronoun spans; used only for transport verification."}
save_json(ROOT / "artifacts/annotation-request-size-validation.json", report)
print(json.dumps(report, indent=2))
