"""Replay actual annotations and probe structural failures in memory only."""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d


def main():
    report=d.verify()
    checks=[]
    for entry in d.corpus_index()["stories"]:
        rs=d.records(entry["id"])
        if not rs: continue
        r=rs[0]
        raw=d.read_json(d.OUTPUT/"responses"/entry["id"]/(r["unit_id"]+".json"))["draft"]
        _,story=d.story_source(entry["id"])
        unit=story["units"][0]
        def reject(name,value):
            try:d.validate_graph(value,story,unit,[])
            except ValueError:checks.append({"story_id":entry["id"],"check":name,"passed":True})
            else:raise AssertionError("Invalid in-memory mutation accepted: "+name)
        bad=deepcopy(raw);bad["unexpected_field"]=True;reject("unknown_fields_rejected",bad)
        if raw["events"]:
            bad=deepcopy(raw)
            bad["events"][0]["arguments"].append({"role":"theme","target":entry["id"]+":unintroduced_future_node","support":"explicit","mention":None})
            reject("unavailable_references_rejected",bad)
            bad=deepcopy(raw);bad["events"][0]["trigger"]["start"]=unit["end_token"]
            reject("future_trigger_location_rejected",bad)
        populated=next((f for f in d.COLLECTIONS if raw[f]),None)
        if populated:
            bad=deepcopy(raw);bad[populated].append(deepcopy(bad[populated][0]));reject("duplicate_ids_rejected",bad)
        if raw["contexts"]:
            bad=deepcopy(raw);bad["contexts"][0]["parents"]=[bad["contexts"][0]["id"]]
            reject("context_cycles_rejected",bad)
        next_future=len(rs)+10
        if next_future<entry["units"]:
            try:d.Author(entry["id"],next_future,rs[0]["actor"])
            except ValueError:checks.append({"story_id":entry["id"],"check":"unexposed_authoring_rejected","passed":True})
            else:raise AssertionError("An unexposed future Author was accepted")
    if not checks:raise ValueError("No actual accepted annotation records available for meaningful checks")
    report.update(in_memory_mutation_checks=checks,production_records_modified=False,
                  semantic_accuracy_measured=False)
    report_path=d.ROOT/"artifacts/direct-annotation-verification.json"
    if report_path.exists():
        previous=d.read_json(report_path)
        d.immutable_json(d.ROOT/"artifacts/direct-annotation-verification"/(d.object_hash(previous)+".json"),previous)
    d.immutable_json(d.ROOT/"artifacts/direct-annotation-verification"/(d.object_hash(report)+".json"),report)
    d.save_json(report_path,report)
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=="__main__":main()
