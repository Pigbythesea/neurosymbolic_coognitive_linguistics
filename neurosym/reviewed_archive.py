"""Read-only, pinned access to the accepted review export. No review is reopened."""
from collections import defaultdict
from pathlib import Path

from .extraction_inputs import checked_index, checked_story
from .io import object_hash, read_json
from .semantics import digest_file, read_jsonl

RECORD_TYPES = ("entities", "events", "mentions", "literals", "contexts", "relations",
                "properties", "qualifiers", "identity_links", "uncertainties")
MODULES = ("reviewed_archive.py", "reviewed_graph.py", "reviewed_compile.py", "reviewed_queries.py")


class ReviewedArchive:
    def __init__(self, root, config, *, verify=True):
        self.root = Path(root).resolve()
        self.path = self.root / config["reviewed_export"]
        self.snapshot = read_json(self.path / "snapshot.json")
        self.manifest = read_json(self.path / "manifest.json")
        self.checksums = read_json(self.path / "checksums.json")
        self.build_hash = self.snapshot["build_hash"]
        if (self.build_hash != config["reviewed_build_hash"] or
                digest_file(self.path / "checksums.json") != self.snapshot["checksums_sha256"] or
                object_hash({"protocol": self.manifest["protocol"], "files": self.checksums}) != self.build_hash):
            raise ValueError("Accepted review identity mismatch; never follow a newer latest pointer.")
        if verify:
            self.verify_files()
        audit = read_json(self.path / "verification.json")
        if not audit["complete"] or audit["reviewed_units"] != 1217:
            raise ValueError("The accepted full-corpus review is required.")
        self.index = checked_index(self.root / config["corpus"])
        self.records = list(read_jsonl(self.path / "annotations.jsonl"))
        self.normalized = {r["id"]: r for r in read_jsonl(self.path / "interface/normalized-records.jsonl")}
        self.scope_bindings = {r["id"]: r for r in read_jsonl(self.path / "interface/scope-bindings.jsonl")}
        self.definitions = read_json(self.path / "interface/definitions.json")
        self.cases = read_json(self.path / "interface/scope-cases.json")["cases"]
        self.cases += list(read_jsonl(self.path / "interface/modal-scope-bindings.jsonl"))
        self.questions = list(read_jsonl(self.path / "unresolved.jsonl"))
        bindings = {b["issue_id"]: b["affected_ids"] for b in read_json(self.path / "question-bindings.json")["bindings"]}
        for q in self.questions:
            q["affected_ids"] = q.get("affected_ids") or bindings.get(q["issue_id"], [])
            if not q["affected_ids"]:
                raise ValueError("Unbound accepted review question: " + q["issue_id"])
        self.by_story = defaultdict(list)
        source = {r["unit_id"]: r for r in read_jsonl(self.path / "source-only.jsonl")}
        self.stories, self.coverage = {}, []
        by_unit = {r["unit_id"]: r for r in self.records}
        if len(by_unit) != len(self.records):
            raise ValueError("Duplicate reviewed units.")
        self.normalized_count = 0
        for entry in self.index["stories"]:
            story = checked_story(self.root / config["corpus"], entry)
            self.stories[entry["id"]] = story
            parent = None
            for unit in story["units"]:
                r, s = by_unit[unit["id"]], source[unit["id"]]
                if (r["story_hash"] != entry["content_hash"] or r["graph_hash"] != object_hash(r["graph"]) or
                        r["compiled_parent_graph_hash"] != parent or s["unit"] != unit or
                        s["words"] != story["words"][unit["start_token"]:unit["end_token"]] or
                        s["text"] != story["text"][unit["char_start"]:unit["prefix_char_end"]] or
                        r["available_at_token"] != unit["end_token"] or
                        r["available_at_seconds"] != unit["offset_seconds"]):
                    raise ValueError("Reviewed corpus/provenance/timing differs: " + unit["id"])
                parent = r["graph_hash"]
                for kind in RECORD_TYPES:
                    for record in r["graph"][kind]:
                        n = self.normalized[record["id"]]
                        if (n["original"] != record or n["record_type"] != kind or
                                n["unit_id"] != unit["id"] or n["source_graph_hash"] != parent or
                                n["available_at_token"] != unit["end_token"] or
                                n["available_at_seconds"] != unit["offset_seconds"]):
                            raise ValueError("Lossless normalization sidecar mismatch: " + record["id"])
                        self.normalized_count += 1
                self.by_story[entry["id"]].append(r)
            self.coverage.append({"story_id": entry["id"], "split": entry["split"],
                                  "total_units": len(story["units"]), "compiled_units": len(story["units"]),
                                  "coverage_complete": True, "missing_units": [], "blocked_after_gap": []})
        if len(self.records) != sum(c["total_units"] for c in self.coverage) or self.normalized_count != len(self.normalized):
            raise ValueError("Extra/missing accepted observations.")

    def verify_files(self):
        for name, expected in self.checksums.items():
            path = self.path / name
            if not path.resolve().is_relative_to(self.path.resolve()) or path.is_symlink() or digest_file(path) != expected:
                raise ValueError("Accepted archive changed: " + name)
        return len(self.checksums)

    def constraints(self, story_id):
        return {"cases": [c for c in self.cases if c["unit_id"].startswith(story_id + "_")],
                "questions": [q for q in self.questions if q["story_id"] == story_id],
                "scope_bindings": {k: v for k, v in self.scope_bindings.items() if k.startswith(story_id + ":")},
                "definitions": self.definitions}
