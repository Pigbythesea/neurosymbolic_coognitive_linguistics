"""Check updated query interfaces using archived targets and released text features.

This is numerical interface verification, not a scientific neural experiment.
No artificial observations, new model annotations, or brain predictions are saved.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from neurosym.annotation import verified_story
from neurosym.dataset import DenizReader
from neurosym.decoders import SemanticDecoder, acceptable_loss, query_vocabulary
from neurosym.graphs import GraphDelta, GraphHistory
from neurosym.io import read_json, save_json
from neurosym.semantic_records import compile_unit
from neurosym.semantic_queries import compile_queries
from neurosym.temporal import checked_alignment


def main():
    torch.set_num_threads(1)
    index = read_json(ROOT / "data/processed/corpus/index.json")
    reader = DenizReader(ROOT / "data/raw/deniz", ROOT / "data/processed/deniz/contract.json")
    selected = {}
    for entry in index["stories"]:
        story = verified_story(ROOT / "data/processed/corpus", entry)
        history = GraphHistory()
        alignment = checked_alignment(ROOT / "data/processed/alignment", entry["id"], index["content_hash"])
        values = reader.features(entry["id"], ["english1000"], trim=False)["english1000"]
        for i, unit in enumerate(story["units"][:4]):
            path = ROOT / "data/annotations/deniz-luna-v1/primary" / entry["id"] / (unit["id"] + ".json")
            if not path.exists():
                break
            saved = read_json(path)
            graph = GraphDelta.model_validate(saved["graph"])
            history.append(graph, unit)
            source, *_ = compile_unit(story, unit, graph, history, saved, i, alignment)
            queries, answers, _, _ = compile_queries(story, unit, graph, history, source, 1729)
            slot = source["raw_feature_bin"]
            if slot < 4:
                continue
            # Released text features exercise tensor operations; these are never
            # called fMRI or frozen-model states and produce no scientific score.
            observed = np.array(values[slot - 4:slot, :24], dtype=np.float32).reshape(4, 8, 3).transpose(1, 0, 2)
            for q, a in zip(queries, answers, strict=True):
                if a["acceptable_indices"]:
                    selected.setdefault(q["input"]["ast"]["op"], {"inputs": q["input"],
                        "acceptable_indices": a["acceptable_indices"], "observed": observed, "source_id": unit["id"]})
    expected = {"concept", "role", "status", "polarity", "binding", "reference", "compose", "relation"}
    if not expected <= set(selected):
        raise ValueError("Archived query coverage is insufficient: " + str(expected - set(selected)))
    vocab = query_vocabulary(list(selected.values()))
    checked = []
    for family in ("prior", "linear", "mlp", "structured"):
        model = SemanticDecoder(family, (8, 4, 3), vocab, hidden=16)
        for op, example in selected.items():
            model.zero_grad(set_to_none=True)
            logits, trace = model(torch.as_tensor(example["observed"]), example["inputs"], capture=True)
            loss = acceptable_loss(logits, example["acceptable_indices"])
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            checked.append({"family": family, "op": op, "source_id": example["source_id"]})
    report = {"status": "passed", "checks": checked, "model_calls": 0, "scientific_fits": 0,
        "scope": "Updated query encoding/forward/backward on archived annotations and released textual features. Identity/literal query coverage awaits real v3 outputs."}
    save_json(ROOT / "artifacts/joint-decoder-verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
