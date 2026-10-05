"""Screen real annotations for pronoun/span mismatches; never correct labels.

This lexical screen is a review queue, not a semantic accuracy measurement.
Dialect, quotation, and metalinguistic uses can produce false positives.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import verified_story, utcnow
from neurosym.io import read_json, save_json


PRONOUN_FORMS = set("""i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs themselves
who whom whose whoever whomever what whatever which whichever that this these those
one oneself somebody someone something anybody anyone anything nobody noone nothing
everybody everyone everything each other another there""".split())


def suspicious_mentions(mentions, words):
    results = []
    for mention in mentions:
        start, end = mention["span"]["start"], mention["span"]["end"]
        if mention["form"] != "pronoun" or end - start != 1 or not 0 <= start < len(words):
            continue
        text = words[start]["text"]
        normalized = text.lower().strip('.,!?:;"')
        # Contractions need a separate linguistic check; this screen doesn't
        # guess a tokenizer-dependent expansion or reject them automatically.
        if normalized in PRONOUN_FORMS or "'" in text or "\u2019" in text:
            continue
        results.append({"mention_id": mention.get("id"), "entity_id": mention.get("entity_id"),
                        "span": mention["span"], "actual_text": text,
                        "form": "pronoun", "confidence": mention["confidence"]})
    return results


def audit(root, config_path):
    config = read_json(config_path)
    output, corpus = root / config["output"], root / config["corpus"]
    index = read_json(corpus / "index.json")
    findings, scanned, totals = [], 0, Counter()
    for entry in index["stories"]:
        story = verified_story(corpus, entry)
        for pass_name in config["passes"]:
            for unit in story["units"]:
                path = output / pass_name / entry["id"] / (unit["id"] + ".json")
                if not path.is_file():
                    continue
                saved = read_json(path)
                scanned += 1
                suspects = suspicious_mentions(saved["graph"]["mentions"], story["words"])
                if not suspects:
                    continue
                totals[entry["id"]] += len(suspects)
                findings.append({"pass": pass_name, "unit_id": unit["id"], "story_id": entry["id"],
                    "annotation_file": path.relative_to(root).as_posix(),
                    "graph_hash": saved["graph_hash"],
                    "current_text": story["text"][unit["char_start"]:unit["prefix_char_end"]],
                    "suspected_mentions": suspects, "review_status": "pending"})
    report = {"recorded_utc": utcnow(), "annotation_output": config["output"],
              "saved_units_scanned": scanned, "flagged_units": len(findings),
              "flagged_mentions": sum(totals.values()), "flagged_mentions_by_story": dict(totals),
              "interpretation": "Lexical screening flags require review. Counts are not error rates or model accuracy. "
                  "This screen cannot detect all semantic errors, omissions, or incorrect coreference. "
                  "No existing labels, graph hashes, or histories were modified.",
              "findings": findings}
    destination = root / "artifacts" / (output.name + "-span-audit.json")
    save_json(destination, report)
    print(f"Scanned {scanned} saved units: {len(findings)} units / {sum(totals.values())} mentions flagged for review.")
    print(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/annotation.json")
    args = parser.parse_args()
    audit(ROOT, args.config)
