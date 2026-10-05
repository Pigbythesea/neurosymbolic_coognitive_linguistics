"""Canonical reading text, explicit word alignment, and causal annotation units."""

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import re

from .deniz import TOKEN, MARKER, normalized_tokens, read_textgrid, sha256
from .io import immutable_json, object_hash, read_json


EXPANSIONS = {
    "gonna": ["going", "to"], "wanna": ["want", "to"], "gotta": ["got", "to"],
    "kinda": ["kind", "of"], "sorta": ["sort", "of"], "outta": ["out", "of"],
    "i'm": ["i", "am"], "you're": ["you", "are"], "we're": ["we", "are"],
    "they're": ["they", "are"], "i've": ["i", "have"], "you've": ["you", "have"],
    "we've": ["we", "have"], "they've": ["they", "have"], "can't": ["can", "not"],
    "cannot": ["can", "not"], "won't": ["will", "not"], "don't": ["do", "not"],
    "doesn't": ["does", "not"], "didn't": ["did", "not"], "isn't": ["is", "not"],
    "wasn't": ["was", "not"], "weren't": ["were", "not"], "aren't": ["are", "not"],
    "couldn't": ["could", "not"], "wouldn't": ["would", "not"], "shouldn't": ["should", "not"],
}
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"}
EVENT_LABEL = re.compile(r"^(?:[\{\[]?\s*(?:BR|LG|LS|NS|CG)\s*[\}\]]?|sp|sil|SENTENCE[\s_]*(?:START|END))$", re.I)


def transcript_tokens(text: str) -> list[dict]:
    # Mask source event tags without shifting source character offsets.
    masked = MARKER.sub(lambda match: " " * len(match.group()), text)
    return [{"index": i, "text": match.group(), "char_start": match.start(), "char_end": match.end()}
            for i, match in enumerate(TOKEN.finditer(masked))]


def expanded(items: list[dict], field: str) -> tuple[list[str], list[int]]:
    sequence, owners = [], []
    for i, item in enumerate(items):
        for token in normalized_tokens(item[field]):
            parts = EXPANSIONS.get(token, [token])
            sequence.extend(parts)
            owners.extend([i] * len(parts))
    return sequence, owners


def align_words(words: list[dict], entries: list[tuple]) -> tuple[list[dict], list[dict]]:
    timed = [{"label": label, "start": start, "end": end, "interval_index": index}
             for index, (start, end, label) in enumerate(entries)
             if not EVENT_LABEL.fullmatch(label.strip()) and normalized_tokens(label)]
    source, source_owner = expanded(words, "text")
    target, target_owner = expanded(timed, "label")
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    matches = {}
    matched_parts = Counter()
    total_parts = Counter(source_owner)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            word = source_owner[block.a + offset]
            matches.setdefault(word, set()).add(target_owner[block.b + offset])
            matched_parts[word] += 1
    aligned = []
    for i, word in enumerate(words):
        chosen = sorted(matches.get(i, []))
        complete = bool(chosen) and matched_parts[i] == total_parts[i]
        exact = (complete and len(chosen) == 1 and
                 normalized_tokens(word["text"]) == normalized_tokens(timed[chosen[0]]["label"]))
        aligned.append({**word, "alignment": "exact" if exact else "normalized" if complete else "unresolved",
                        "onset_seconds": min(timed[j]["start"] for j in chosen) if complete else None,
                        "offset_seconds": max(timed[j]["end"] for j in chosen) if complete else None,
                        "textgrid_interval_indices": [timed[j]["interval_index"] for j in chosen] if complete else []})
    differences = [{"operation": operation, "source_expanded_span": [a, b], "timing_expanded_span": [c, d],
                    "source_tokens": source[a:b], "timing_tokens": target[c:d]}
                   for operation, a, b, c, d in matcher.get_opcodes() if operation != "equal"]
    # Missing transcript words do not receive invented interpolated timestamps.
    previous = None
    for word in aligned:
        word["previous_anchor_seconds"] = previous
        if word["offset_seconds"] is not None:
            previous = word["offset_seconds"]
    following = None
    for word in reversed(aligned):
        word["following_anchor_seconds"] = following
        if word["onset_seconds"] is not None:
            following = word["onset_seconds"]
    return aligned, differences


def annotation_units(story: str, text: str, words: list[dict]) -> list[dict]:
    """Punctuation-delimited prefixes; boundaries are operational, not gold parses."""
    units = []
    start = 0
    for i, word in enumerate(words):
        next_start = words[i+1]["char_start"] if i+1 < len(words) else len(text)
        gap = MARKER.sub("", text[word["char_end"]:next_start])
        period = "." in gap and word["text"].casefold() not in ABBREVIATIONS
        if len(word["text"]) == 1 and word["text"].isupper() and "." in gap and i+1 < len(words):
            period = False  # Initials, including speaker names and acronyms.
        if word["text"].isdigit() and i+1 < len(words) and words[i+1]["text"].isdigit() and gap == ".":
            period = False
        boundary = i+1 == len(words) or period or "?" in gap or "!" in gap
        if not boundary:
            continue
        span = words[start:i+1]
        unit = {"id": f"{story}_u{len(units):04d}", "story_id": story,
                "start_token": start, "end_token": i+1,
                "char_start": words[start]["char_start"], "prefix_char_end": next_start,
                "onset_seconds": span[0]["onset_seconds"], "offset_seconds": span[-1]["offset_seconds"],
                "timing_counts": dict(Counter(item["alignment"] for item in span)),
                "endpoint_timing_resolved": span[-1]["offset_seconds"] is not None}
        units.append(unit)
        start = i+1
    if not units or units[-1]["end_token"] != len(words):
        raise ValueError("Annotation units do not cover the complete transcript.")
    return units


def build_corpus(raw_root: Path, contract_path: Path, output: Path) -> dict:
    contract = read_json(contract_path)
    source_hashes = {item["path"]: item["sha256"] for item in contract["source_files"]}
    index = {"format_version": 1, "dataset_manifest_sha256": contract["manifest_sha256"],
             "preparation_code_sha256": sha256(Path(__file__)),
             "transcript_policy": "Released reading transcript is the semantic text; TextGrid supplies timing, not replacement wording.",
             "annotation_availability": "Every graph delta becomes available only at the end of its supplied prefix.",
             "unit_policy": "Punctuation-delimited prefixes; no forced length cut; no future text in annotation input.",
             "timing_policy": "Exact/normalized matches keep observed interval unions; unresolved words have null times and adjacent anchors.",
             "stories": []}
    for story_id, info in sorted(contract["stories"].items()):
        paths = {"transcript": info["transcript"],
                 "textgrid": f"stimuli/textgrids/stimulus_textgrids/{info['textgrid']}.TextGrid"}
        for relative in paths.values():
            if sha256(raw_root / relative) != source_hashes[relative]:
                raise ValueError("Local stimulus differs from the inspected cluster source: " + relative)
        text = (raw_root / paths["transcript"]).read_text(encoding="utf-8-sig")
        grid = read_textgrid(raw_root / paths["textgrid"])
        words, differences = align_words(transcript_tokens(text), grid["word_entries"])
        units = annotation_units(story_id, text, words)
        record = {"format_version": 1, "story_id": story_id, "text": text, "words": words, "units": units,
                  "source_paths": paths, "source_sha256": {key: source_hashes[value] for key, value in paths.items()},
                  "timing_differences": differences}
        record["content_hash"] = object_hash(record)
        immutable_json(output / "stories" / f"{story_id}.json", record)
        index["stories"].append({"id": story_id, "path": f"stories/{story_id}.json", "content_hash": record["content_hash"],
                                 "split": "heldout" if story_id == "story_11" else "development",
                                 "words": len(words), "units": len(units),
                                 "timing_counts": dict(Counter(word["alignment"] for word in words)),
                                 "units_with_unresolved_endpoint": sum(not item["endpoint_timing_resolved"] for item in units)})
    index["content_hash"] = object_hash(index)
    immutable_json(output / "index.json", index)
    print(f"CORPUS READY: {len(index['stories'])} stories; {sum(s['words'] for s in index['stories'])} words; "
          f"{sum(s['units'] for s in index['stories'])} causal annotation units.")
    for story in index["stories"]:
        print(f"  {story['id']}: {story['units']} units; timing={story['timing_counts']}; "
              f"unresolved unit endpoints={story['units_with_unresolved_endpoint']}")
    print("Corpus:", output / "index.json")
    return index
