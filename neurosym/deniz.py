"""Read and inspect the actual Deniz release without changing its arrays or timing.

The inspection establishes a data contract for the downstream research pipeline.
It does not fit models, choose voxels using held-out responses, or infer concepts.
"""

from __future__ import annotations

import datetime as dt
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import unicodedata
import zipfile

import h5py
import numpy as np
from praatio.utilities import textgrid_io


TOKEN = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
MARKER = re.compile(r"\{[^{}]*\}")
SILENCE = {"", "sp", "sil", "sentence_start", "sentence_end"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def normalized_tokens(text: str) -> list[str]:
    text = MARKER.sub(" ", unicodedata.normalize("NFKC", text))
    return [match.group().casefold().replace("\u2019", "'") for match in TOKEN.finditer(text)]


def read_textgrid(path: Path) -> dict:
    # The higher-level IntervalTier constructor rejects 1e-7-second overlaps
    # present in the released phone/word tiers. Parse the original entries and
    # report overlaps explicitly; never round, repair, or delete source times.
    grid = textgrid_io.parseTextgridStr(path.read_text(encoding="utf-8-sig"),
                                      includeEmptyIntervals=True)
    tiers = []
    word_entries = None
    for raw in grid["tiers"]:
        if raw["class"] != "IntervalTier":
            raise ValueError(f"Unexpected tier type in {path.name}: {raw['class']}")
        entries = [(float(start), float(end), str(label)) for start, end, label in raw["entries"]]
        if any(not math.isfinite(t) for entry in entries for t in entry[:2]):
            raise ValueError(f"Nonfinite TextGrid time: {path.name}/{raw['name']}")
        if any(end < start for start, end, _ in entries):
            raise ValueError(f"Negative interval duration: {path.name}/{raw['name']}")
        if any(entries[i][0] < entries[i-1][0] for i in range(1, len(entries))):
            raise ValueError(f"Unordered TextGrid intervals: {path.name}/{raw['name']}")
        overlaps = [entries[i-1][1] - entries[i][0] for i in range(1, len(entries))
                    if entries[i][0] < entries[i-1][1]]
        tiers.append({"name": raw["name"], "intervals": len(entries),
                      "zero_duration_intervals": sum(a == b for a, b, _ in entries),
                      "overlaps": len(overlaps), "max_overlap_seconds": max(overlaps, default=0.0),
                      "overlaps_above_one_microsecond": sum(value > 1e-6 for value in overlaps)})
        if raw["name"].casefold() in ("word", "words"):
            if word_entries is not None:
                raise ValueError(f"Ambiguous word tier: {path.name}")
            word_entries = entries
    if word_entries is None:
        raise ValueError(f"No word tier: {path.name}")
    lexical = [(start, end, label) for start, end, label in word_entries
               if label.strip().casefold() not in SILENCE and not MARKER.fullmatch(label.strip())]
    return {"path": path.name, "xmin": float(grid["xmin"]), "xmax": float(grid["xmax"]),
            "tiers": tiers, "lexical_intervals": len(lexical),
            "sentence_starts": sum(label.casefold() == "sentence_start" for _, _, label in word_entries),
            "sentence_ends": sum(label.casefold() == "sentence_end" for _, _, label in word_entries),
            "tokens": [token for _, _, label in lexical for token in normalized_tokens(label)],
            "word_entries": word_entries}


def inspect_stimuli(root: Path, issues: list[dict]) -> dict:
    grids = {path.stem: read_textgrid(path) for path in sorted(root.glob("stimuli/**/*.TextGrid"))}
    stories = {}
    selected = []
    for path in sorted(root.glob("stimuli/story_*.txt")):
        tokens = normalized_tokens(path.read_text(encoding="utf-8-sig"))
        candidates = []
        for name, grid in grids.items():
            matcher = SequenceMatcher(None, tokens, grid["tokens"], autojunk=False)
            candidates.append((matcher.ratio(), name, matcher))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not tokens or len(candidates) < 2:
            raise ValueError("Missing transcript or timing files.")
        score, name, matcher = candidates[0]
        grid = grids[name]
        matched = sum(block.size for block in matcher.get_matching_blocks())
        mismatches = []
        for operation, a, b, c, d in matcher.get_opcodes():
            if operation != "equal":
                mismatches.append({"operation": operation, "transcript_token_span": [a, b],
                                   "timing_token_span": [c, d],
                                   "transcript": tokens[a:b], "timing": grid["tokens"][c:d]})
        confident_identity = score >= 0.9 and score - candidates[1][0] >= 0.2
        # This threshold verifies story identity only. Even a perfect identity
        # match does not establish every token's temporal or character alignment.
        if not confident_identity:
            issues.append({"severity": "error", "location": path.name,
                           "detail": "Transcript-to-TextGrid story identity is ambiguous."})
        if mismatches:
            issues.append({"severity": "warning", "location": path.name,
                           "detail": f"{len(mismatches)} transcript/timing token differences; retain explicit alignment evidence."})
        selected.append(name)
        stories[path.stem] = {"transcript": path.relative_to(root).as_posix(), "textgrid": name,
                             "identity_confirmed_by_text": confident_identity,
                             "sequence_similarity": score,
                             "runner_up": {"textgrid": candidates[1][1], "sequence_similarity": candidates[1][0]},
                             "transcript_tokens": len(tokens), "timing_tokens": len(grid["tokens"]),
                             "exact_matched_tokens": matched, "token_differences": mismatches,
                             "stimulus_end_seconds": grid["xmax"]}
    if len(stories) != 11 or len(grids) != 11 or len(set(selected)) != 11:
        issues.append({"severity": "error", "location": "stimuli",
                       "detail": "Expected 11 distinct transcripts matched one-to-one to 11 TextGrids."})
    for name, grid in grids.items():
        for tier in grid["tiers"]:
            if tier["overlaps_above_one_microsecond"]:
                issues.append({"severity": "error", "location": name + "/" + tier["name"],
                               "detail": "Timing intervals overlap by more than one microsecond."})
    return {"stories": stories,
            "textgrids": {name: {key: value for key, value in grid.items()
                                   if key not in ("tokens", "word_entries")} for name, grid in grids.items()},
            "normalization_for_identity_only": "NFKC, casefold, apostrophes, punctuation tokenization, remove brace event markers"}


def block_slices(shape: tuple[int, ...], itemsize: int, block_bytes: int):
    if not shape:
        yield ()
        return
    # Preserve complete trailing dimensions where possible, so contiguous HDF5
    # response arrays are read in time blocks, including repeated validation runs.
    remaining = max(1, block_bytes // max(1, itemsize))
    sizes = [1] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        sizes[axis] = max(1, min(shape[axis], remaining))
        remaining = max(1, remaining // sizes[axis])
    for starts in itertools.product(*(range(0, n, step) for n, step in zip(shape, sizes))):
        yield tuple(slice(start, min(start + step, n)) for start, step, n in zip(starts, sizes, shape))


def numeric_summary(values, block_bytes: int) -> dict:
    if values.dtype.kind not in "biuf":
        return {"scanned": False, "reason": "non-real-numeric dtype"}
    finite_count = nan_count = positive_inf = negative_inf = zero_count = 0
    minimum = maximum = None
    for selection in block_slices(values.shape, values.dtype.itemsize, block_bytes):
        block = np.asarray(values[selection])
        finite = np.isfinite(block)
        finite_count += int(finite.sum())
        nan_count += int(np.isnan(block).sum())
        positive_inf += int(np.isposinf(block).sum())
        negative_inf += int(np.isneginf(block).sum())
        zero_count += int(np.count_nonzero(block == 0))
        if finite.any():
            usable = block[finite]
            lower, upper = float(usable.min()), float(usable.max())
            minimum = lower if minimum is None else min(minimum, lower)
            maximum = upper if maximum is None else max(maximum, upper)
    return {"scanned": True, "elements": int(math.prod(values.shape)), "finite": finite_count,
            "nan": nan_count, "positive_inf": positive_inf, "negative_inf": negative_inf,
            "zero": zero_count, "min": minimum, "max": maximum}


def inspect_hdf(path: Path, block_bytes: int) -> dict:
    datasets = {}
    with h5py.File(path, "r") as stream:
        attributes = {key: json_value(value) for key, value in stream.attrs.items()}

        def visit(name, value):
            if not isinstance(value, h5py.Dataset):
                return
            row = {"shape": list(value.shape), "dtype": str(value.dtype),
                   "chunks": json_value(value.chunks), "compression": value.compression,
                   "attributes": {key: json_value(item) for key, item in value.attrs.items()},
                   "numeric": numeric_summary(value, block_bytes)}
            if value.size <= 32:
                row["values"] = json_value(value[()])
            datasets[name] = row

        stream.visititems(visit)
    return {"attributes": attributes, "datasets": datasets}


def inspect_mapper(path: Path, metadata: dict, issues: list[dict]) -> dict:
    matrices = {}
    rois = {}
    with h5py.File(path, "r") as stream:
        for name in stream:
            if name.endswith("_shape") and name[:-6] + "_indptr" in stream:
                prefix = name[:-6]
                shape = tuple(int(item) for item in stream[name][()])
                indices = stream[prefix + "_indices"][()]
                indptr = stream[prefix + "_indptr"][()]
                n_values = stream[prefix + "_data"].size
                valid = (len(shape) == 2 and indptr.shape == (shape[0] + 1,) and
                         indices.shape == (n_values,) and indptr[0] == 0 and indptr[-1] == n_values and
                         bool(np.all(indptr[1:] >= indptr[:-1])) and
                         bool(np.all((indices >= 0) & (indices < shape[1]))))
                matrices[prefix] = {"shape": list(shape), "nonzero_entries": n_values, "valid_csr": bool(valid)}
                if not valid:
                    issues.append({"severity": "error", "location": path.name + "/" + prefix,
                                   "detail": "Invalid released CSR mapping arrays."})
            if name.startswith("roi_mask_"):
                values = stream[name][()]
                rois[name[len("roi_mask_"):]] = {"shape": list(values.shape),
                                                 "positive_entries": int(np.count_nonzero(values > 0))}
    dimensions = sorted({item["shape"][1] for item in matrices.values() if len(item["shape"]) == 2})
    if len(dimensions) != 1:
        issues.append({"severity": "error", "location": path.name,
                       "detail": "Mapper voxel dimensions are missing or inconsistent."})
    return {"matrices": matrices, "voxel_dimensions": dimensions, "roi_masks": rois,
            "has_flatmap_mask": "flatmap_mask" in metadata["datasets"]}


def check_sources(manifest: Path, data_root: Path, scope: str) -> tuple[list[dict], dict]:
    source = json.loads(manifest.read_text(encoding="utf-8"))
    entries = source["datasets"]["deniz"]["files"]
    if scope == "all":
        verification = json.loads((data_root / "verification.reading-core.deniz.json").read_text(encoding="utf-8"))
        selected_hash = hashlib.sha256(json.dumps(sorted([("deniz", entry) for entry in entries],
                                                       key=lambda pair: pair[1]["path"]),
                                                 sort_keys=True).encode("utf-8")).hexdigest()
        if (verification["manifest_sha256"] != sha256(manifest) or verification["selection_sha256"] != selected_hash
                or verification["files"] != len(entries)):
            raise ValueError("The completed verification does not match this reading manifest.")
    else:
        entries = [entry for entry in entries if not entry["path"].startswith("responses/")]
        verification = {"scope": "support files only; no response arrays inspected"}
    checked = []
    for entry in entries:
        path = data_root / "raw" / "deniz" / entry["path"]
        receipt_path = data_root / ".receipts" / "deniz" / (entry["path"] + ".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity = hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()
        if (path.stat().st_size != entry["bytes"] or path.stat().st_mtime_ns != receipt["mtime_ns"]
                or receipt["source_fingerprint"] != identity):
            raise ValueError("Source identity, size, or modification time changed: " + str(path))
        checked.append({"path": entry["path"], "bytes": entry["bytes"], "sha256": receipt["sha256"]})
    return checked, verification


def inspect_dataset(manifest: Path, data_root: Path, output: Path, scope: str = "all", block_mib: int = 32) -> dict:
    checked, verification = check_sources(manifest, data_root, scope)
    raw = data_root / "raw" / "deniz"
    output.mkdir(parents=True, exist_ok=True)
    report = {"format_version": 1, "scope": scope, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "environment": {"python": sys.version, "platform": platform.platform(), "host": socket.gethostname(),
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                              "packages": {name: importlib.metadata.version(name)
                                           for name in ("numpy", "h5py", "praatio", "typing-extensions")}},
              "manifest_sha256": sha256(manifest), "inspection_code_sha256": sha256(Path(__file__)),
              "prior_verification": verification,
              "sources": checked, "issues": [], "hdf5": {}, "npz": {}, "mappers": {}}
    report["stimuli"] = inspect_stimuli(raw, report["issues"])
    print("Matched transcript identities against all 11 TextGrids.", flush=True)
    for index, entry in enumerate(checked, 1):
        path = raw / entry["path"]
        if path.suffix == ".hdf":
            print(f"[{index}/{len(checked)}] Inspecting all numeric arrays: {entry['path']}", flush=True)
            metadata = inspect_hdf(path, block_mib * 2**20)
            report["hdf5"][entry["path"]] = metadata
            if entry["path"].startswith("mappers/"):
                report["mappers"][path.stem.replace("_mappers", "")] = inspect_mapper(path, metadata, report["issues"])
            elif entry["path"].startswith(("responses/", "features/")):
                for name, array in metadata["datasets"].items():
                    numeric = array["numeric"]
                    if numeric.get("scanned") and numeric["finite"] != numeric["elements"]:
                        report["issues"].append({"severity": "warning", "location": entry["path"] + "/" + name,
                                                 "detail": "Nonfinite values exist; downstream fitting requires an explicit validity mask."})
        elif path.suffix == ".npz":
            arrays = {}
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    # Largest released motion-energy array is about 200 MB;
                    # only one array is retained, and summaries use small blocks.
                    array = archive[name]
                    arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                    "numeric": numeric_summary(array, block_mib * 2**20)}
                    del array
            report["npz"][entry["path"]] = arrays
    report["alignment"] = inspect_alignment(report)
    report["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["status"] = "needs_attention" if any(item["severity"] == "error" for item in report["issues"]) else "complete"
    write_json(output / "report.json", report)
    summary = format_summary(report)
    (output / "summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    # Small, actual inspection output only; no response arrays leave the cluster.
    bundle_path = output.parent / ("deniz-inspection.zip" if scope == "all" else "deniz-support-inspection.zip")
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.json", "summary.txt"):
            archive.write(output / name, arcname=name)
    print(summary, flush=True)
    print(f"REPORT BUNDLE: {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KiB)", flush=True)
    return report


def inspect_alignment(report: dict) -> dict:
    issues = report["issues"]
    features = {}
    for path, metadata in report["hdf5"].items():
        if path.startswith("features/"):
            for name, array in metadata["datasets"].items():
                story, feature = name.split("/", 1)
                features.setdefault(story, {})[feature] = array["shape"]
    frames = {}
    for story, arrays in features.items():
        lengths = {shape[0] for shape in arrays.values() if len(shape) == 2}
        if len(lengths) != 1 or any(len(shape) != 2 for shape in arrays.values()):
            issues.append({"severity": "error", "location": story, "detail": "Inconsistent released feature time axes."})
        else:
            frames[story] = lengths.pop()
    subjects = {}
    for path, metadata in report["hdf5"].items():
        if not path.startswith("responses/"):
            continue
        match = re.fullmatch(r"responses/(subject\d+)_reading_fmri_data_(trn|val)\.hdf", path)
        if not match:
            issues.append({"severity": "error", "location": path, "detail": "Unrecognized response filename."})
            continue
        subject, split = match.groups()
        dimensions = report["mappers"].get(subject, {}).get("voxel_dimensions", [])
        expected_stories = {f"story_{i:02d}" for i in range(1, 11)} if split == "trn" else {"story_11"}
        if set(metadata["datasets"]) != expected_stories:
            issues.append({"severity": "error", "location": path, "detail": "Unexpected response story keys."})
        for story, array in metadata["datasets"].items():
            shape = array["shape"]
            voxel_axes = [axis for axis, size in enumerate(shape) if size in dimensions]
            # The release README documents 5 trailing response TRs absent from
            # features. Verify this against every response array, not just one.
            time_axes = [axis for axis, size in enumerate(shape) if story in frames and size == frames[story] + 5]
            contract = len(voxel_axes) == len(time_axes) == 1 and voxel_axes[0] != time_axes[0]
            other_axes = [axis for axis in range(len(shape)) if axis not in voxel_axes + time_axes]
            # Actual release: subjects 04, 06, and 09 retain a length-one
            # repetition axis in training. Other subjects store time x voxel.
            expected_repeats = 1 if split == "trn" else 2
            layout_valid = (split == "trn" and len(shape) == 2 and not other_axes) or (
                len(shape) == 3 and len(other_axes) == 1 and shape[other_axes[0]] == expected_repeats)
            contract = contract and layout_valid
            record = {"shape": shape, "split": split, "contract_matches_release_documentation": bool(contract)}
            if contract:
                record.update(time_axis=time_axes[0], voxel_axis=voxel_axes[0],
                              repeat_axis=other_axes[0] if other_axes else None, repeats=expected_repeats,
                              timepoints=shape[time_axes[0]], voxels=shape[voxel_axes[0]])
            else:
                issues.append({"severity": "error", "location": path + "/" + story,
                               "detail": "Response shape differs from expected time/voxel/repetition contract; inspect original metadata."})
            subjects.setdefault(subject, {})[story] = record
    if report["scope"] == "all" and len(subjects) != 9:
        issues.append({"severity": "error", "location": "responses", "detail": "Expected all nine released reading subjects."})
    # This is an explicit length consistency check, not a claim that shapes
    # alone establish absolute stimulus onsets or interpolation conventions.
    expected_train = sum(n - 15 for story, n in frames.items() if story != "story_11")
    expected_validation = frames.get("story_11", 15) - 15
    moten = report["npz"].get("features/moth_en_moten_20210928.npz", {})
    motion_match = (moten.get("moten_Rstim", {}).get("shape", [None])[0] == expected_train and
                    moten.get("moten_Pstim", {}).get("shape", []) == [1, expected_validation, 6555])
    if not motion_match:
        issues.append({"severity": "error", "location": "motion energy", "detail": "Pretrimmed motion energy does not match per-story feature lengths."})
    return {"features": features, "feature_timepoints": frames, "subjects": subjects,
            "documented_tr_seconds": 2.0045,
            "published_trim_length_check": {"response_slice": "[10:-10] on time axis", "feature_slice": "[10:-5] on time axis",
                                            "train_timepoints": expected_train, "validation_timepoints": expected_validation,
                                            "matches_released_motion_energy": motion_match},
            "absolute_onset_note": "README includes leading silence in features and 5 extra trailing TRs in responses; array shapes alone cannot validate absolute word-to-scan timing."}


def format_summary(report: dict) -> str:
    lines = [f"DENIZ INSPECTION: {report['status'].upper()} | scope={report['scope']}",
             f"Sources: {len(report['sources'])}; HDF5 files: {len(report['hdf5'])}; manifest: {report['manifest_sha256']}",
             "Transcript -> timing identity (sequence similarity):"]
    for story, item in report["stimuli"]["stories"].items():
        lines.append(f"  {story} -> {item['textgrid']}: {item['sequence_similarity']:.6f}, {len(item['token_differences'])} token differences")
    for subject, stories in report["alignment"]["subjects"].items():
        validation = stories.get("story_11", {})
        lines.append(f"  {subject}: {len(stories)} stories, validation shape={validation.get('shape')}, "
                     f"repeats={validation.get('repeats', 'unresolved')}, voxels={validation.get('voxels', 'unresolved')}")
    trim = report["alignment"]["published_trim_length_check"]
    lines.append(f"Published trim length check: train={trim['train_timepoints']} TRs, validation={trim['validation_timepoints']} TRs; motion energy match={trim['matches_released_motion_energy']}")
    errors = sum(item["severity"] == "error" for item in report["issues"])
    warnings = sum(item["severity"] == "warning" for item in report["issues"])
    lines.append(f"Issues: {errors} errors; {warnings} warnings. Full details in report.json.")
    for item in report["issues"]:
        if item["severity"] == "error":
            lines.append(f"  ERROR {item['location']}: {item['detail']}")
    lines.append("Full response inspection completed." if report["scope"] == "all" else "Support files inspected; no response arrays were inspected in this run.")
    return "\n".join(lines) + "\n"
