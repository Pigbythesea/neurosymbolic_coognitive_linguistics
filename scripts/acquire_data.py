"""Inventory and acquire the real study data, using Python's standard library.

The inventory pins Deniz to a Git commit and records OpenNeuro object identities.
Downloads use curl, retain interrupted files, and write SHA-256 provenance receipts.
No Git-annex installation, AWS account, or Python dependencies are needed.
Use plan to inspect sizes; download and verify require an explicit --profile.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "datasets.json"
DENIZ_BASE = "https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri"
DENIZ_REVISION = "d51d64bf5026441edfbec804d6ffeaea92bb3018"
S3_BASE = "https://s3.amazonaws.com/openneuro.org"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PROFILES = {
    "stimuli": "Deniz transcripts and word timings for local annotation; no MRI, features, or models.",
    "reading-core": "All Deniz reading responses, text, timing, features, cortical mappers, and source documentation.",
    "local": "Reading core plus LeBel and Reading Brain text, timing, reference features, and metadata; no additional MRI or audio.",
    "full": "Every file in the manifest, including listening responses and Reading Brain raw MRI.",
}


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def curl_binary() -> str:
    found = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not found:
        raise RuntimeError("curl is required; it is included with Windows 10/11.")
    return found


def request(url: str, *, head: bool = False, max_bytes: int | None = None) -> bytes:
    command = [curl_binary(), "-q", "--fail", "--silent", "--show-error", "--location",
               "--connect-timeout", "10", "--max-time", "25"]
    if head:
        command.append("--head")
    if max_bytes is not None:
        command += ["--max-filesize", str(max_bytes)]
    result = subprocess.run(command + [url], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Cannot access {url}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_target(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(
        part in (".", "..") or ":" in part or "\\" in part for part in parts
    ):
        raise ValueError(f"Unsafe data path: {relative!r}")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Data path leaves destination: {relative!r}")
    return target


def s3_inventory(dataset_id: str) -> list[dict]:
    objects = []
    token = None
    seen_tokens = set()
    while True:
        params = {"list-type": "2", "prefix": dataset_id + "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        tree = ET.fromstring(request(S3_BASE + "?" + urllib.parse.urlencode(params)))
        for node in tree.findall("s:Contents", NS):
            key = node.findtext("s:Key", namespaces=NS)
            if not key or not key.startswith(dataset_id + "/"):
                raise RuntimeError("Unexpected object key in the OpenNeuro listing.")
            if key.endswith("/"):
                continue
            objects.append({
                "path": key[len(dataset_id) + 1:],
                "url": S3_BASE + "/" + urllib.parse.quote(key, safe="/"),
                "bytes": int(node.findtext("s:Size", namespaces=NS)),
                "etag": node.findtext("s:ETag", namespaces=NS),
                "last_modified": node.findtext("s:LastModified", namespaces=NS),
            })
        if tree.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        token = tree.findtext("s:NextContinuationToken", namespaces=NS)
        if not token or token in seen_tokens:
            raise RuntimeError("Invalid OpenNeuro pagination token; inventory is incomplete.")
        seen_tokens.add(token)
    if not objects:
        raise RuntimeError(f"No public objects found for {dataset_id}.")
    return objects


def select_openneuro(dataset_id: str, entries: list[dict]) -> list[dict]:
    selected = []
    for entry in entries:
        name = entry["path"]
        parts = PurePosixPath(name).parts
        if any(part.startswith(".") for part in parts) or name == "annex-uuid":
            continue
        if dataset_id == "ds003020":
            keep = (name.startswith(("derivatives/", "stimuli/")) or len(parts) == 1
                    or name.endswith((".json", ".tsv")))
        elif dataset_id == "ds003974":
            keep = (len(parts) == 1 or "/anat/" in name or "/fmap/" in name
                    or ("/func/" in name and "task-read" in parts[-1]))
        else:
            raise ValueError(dataset_id)
        if keep:
            selected.append(entry)
    return selected


def deniz_inventory() -> list[dict]:
    prefix = "/denizenslab/narratives_reading_listening_fmri/src/" + DENIZ_REVISION + "/"
    pending = [""]
    visited = set()
    files = set()
    directories = {"code", "features", "mappers", "responses", "stimuli"}
    metadata = {"README.md", "LICENSE", "datacite.yml"}
    while pending:
        folder = pending.pop()
        if folder in visited:
            continue
        visited.add(folder)
        url = DENIZ_BASE + "/src/" + DENIZ_REVISION
        if folder:
            url += "/" + urllib.parse.quote(folder, safe="/")
        document = request(url).decode("utf-8")
        rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", document, re.S)
        children = 0
        for row in rows:
            cell = re.search(r'<td\b[^>]*class="name"[^>]*>(.*?)</td>', row, re.S)
            if not cell:
                continue
            link = re.search(r'href="([^"]+)"', cell.group(1))
            if not link:
                continue
            href = html.unescape(link.group(1))
            if not href.startswith(prefix):
                continue
            relative = urllib.parse.unquote(href[len(prefix):])
            if PurePosixPath(relative).parent.as_posix() != (folder or "."):
                continue
            children += 1
            is_directory = "octicon-file-directory" in cell.group(1)
            if is_directory and (folder or relative in directories):
                pending.append(relative)
            elif not is_directory and (folder or relative in metadata):
                if not PurePosixPath(relative).name.startswith("."):
                    files.add(relative)
        if not children:
            raise RuntimeError(f"Could not parse the Deniz directory: {url}")
    if not any(path.startswith("responses/") for path in files):
        raise RuntimeError("Deniz response inventory is empty.")

    def describe(path: str) -> dict:
        url = DENIZ_BASE + "/raw/" + DENIZ_REVISION + "/" + urllib.parse.quote(path, safe="/")
        headers = request(url, head=True).decode("iso-8859-1")
        lengths = re.findall(r"^content-length:\s*(\d+)\s*$", headers, re.I | re.M)
        if not lengths:
            # Ordinary Git blobs (including a small feature HDF5) are chunked.
            # Inspect their real bytes; never accidentally download a large MRI
            # payload during inventory when a server omits Content-Length.
            payload = request(url, max_bytes=32 * 2**20)
            return {"path": path, "url": url, "bytes": len(payload),
                    "git_revision": DENIZ_REVISION,
                    "source_sha256": hashlib.sha256(payload).hexdigest()}
        return {"path": path, "url": url, "bytes": int(lengths[-1]),
                "git_revision": DENIZ_REVISION}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(describe, sorted(files)))


def summary(datasets: dict) -> dict:
    result = {}
    for name, dataset in datasets.items():
        entries = dataset["files"]
        total = sum(entry["bytes"] for entry in entries)
        groups = {}
        for entry in entries:
            parts = PurePosixPath(entry["path"]).parts
            group = "/".join(parts[:2]) if len(parts) > 2 and parts[0] == "derivatives" else parts[0]
            if entry["path"].startswith("derivatives/preprocessed_data/") and len(parts) > 3:
                group = "/".join(parts[:3])
            if name == "deniz" and parts[0] == "responses":
                group = "responses/" + ("reading" if "_reading_" in parts[-1] else "listening")
            if len(parts) == 1:
                group = "root_metadata"
            groups[group] = groups.get(group, 0) + entry["bytes"]
        result[name] = {"files": len(entries), "bytes": total,
                        "GiB": round(total / 2**30, 2),
                        "groups_GiB": {group: round(size / 2**30, 3)
                                       for group, size in sorted(groups.items())}}
    return result


def inventory(manifest_path: Path) -> None:
    if manifest_path.exists():
        raise RuntimeError(f"Manifest already exists: {manifest_path}. Reuse it or choose a new --manifest path.")
    datasets = {}
    print("Inventorying Deniz at commit " + DENIZ_REVISION, flush=True)
    datasets["deniz"] = {"source": DENIZ_BASE, "revision": DENIZ_REVISION,
                         "selection": "Published responses, stimuli, features, mappers, code, and metadata.",
                         "files": deniz_inventory()}
    for name, dataset_id in (("lebel", "ds003020"), ("reading_brain", "ds003974")):
        print("Inventorying " + dataset_id, flush=True)
        files = select_openneuro(dataset_id, s3_inventory(dataset_id))
        changes = request(S3_BASE + "/" + dataset_id + "/CHANGES").decode("utf-8-sig")
        datasets[name] = {
            "source": "https://openneuro.org/datasets/" + dataset_id,
            "release_note": changes.splitlines()[0],
            "selection": ("All released derivatives, audio stimuli, and acquisition metadata."
                          if name == "lebel" else
                          "Reading-task BOLD, anatomy, field maps, text, eye tracking, and participant metadata."),
            "files": sorted(files, key=lambda item: item["path"]),
        }
    manifest = {"format_version": 1, "created_utc": timestamp(), "datasets": datasets}
    for name, dataset in datasets.items():
        paths = [entry["path"] for entry in dataset["files"]]
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Duplicate download paths in {name}.")
        for relative in paths:
            safe_target(ROOT / "data" / "raw" / name, relative)
    atomic_json(manifest_path, manifest)
    report = summary(datasets)
    report["total_GiB"] = round(sum(item["bytes"] for item in report.values()) / 2**30, 2)
    report["workspace_free_GiB"] = round(shutil.disk_usage(ROOT).free / 2**30, 2)
    print(json.dumps(report, indent=2))
    print("Manifest: " + str(manifest_path), flush=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def receipt_path(data_root: Path, dataset: str, relative: str) -> Path:
    return safe_target(data_root / ".receipts" / dataset, relative + ".json")


def validate_file(target: Path, entry: dict) -> None:
    if target.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"Size mismatch: {target}")
    with target.open("rb") as stream:
        start = stream.read(512)
    lower = start.lstrip().lower()
    if lower.startswith((b"<!doctype html", b"<html", b"<?xml", b".git/annex/objects/")):
        raise RuntimeError(f"Received a webpage, XML error, or annex pointer instead of data: {target}")
    suffix = PurePosixPath(entry["path"]).suffix.lower()
    if suffix in (".hdf", ".h5", ".hdf5", ".hf5"):
        signature = b"\x89HDF\r\n\x1a\n"
        with target.open("rb") as stream:
            found = False
            for offset in (0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
                stream.seek(offset)
                if stream.read(8) == signature:
                    found = True
                    break
        if not found:
            raise RuntimeError(f"Missing HDF5 file signature: {target}")
    if entry["path"].endswith(".nii.gz") and not start.startswith(b"\x1f\x8b"):
        raise RuntimeError(f"Missing gzip signature for NIfTI file: {target}")
    if suffix in (".xlsx", ".npz") and not start.startswith(b"PK"):
        raise RuntimeError(f"Missing ZIP container signature: {target}")


def run_transfer(command: list[str], stop: threading.Event) -> int:
    if stop.is_set():
        raise InterruptedError("Transfer cancelled; partial data retained.")
    process = subprocess.Popen(command)
    try:
        while True:
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    raise InterruptedError("Transfer cancelled; partial data retained.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def fetch_one(data_root: Path, dataset: str, entry: dict, *,
              stop: threading.Event | None = None, quiet: bool = False) -> str:
    stop = stop if stop is not None else threading.Event()
    if stop.is_set():
        raise InterruptedError("Transfer cancelled; partial data retained.")
    target = safe_target(data_root / "raw" / dataset, entry["path"])
    receipt = receipt_path(data_root, dataset, entry["path"])
    identity = fingerprint(entry)
    if target.exists():
        if not receipt.exists():
            raise RuntimeError(f"Existing file has no provenance receipt: {target}. Move it aside before downloading.")
        saved = read_json(receipt)
        if (saved.get("source_fingerprint") != identity or target.stat().st_size != entry["bytes"]
                or target.stat().st_mtime_ns != saved.get("mtime_ns")):
            raise RuntimeError(f"Existing data or source identity changed: {target}. Run verify and inspect this file.")
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial_meta = target.with_name(target.name + ".part.json")
    if partial.exists():
        if not partial_meta.exists() or read_json(partial_meta).get("source_fingerprint") != identity:
            raise RuntimeError(f"Cannot resume a partial file with different or missing source identity: {partial}")
        if partial.stat().st_size > entry["bytes"]:
            raise RuntimeError(f"Partial file is larger than the source: {partial}")
    else:
        atomic_json(partial_meta, {"source_fingerprint": identity, "source": entry})
    if not partial.exists() or partial.stat().st_size < entry["bytes"]:
        command = [curl_binary(), "-q", "--fail", "--location", "--show-error",
                   "--silent" if quiet else "--progress-bar",
                   "--connect-timeout", "30", "--retry", "3", "--retry-delay", "2",
                   "--speed-limit", "1024", "--speed-time", "120",
                   "--output", str(partial)]
        if entry.get("etag"):
            command += ["--header", "If-Match: " + entry["etag"]]
        returncode = run_transfer(command + ["--continue-at", "-", entry["url"]], stop)
        if returncode == 33:
            print("Server cannot resume this file; restarting its retained partial download.", flush=True)
            returncode = run_transfer(command + [entry["url"]], stop)
        if returncode:
            raise RuntimeError(f"Download failed (curl {returncode}): {entry['url']}. Partial data retained.")
    if stop.is_set():
        raise InterruptedError("Transfer cancelled; partial data retained.")
    validate_file(partial, entry)
    sha256 = digest(partial)
    if entry.get("source_sha256") and sha256 != entry["source_sha256"]:
        raise RuntimeError(f"Source SHA-256 mismatch: {target}")
    atomic_json(receipt, {"source_fingerprint": identity, "source": entry,
                          "sha256": sha256, "mtime_ns": partial.stat().st_mtime_ns,
                          "completed_utc": timestamp()})
    os.replace(partial, target)
    partial_meta.unlink(missing_ok=True)
    return "downloaded"


def included_in_profile(profile: str, dataset: str, path: str) -> bool:
    if profile == "stimuli":
        return dataset == "deniz" and path.startswith("stimuli/") and path.lower().endswith((".txt", ".textgrid"))
    if profile == "full":
        return True
    if dataset == "deniz":
        return "_listening_" not in path and not path.lower().endswith(".wav")
    if profile == "reading-core":
        return False
    if dataset == "lebel":
        return not (path.startswith(("derivatives/preprocessed_data/",
                                     "derivatives/freesurfer_subjdir/",
                                     "derivatives/pycortex-db/"))
                    or path.lower().endswith(".wav"))
    if dataset == "reading_brain":
        return len(PurePosixPath(path).parts) == 1 or path.endswith((".json", ".tsv"))
    raise ValueError(f"No profile selection rule for {dataset}.")


def selected_entries(manifest_path: Path, selected: list[str] | None,
                     profile: str, *, allow_empty: bool = False) -> list[tuple[str, dict]]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown download profile: {profile}")
    manifest = read_json(manifest_path)
    if manifest.get("format_version") != 1:
        raise RuntimeError("Unsupported download manifest version.")
    datasets = manifest["datasets"]
    names = list(dict.fromkeys(selected or datasets))
    missing = set(names) - set(datasets)
    if missing:
        raise RuntimeError("Unknown datasets: " + ", ".join(sorted(missing)))
    result = [(name, entry) for name in names for entry in datasets[name]["files"]
              if included_in_profile(profile, name, entry["path"])]
    if not result and not allow_empty:
        raise RuntimeError("No files match this profile and dataset selection.")
    for _, entry in result:
        url = urllib.parse.urlparse(entry["url"])
        if url.scheme != "https" or url.hostname not in ("gin.g-node.org", "s3.amazonaws.com"):
            raise RuntimeError("Unexpected download host in manifest.")
    return result


def selection_report(entries: list[tuple[str, dict]], profile: str) -> dict:
    return {"profile": profile, "description": PROFILES[profile],
            "datasets": sorted({name for name, _ in entries}),
            "files": len(entries), "bytes": sum(entry["bytes"] for _, entry in entries),
            "selection_sha256": hashlib.sha256(
                json.dumps(sorted(entries, key=lambda pair: (pair[0], pair[1]["path"])),
                           sort_keys=True).encode("utf-8")).hexdigest()}


def completion_path(data_root: Path, action: str, report: dict) -> Path:
    # A metadata-only selection must never look like a completed full corpus.
    scope = report["profile"] + "." + "+".join(report["datasets"])
    return data_root / f"{action}.{scope}.json"


def plan(manifest_path: Path, selected: list[str] | None, profile: str | None) -> None:
    reports = {}
    for name in ([profile] if profile else PROFILES):
        entries = selected_entries(manifest_path, selected, name, allow_empty=True)
        report = selection_report(entries, name)
        groups = {dataset: {"files": [entry for item, entry in entries if item == dataset]}
                  for dataset in report["datasets"]}
        reports[name] = {**report, "GiB": round(report["bytes"] / 2**30, 2),
                         "contents": summary(groups)}
    print(json.dumps(reports, indent=2))


def download(manifest_path: Path, data_root: Path, selected: list[str] | None,
             profile: str, jobs: int = 1) -> None:
    entries = selected_entries(manifest_path, selected, profile)
    data_root.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for name, entry in entries:
        target = safe_target(data_root / "raw" / name, entry["path"])
        partial = target.with_name(target.name + ".part")
        if not target.exists():
            remaining += max(0, entry["bytes"] - (partial.stat().st_size if partial.exists() else 0))
    free = shutil.disk_usage(data_root).free
    if free < remaining + 2**30:
        raise RuntimeError(f"Need {remaining / 2**30:.2f} GiB plus 1 GiB free space; have {free / 2**30:.2f} GiB. "
                           "Use --data-root on a larger disk.")
    print(f"Profile: {profile}; files: {len(entries)}; jobs: {jobs}; "
          f"remaining download: {remaining / 2**30:.2f} GiB",
          flush=True)
    counts = {"cached": 0, "downloaded": 0}
    stop = threading.Event()

    def transfer(item: tuple[str, dict]) -> str:
        name, entry = item
        return fetch_one(data_root, name, entry, stop=stop, quiet=jobs > 1 or not sys.stderr.isatty())

    # Keep at most `jobs` requests in flight. A failure prevents further starts
    # and stops the other curl processes while preserving resumable partials.
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {}
        iterator = iter(entries)

        def enqueue() -> None:
            item = next(iterator, None)
            if item is not None:
                name, entry = item
                print(f"START {name}/{entry['path']} ({entry['bytes'] / 2**20:.2f} MiB)", flush=True)
                pending[pool.submit(transfer, item)] = item

        last_report = time.monotonic()
        try:
            for _ in range(jobs):
                enqueue()
            while pending:
                done, _ = concurrent.futures.wait(pending, timeout=30,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                completed = []
                for future in done:
                    name, entry = pending.pop(future)
                    status = future.result()
                    counts[status] += 1
                    completed.append(future)
                    print(f"[{sum(counts.values())}/{len(entries)}] {status.upper()} "
                          f"{name}/{entry['path']}", flush=True)
                for _ in completed:
                    enqueue()
                if time.monotonic() - last_report >= 30 and pending:
                    # Report actual retained bytes, including files currently
                    # being checked. This is progress, not an integrity claim.
                    retained = 0
                    for name, entry in entries:
                        target = safe_target(data_root / "raw" / name, entry["path"])
                        partial = target.with_name(target.name + ".part")
                        for candidate in (target, partial):
                            try:
                                retained += min(candidate.stat().st_size, entry["bytes"])
                                break
                            except FileNotFoundError:
                                pass
                    print(f"PROGRESS {sum(counts.values())}/{len(entries)} files complete; "
                          f"{retained / 2**30:.2f} GiB on disk; {len(pending)} active", flush=True)
                    last_report = time.monotonic()
        finally:
            stop.set()
            for future in pending:
                future.cancel()
    result = {"completed_utc": timestamp(), "manifest_sha256": digest(manifest_path),
              **selection_report(entries, profile), **counts}
    atomic_json(completion_path(data_root, "download_complete", result), result)
    print("DOWNLOAD COMPLETE " + json.dumps(result), flush=True)


def verify(manifest_path: Path, data_root: Path, selected: list[str] | None,
           profile: str) -> None:
    entries = selected_entries(manifest_path, selected, profile)
    total = 0
    for index, (name, entry) in enumerate(entries, 1):
        target = safe_target(data_root / "raw" / name, entry["path"])
        receipt = receipt_path(data_root, name, entry["path"])
        if not target.is_file() or not receipt.is_file():
            raise RuntimeError(f"Missing data or receipt: {target}")
        saved = read_json(receipt)
        if saved.get("source_fingerprint") != fingerprint(entry):
            raise RuntimeError(f"Source identity mismatch: {target}")
        validate_file(target, entry)
        if digest(target) != saved.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch: {target}")
        total += entry["bytes"]
        if index % 100 == 0 or index == len(entries):
            print(f"Verified {index}/{len(entries)} files; {total / 2**30:.2f} GiB", flush=True)
    report = {"verified_utc": timestamp(), "manifest_sha256": digest(manifest_path),
              **selection_report(entries, profile)}
    atomic_json(completion_path(data_root, "verification", report), report)
    print("VERIFICATION COMPLETE " + json.dumps(report), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "plan", "download", "verify"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", action="append", choices=("deniz", "lebel", "reading_brain"))
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        help="Required for download/verify. Plan shows all profiles when omitted.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Concurrent file downloads, 1 to 8 (default: 1).")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8.")
    if args.action in ("download", "verify") and args.profile is None:
        parser.error("--profile is required; use reading-core for the complete reading dataset.")
    if args.action == "inventory" and (args.dataset or args.profile):
        parser.error("inventory records all sources; profiles and datasets select files for later actions.")
    try:
        if args.action == "inventory":
            inventory(args.manifest.resolve())
        elif args.action == "plan":
            plan(args.manifest.resolve(), args.dataset, args.profile)
        elif args.action == "download":
            download(args.manifest.resolve(), args.data_root.resolve(), args.dataset, args.profile, args.jobs)
        else:
            verify(args.manifest.resolve(), args.data_root.resolve(), args.dataset, args.profile)
    except (RuntimeError, ValueError, OSError, ET.ParseError, subprocess.TimeoutExpired) as error:
        print("STOP: " + str(error), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Partial downloads are retained; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    # Slurm sends TERM before enforcing a time limit. Unwind the worker pool so
    # active curl processes stop and the next job can resume their partials.
    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
