"""Package the actual project sources for pasting into an existing SSH session.

This runs locally. It makes no network connection and never executes the remote
setup. The generated script refuses to run outside a Slurm compute allocation.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import textwrap
import zipfile

from acquire_data import atomic_json, selected_entries, selection_report


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_ROOT = "/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics"
ORIGIN = "https://github.com/Pigbythesea/neurosymbolic_coognitive_linguistics.git"
FILES = sorted({
    ".gitignore",
    ".gitattributes",
    "manifests/deniz-reading.json",
    "manifests/deniz-reading-responses.urls.txt",
    *(path.relative_to(ROOT).as_posix()
      for folder in ("scripts", "neurosym", "requirements", "configs", "prompts", "schemas")
      for path in (ROOT / folder).rglob("*")
      if path.is_file() and path.suffix in (".py", ".sh", ".sbatch", ".cmd", ".txt", ".json")),
})


SETUP = '''import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import tempfile
import zipfile

if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0].startswith("login"):
    raise SystemExit("STOP: paste this into an allocated compute-node shell, not the login node.")
if sys.version_info < (3, 11):
    raise SystemExit("STOP: Python 3.11 or newer is required.")

root = Path(CLUSTER_ROOT)
if not root.parent.is_dir() or root.parent.resolve() != root.parent:
    raise SystemExit("STOP: the inspected project parent is missing or resolves elsewhere.")
if root.resolve() != root:
    raise SystemExit("STOP: the project directory resolves outside the expected location.")

payload = base64.b64decode(PAYLOAD, validate=False)
if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
    raise SystemExit("STOP: the pasted package is incomplete or corrupted. No setup was performed.")

receipt_path = root / ".cluster-source.json"
if receipt_path.resolve() != receipt_path:
    raise SystemExit("STOP: source receipt resolves elsewhere.")
previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
previous_files = previous.get("files", {})
if not isinstance(previous_files, dict):
    raise SystemExit("STOP: invalid previous source receipt.")

planned = []
seen = set()
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for entry in bundle.infolist():
        relative = PurePosixPath(entry.filename)
        if (relative.is_absolute() or ".." in relative.parts or "\\\\" in entry.filename
                or entry.is_dir() or entry.filename in seen):
            raise SystemExit("STOP: invalid archive member.")
        seen.add(entry.filename)
        target = root.joinpath(*relative.parts)
        if target.resolve() != target or not target.is_relative_to(root):
            raise SystemExit("STOP: a destination resolves outside its expected path.")
        content = bundle.read(entry)
        before = target.read_bytes() if target.is_file() else None
        if target.exists() and not target.is_file():
            raise SystemExit("STOP: source destination is not a file: " + str(target))
        if before is not None and before != content:
            if hashlib.sha256(before).hexdigest() != previous_files.get(entry.filename):
                raise SystemExit("STOP: source was edited on the cluster; nothing overwritten: " + str(target))
        planned.append((target, content, before))

os.umask(0o077)
root.mkdir(exist_ok=True)
for target, content, before in planned:
    target.parent.mkdir(parents=True, exist_ok=True)
    if before is None:
        # Exclusive creation prevents overwriting a concurrent edit.
        with target.open("xb") as stream:
            stream.write(content)
    elif before != content:
        # Only replace sources matching their prior deployment receipt. Retain
        # their exact previous bytes so an update is reversible without Git commits.
        backup = root / "artifacts/source-backups" / hashlib.sha256(before).hexdigest() / target.relative_to(root)
        if backup.resolve() != backup:
            raise SystemExit("STOP: backup destination resolves elsewhere.")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            if backup.read_bytes() != before:
                raise SystemExit("STOP: inconsistent source backup.")
        else:
            with backup.open("xb") as stream:
                stream.write(before)
        if target.read_bytes() != before:
            raise SystemExit("STOP: source changed during setup: " + str(target))
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
for name in ("data", "logs", ".tmp"):
    target = root / name
    if target.resolve() != target:
        raise SystemExit("STOP: output directory is a symlink: " + str(target))
    target.mkdir(exist_ok=True)

if not (root / ".git").exists():
    subprocess.run(["git", "init", "--initial-branch=main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", ORIGIN], check=True)

receipt = {
    "bundle_sha256": EXPECTED_SHA256,
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "files": {**previous_files, **{path.relative_to(root).as_posix(): hashlib.sha256(content).hexdigest()
                                  for path, content, _ in planned}},
}
descriptor, temporary = tempfile.mkstemp(prefix=".cluster-source.", dir=root)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\\n")
os.replace(temporary, receipt_path)
print("SETUP COMPLETE:", root)
print("Source files:", len(planned), "| Existing cluster edits protected; replaced source bytes backed up.")
print("Project sources are ready. Submit the task-specific job separately.")
'''


def main() -> None:
    # Transfer the exact complete reading selection. The full multi-corpus
    # inventory and proposal documents remain local; source entry identities
    # are unchanged and their original manifest hash is recorded.
    original_path = ROOT / "manifests" / "datasets.json"
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    entries = selected_entries(original_path, None, "reading-core")
    reading_path = ROOT / "manifests" / "deniz-reading.json"
    atomic_json(reading_path, {
        "format_version": 1,
        "created_utc": original["created_utc"],
        "derived_from_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "selection": selection_report(entries, "reading-core"),
        "datasets": {"deniz": {
            **original["datasets"]["deniz"],
            "selection": "Complete released reading responses and associated text, timing, features, mappers, and code.",
            "files": [entry for _, entry in entries],
        }},
    })
    if selected_entries(reading_path, None, "reading-core") != entries:
        raise RuntimeError("The reading manifest differs from the original source selection.")
    contents = {}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in FILES:
            path = ROOT.joinpath(*PurePosixPath(name).parts)
            content = path.read_bytes()
            if path.suffix in (".py", ".sh", ".sbatch"):
                content = content.replace(b"\r\n", b"\n")
            if path.suffix == ".py":
                ast.parse(content, filename=name, feature_version=(3, 11))
            contents[name] = content
            member = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(member, content)
    payload = archive.getvalue()
    sha256 = hashlib.sha256(payload).hexdigest()
    encoded = "\n".join(textwrap.wrap(base64.b64encode(payload).decode("ascii"), width=76))
    code = (f"CLUSTER_ROOT = {CLUSTER_ROOT!r}\nORIGIN = {ORIGIN!r}\n"
            f"EXPECTED_SHA256 = {sha256!r}\nPAYLOAD = '''\n{encoded}\n'''\n" + SETUP)
    ast.parse(code, filename="cluster-setup", feature_version=(3, 11))

    # Verify the actual generated archive, not a synthetic transfer fixture.
    restored = base64.b64decode(encoded)
    if restored != payload:
        raise RuntimeError("Base64 package validation failed.")
    with zipfile.ZipFile(io.BytesIO(restored)) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(FILES):
            raise RuntimeError("Archive validation failed.")
        for name, content in contents.items():
            if bundle.read(name) != content:
                raise RuntimeError("Packaged source differs: " + name)

    script = ("(\nset -eu\n"
              'case "$(hostname -s)" in login*) printf "%s\\n" "STOP: use an allocated compute node." >&2; exit 1;; esac\n'
              'if [ -z "${SLURM_JOB_ID:-}" ]; then printf "%s\\n" "STOP: no Slurm allocation." >&2; exit 1; fi\n'
              "/usr/bin/python3.11 -I -B - <<'NEUROSYM_CLUSTER_SETUP'\n" + code
              + "NEUROSYM_CLUSTER_SETUP\n)\n")
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "cluster-setup.sh").write_text(script, encoding="utf-8", newline="\n")
    (output / "cluster-source.zip").write_bytes(payload)
    report = {"source_files": len(FILES), "archive_bytes": len(payload),
              "paste_script_bytes": len(script.encode("utf-8")), "archive_sha256": sha256,
              "cluster_root": CLUSTER_ROOT,
              "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
    (output / "cluster-source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files_sha256"}, indent=2))
    print("Paste script: " + str(output / "cluster-setup.sh"))
    print("Validated source bytes, archive CRCs, SHA-256, and Python 3.11 syntax.")


if __name__ == "__main__":
    main()
