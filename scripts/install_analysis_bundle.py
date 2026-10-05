"""Hash-checked installation, only on an allocated node in the inspected project."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import tempfile
import zipfile

ROOT = Path("/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics")


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0].startswith("login"):
        raise SystemExit("STOP: analysis installation requires an allocated compute node.")
    if ROOT.resolve() != ROOT or not ROOT.is_dir():
        raise SystemExit("STOP: inspected project workspace is missing or resolves elsewhere.")
    bundle_path = Path(sys.argv[1]).resolve()
    if not bundle_path.is_relative_to(ROOT):
        raise SystemExit("STOP: bundle is outside the inspected project.")
    receipt = ROOT / ".analysis-source.json"
    if receipt.is_symlink():
        raise SystemExit("STOP: source receipt is a symlink.")
    previous = json.loads(receipt.read_text()) if receipt.exists() else {"files": {}}
    changes = []
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("BUNDLE.json"))
        if set(bundle.namelist()) != set(manifest["files"]) | {"BUNDLE.json"} or len(bundle.namelist()) != len(set(bundle.namelist())):
            raise SystemExit("STOP: bundle inventory differs from manifest.")
        for name, digest in manifest["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise SystemExit("STOP: invalid archive path.")
            path = ROOT.joinpath(*relative.parts)
            if path.resolve() != path or not path.is_relative_to(ROOT):
                raise SystemExit("STOP: destination resolves elsewhere: " + name)
            body = bundle.read(name)
            if hashlib.sha256(body).hexdigest() != digest:
                raise SystemExit("STOP: bundle content changed: " + name)
            before = path.read_bytes() if path.is_file() else None
            if path.exists() and before is None:
                raise SystemExit("STOP: destination is not a regular file: " + name)
            if before is not None and before != body:
                immutable = name.startswith("analysis_code/") or (name.startswith("data/") and not name.endswith("/latest.json"))
                if immutable or hashlib.sha256(before).hexdigest() != previous["files"].get(name):
                    raise SystemExit("STOP: independently edited/shared cluster file differs; no files changed: " + name)
            changes.append((path, body, before))
    # Validate the isolated entry point before making any changes.
    entry = manifest["code_root"] + "/scripts/run_analysis.py"
    if not manifest["code_root"].startswith("analysis_code/") or entry not in manifest["files"]:
        raise SystemExit("STOP: invalid isolated analysis entry point.")
    for path, body, before in changes:
        if body == before:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if before is not None:
            backup = ROOT / "artifacts/analysis-source-backups" / hashlib.sha256(before).hexdigest() / path.relative_to(ROOT)
            if backup.resolve() != backup:
                raise SystemExit("STOP: backup resolves elsewhere.")
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(before)
            if path.read_bytes() != before:
                raise SystemExit("STOP: concurrent source edit.")
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
        os.replace(temporary, path)
    descriptor, temporary = tempfile.mkstemp(dir=ROOT, prefix=".analysis-source.")
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary, receipt)
    print("ANALYSIS SOURCES INSTALLED:", len(changes), "files; code root", manifest["code_root"], flush=True)


if __name__ == "__main__":
    main()
