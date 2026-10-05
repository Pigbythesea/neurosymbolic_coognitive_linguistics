"""Install a reviewed local bundle only inside an allocated compute-node session."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import tempfile
import zipfile

ROOT = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0].startswith('login'):
        raise SystemExit('STOP: installation requires an allocated compute node.')
    if ROOT.resolve() != ROOT or not ROOT.is_dir():
        raise SystemExit('STOP: inspected project root is missing or resolves elsewhere.')
    bundle_path = Path(sys.argv[1]).resolve()
    if not bundle_path.is_relative_to(ROOT):
        raise SystemExit('STOP: source bundle must be inside the inspected project.')
    receipt_path = ROOT / '.extraction-source.json'
    if receipt_path.is_symlink():
        raise SystemExit('STOP: deployment receipt is a symlink.')
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'files': {}}
    changes = []
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read('BUNDLE.json'))
        if set(bundle.namelist()) != set(manifest['files']) | {'BUNDLE.json'} or len(bundle.namelist()) != len(set(bundle.namelist())):
            raise SystemExit('STOP: bundle member list differs from the manifest.')
        for name, digest in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise SystemExit('STOP: invalid bundle path.')
            path = ROOT.joinpath(*relative.parts)
            if path.resolve() != path or not path.is_relative_to(ROOT):
                raise SystemExit('STOP: destination resolves elsewhere: ' + name)
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise SystemExit('STOP: bundle content hash mismatch: ' + name)
            before = path.read_bytes() if path.is_file() else None
            if path.exists() and before is None:
                raise SystemExit('STOP: destination is not a file: ' + name)
            if before is not None and before != data:
                # Never replace shared inputs or an independently edited source.
                if name.startswith('data/') or hashlib.sha256(before).hexdigest() != previous['files'].get(name):
                    raise SystemExit('STOP: existing cluster file differs; no files changed: ' + name)
            changes.append((path, data, before))
    for path, data, before in changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        if before == data:
            continue
        if before is None:
            with path.open('xb') as stream:
                stream.write(data)
        else:
            backup = ROOT / 'artifacts/extraction-source-backups' / hashlib.sha256(before).hexdigest() / path.relative_to(ROOT)
            if backup.resolve() != backup:
                raise SystemExit('STOP: source backup resolves elsewhere.')
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(before)
            if path.read_bytes() != before:
                raise SystemExit('STOP: concurrent cluster source edit: ' + str(path))
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, path)
    descriptor, temporary = tempfile.mkstemp(dir=ROOT, prefix='.extraction-source.')
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary, receipt_path)
    ignore = ROOT / '.gitignore'
    if ignore.resolve() != ignore:
        raise SystemExit('STOP: .gitignore resolves elsewhere.')
    current = ignore.read_text(encoding='utf-8') if ignore.exists() else ''
    additions = [line for line in ('.venv-extraction/', '.extraction-source.json') if line not in current.splitlines()]
    if additions:
        with ignore.open('a', encoding='utf-8') as stream:
            stream.write(('\n' if current and not current.endswith('\n') else '') + '\n'.join(additions) + '\n')
    print('EXTRACTION SOURCES INSTALLED:', len(changes), 'files; annotation files excluded.', flush=True)


if __name__ == '__main__':
    main()
