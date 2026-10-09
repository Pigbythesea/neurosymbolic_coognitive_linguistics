"""Prepare a small versioned administrative packet; leave analysis ZIP untouched."""
import hashlib
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_plan import CODE_ROOT, MANIFEST, digest, load_manifest, make_plan, read, write
from verify_production import validate


def main():
    root = Path(__file__).resolve().parents[1]
    verification = root / 'artifacts/production-verification.json'
    qualification = read(verification) if verification.exists() else {}
    if (qualification.get('status') != 'verified' or not qualification.get('verified_source_files') or
            any(hashlib.sha256((root / 'scripts' / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != h
                for n, h in qualification.get('verified_source_files', {}).items())):
        qualification = validate(root)
    m = load_manifest(root / 'artifacts/execution' / MANIFEST / 'manifest.json')
    plan = make_plan(m)
    if qualification['plan_sha256'] != digest(plan):
        raise ValueError('Verified scheduling plan differs.')
    names = ['production_plan.py', 'submit_production.py', 'report_production.py', 'run_production.sbatch']
    contents = {name: (root / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n') for name in names}
    import json
    contents['production-plan.json'] = (json.dumps(plan, indent=2) + '\n').encode()
    qualification['verified_files'] = {name: hashlib.sha256(body).hexdigest() for name, body in contents.items()}
    contents['verification.json'] = (json.dumps(qualification, indent=2) + '\n').encode()
    packet = {'format_version': 1, 'manifest_hash': MANIFEST, 'code_root': CODE_ROOT,
              'files': {name: hashlib.sha256(body).hexdigest() for name, body in contents.items()}}
    packet_hash = digest(packet)
    contents['packet.json'] = (json.dumps(packet, indent=2) + '\n').encode()
    directory = root / 'artifacts/production' / packet_hash
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in contents.items():
        path = directory / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError('Immutable production packet collision: ' + name)
        path.write_bytes(body)
    destination = root / 'artifacts/production-run.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in contents.items():
            info = zipfile.ZipInfo(name, (2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    transfer = {'packet_hash': packet_hash, 'manifest_hash': MANIFEST, 'default_max_active_gpus': 6,
                'milestone': plan['milestone']['name'], 'selection_sha256': plan['milestone']['selection_sha256'],
                'selected_workers': plan['milestone']['workers'], 'selected_logical_items': plan['milestone']['logical_items'],
                'zip_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'directory': directory.relative_to(root).as_posix()}
    write(root / 'artifacts/production-transfer.json', transfer)
    write(root / 'artifacts/production-verification.json', qualification)
    for name in ['install_production_packet.py', 'start_production.sbatch']:
        (root / 'artifacts' / name).write_bytes((root / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n'))
    print('PRODUCTION PACKET READY:', directory)
    print('First-day selection only; automatic stop; six GPU maximum. No scientific source change, upload or cluster submission.')


if __name__ == '__main__':
    main()
