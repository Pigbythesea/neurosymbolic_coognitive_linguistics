"""Extract a verified administrative packet on allocated compute only."""
import hashlib
import json
import os
from pathlib import Path
import socket
import zipfile


def main():
    root = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login') or Path.cwd().resolve() != root:
        raise RuntimeError('Install requires allocated compute in the project directory.')
    transfer = json.loads((root / 'artifacts/production-transfer.json').read_text())
    source = root / 'artifacts/production-run.zip'
    if hashlib.sha256(source.read_bytes()).hexdigest() != transfer['zip_sha256']:
        raise ValueError('Transferred production archive checksum mismatch.')
    destination = (root / transfer['directory']).resolve()
    if destination.parent != root / 'artifacts/production' or destination.name != transfer['packet_hash']:
        raise ValueError('Invalid packet destination.')
    with zipfile.ZipFile(source) as archive:
        packet = json.loads(archive.read('packet.json'))
        key = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if key != transfer['packet_hash'] or packet['manifest_hash'] != transfer['manifest_hash']:
            raise ValueError('Packet identity mismatch.')
        if set(archive.namelist()) != set(packet['files']) | {'packet.json'}:
            raise ValueError('Unexpected archive members.')
        bodies = {}
        for name in archive.namelist():
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Archive path is not a flat packet filename.')
            body = archive.read(name)
            if name != 'packet.json' and hashlib.sha256(body).hexdigest() != packet['files'][name]:
                raise ValueError('Packet member checksum mismatch.')
            path = destination / name
            if path.exists() and path.read_bytes() != body:
                raise ValueError('Existing immutable packet differs.')
            bodies[name] = body
        destination.mkdir(parents=True, exist_ok=True)
        for name, body in bodies.items():
            (destination / name).write_bytes(body)
    print(destination)


if __name__ == '__main__':
    main()
