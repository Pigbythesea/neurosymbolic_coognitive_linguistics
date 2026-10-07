"""Import a collected real-data audit, checking paths, sizes and SHA256s."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'artifacts'
    source = args.archive.resolve(strict=True)
    if source.parent != root or not source.name.startswith('trace-audit-'):
        raise ValueError('Expected a trace-audit ZIP directly in artifacts.')
    destination = source.with_suffix('')
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        checks = json.loads(archive.read('checksums.json'))
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(checks) | {'checksums.json'}:
            raise ValueError('Archive contents differ from the checksum inventory.')
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path: ' + name)
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive target escaped its directory.')
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            temporary = path.with_name(path.name + '.importing')
            with archive.open(name) as incoming, temporary.open('wb') as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    size += len(block); digest.update(block); outgoing.write(block)
            if name in checks and checks[name] != {'bytes': size, 'sha256': digest.hexdigest()}:
                temporary.unlink()
                raise ValueError('Checksum mismatch: ' + name)
            temporary.replace(path)
    print('VERIFIED AUDIT:', destination)


if __name__ == '__main__':
    main()
