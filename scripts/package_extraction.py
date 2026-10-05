"""Package this workstream and real prepared inputs, excluding all annotation-owned code."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'neurosym/__init__.py', 'neurosym/io.py', 'neurosym/dataset.py',
    'neurosym/extraction_inputs.py', 'neurosym/model_registry.py', 'neurosym/temporal.py',
    'neurosym/tokenization.py', 'neurosym/extraction.py', 'neurosym/model_features.py',
    'configs/extraction.json', 'configs/alignment.json', 'manifests/frozen-models.lock.json',
    'requirements/extraction-metadata.txt', 'requirements/extraction.txt',
    'scripts/resolve_extraction_models.py', 'scripts/download_extraction_models.py',
    'scripts/prepare_alignment.py', 'scripts/prepare_tokenization.py', 'scripts/extract_model_states.py',
    'scripts/align_model_features.py', 'scripts/validate_extraction.py', 'scripts/install_extraction_bundle.py',
    'scripts/setup_extraction.sbatch', 'scripts/extract_models.sbatch',
    'docs/extraction.md', 'data/processed/deniz/contract.json',
]


def main():
    files = sorted(set(FILES) | {p.relative_to(ROOT).as_posix()
                               for folder in ('data/processed/corpus', 'data/processed/alignment')
                               for p in (ROOT / folder).rglob('*.json')})
    contents = {name: (ROOT / name).read_bytes() for name in files}
    for name, data in contents.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name, feature_version=(3, 11))
        if name.endswith('.sbatch') and b'\r\n' in data:
            raise ValueError('Slurm scripts must use LF newlines: ' + name)
    manifest = {'format_version': 1, 'files': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
    destination = ROOT / 'artifacts/extraction-source.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
        bundle.writestr('BUNDLE.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as bundle:
        if bundle.testzip() is not None or any(bundle.read(name) != data for name, data in contents.items()):
            raise ValueError('Actual extraction bundle failed byte/CRC verification.')
    for name in ('install_extraction_bundle.py', 'setup_extraction.sbatch'):
        shutil.copyfile(ROOT / 'scripts' / name, ROOT / 'artifacts' / name)
    print('EXTRACTION BUNDLE:', len(files), 'files;', destination.stat().st_size, 'bytes')
    print('SHA256:', hashlib.sha256(destination.read_bytes()).hexdigest())
    print('Annotation code, configuration, prompts and schema are excluded.')


if __name__ == '__main__':
    main()
