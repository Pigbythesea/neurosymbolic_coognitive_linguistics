"""Save a portable deterministic ZIP of the selected, verified annotation export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    directory = (args.input or d.ROOT/d.read_json(d.OUTPUT/'latest.json')['directory']).resolve()
    verification = d.verify_export(directory)
    if not d.read_json(directory/'report.json')['complete']:
        raise ValueError('Portable production bundle requires complete corpus coverage')
    digest = verification['build_hash']
    target = d.OUTPUT/'bundles'/(digest[:20]+'.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    names = sorted([*d.read_json(directory/'checksums.json'), 'checksums.json'])
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            path = (directory/name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('Archive member leaves the frozen package')
            info = zipfile.ZipInfo(digest[:20]+'/'+name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    new_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != new_hash:
            raise ValueError('Existing bundle differs; never overwrite a different archive')
        temporary.unlink()
    else:
        temporary.replace(target)
    report = {'build_hash':digest, 'archive':target.relative_to(d.ROOT).as_posix(),
              'sha256':new_hash, 'bytes':target.stat().st_size, 'members':len(names),
              'complete':True, 'archive_timestamps':'fixed 1980-01-01',
              'frozen_export_verification':verification,
              'semantic_accuracy_measured':False, 'human_review':'pending'}
    d.immutable_json(target.with_suffix('.json'), report)
    d.immutable_json(d.OUTPUT/'export-verifications'/(digest+'.json'), verification)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
