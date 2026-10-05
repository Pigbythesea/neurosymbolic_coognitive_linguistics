"""Package a verified immutable annotation review for portable retention."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d
from neurosym import independent_review as r
from neurosym import independent_review_export as e


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory')
    args=parser.parse_args()
    directory=Path(args.directory) if args.directory else d.ROOT/r.read(r.OUT/'latest.json')['directory']
    verification=e.verify_export(directory)
    build=r.read(directory/'snapshot.json')['build_hash']
    memory=io.BytesIO()
    paths=sorted(p for p in directory.rglob('*') if p.is_file())
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in paths:
            name=build[:20]+'/'+path.relative_to(directory).as_posix()
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3
            info.external_attr=0o100644<<16
            archive.writestr(info,path.read_bytes(),compresslevel=9)
    payload=memory.getvalue()
    # Check every archived byte against its immutable input; do not extract.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC mismatch')
        for path in paths:
            name=build[:20]+'/'+path.relative_to(directory).as_posix()
            if archive.read(name)!=path.read_bytes(): raise ValueError('Archive payload differs: '+name)
    destination=r.OUT/'bundles'/(build[:20]+'.zip')
    d.immutable_bytes(destination,payload)
    receipt={'build_hash':build,'zip':destination.relative_to(d.ROOT).as_posix(),
             'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'files':len(paths),
             'all_archive_bytes_verified':True,'export_verification':verification}
    d.immutable_bytes(destination.with_suffix('.receipt.json'),e.json_bytes(receipt))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
