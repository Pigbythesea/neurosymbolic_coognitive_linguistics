"""Audit, package and verify the complete independent annotation review."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review_export as export


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit'); audit.add_argument('--partial',action='store_true')
    sub.add_parser('build')
    verify=sub.add_parser('verify'); verify.add_argument('--directory',required=True)
    args=parser.parse_args()
    if args.command=='audit': result=export.audit(not args.partial)['summary']
    elif args.command=='build': result=export.build()
    else: result=export.verify_export(Path(args.directory))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
