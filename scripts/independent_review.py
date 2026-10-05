"""CMD entry point for the independent source-first semantic review."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurosym import independent_review as r

p=argparse.ArgumentParser()
p.add_argument('command',choices=['init','status','register','next','interpret','commit','ledger'])
p.add_argument('--story')
p.add_argument('--actor')
p.add_argument('--input')
args=p.parse_args()
if args.command=='init': result=r.initialize()
elif args.command=='status': result=r.status()
elif args.command=='register': result=r.register(args.story,args.actor,'Fresh independent story context; no prior target-story source or annotations; only generic instructions and schema before registration.')
elif args.command=='next': result=r.next_source(args.story)
elif args.command=='interpret': result=r.interpret(args.story,r.read(args.input))
elif args.command=='commit': result=r.commit(args.story,r.read(args.input))
elif args.command=='ledger': result=r.ledger(args.story)
print(json.dumps(result,ensure_ascii=True,separators=(',',':')))
