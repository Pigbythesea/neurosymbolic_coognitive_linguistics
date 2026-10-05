"""Local direct-authoring checkpoint and export operations (no model invocation)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as direct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("init", "register", "next", "ledger", "status", "export", "verify", "verify-export", "accept"))
    p.add_argument("--story")
    p.add_argument("--actor")
    p.add_argument("--input", type=Path)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    if a.operation=="init": result=direct.initialize()
    elif a.operation=="register": result=direct.register(a.story,a.actor)
    elif a.operation=="next": result=direct.next_packet(a.story,a.actor)
    elif a.operation=="ledger": result=direct.available_ledger(a.story)
    elif a.operation=="status": result=direct.status()
    elif a.operation=="export": result=direct.export_package(a.require_complete)
    elif a.operation=="verify": result=direct.verify()
    elif a.operation=="verify-export": result=direct.verify_export(a.input)
    else: result=direct.accept(json.loads(a.input.read_text(encoding="utf-8")),a.actor,a.input)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=="__main__":
    main()
