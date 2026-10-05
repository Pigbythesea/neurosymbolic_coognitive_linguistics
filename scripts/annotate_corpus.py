"""Run full-corpus prefix annotations through the authenticated Codex CLI."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import annotate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/annotation.json")
    parser.add_argument("--codex", help="Codex executable; alternatively set ANNOTATION_CODEX.")
    parser.add_argument("--pass", dest="pass_name", choices=("primary", "blind"),
                        help="Explicit pass selection; otherwise run configured default_passes (primary only).")
    args = parser.parse_args()
    try:
        report = annotate(ROOT, args.config, codex_path=args.codex, pass_name=args.pass_name)
        if report["status"] != "complete":
            print("Available work finished. Deferred stories are listed in progress.json and failures.json.")
            return 2
    except KeyboardInterrupt:
        print("Stopped. Completed units are retained. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print("STOP: " + str(error), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
