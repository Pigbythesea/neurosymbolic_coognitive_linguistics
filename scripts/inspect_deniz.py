"""Inspect the verified Deniz release; run full response inspection on a compute node."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurosym.deniz import inspect_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/deniz-reading.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/deniz-inspection")
    parser.add_argument("--scope", choices=("all", "support"), default="all",
                        help="support inspects all non-response files, for local data-reader development")
    args = parser.parse_args()
    try:
        report = inspect_dataset(args.manifest.resolve(), args.data_root.resolve(), args.output.resolve(), args.scope)
    except (OSError, ValueError, KeyError) as error:
        print("STOP: " + str(error), file=sys.stderr, flush=True)
        return 1
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
