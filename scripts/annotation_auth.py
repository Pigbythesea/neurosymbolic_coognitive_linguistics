"""Inspect or establish the same ChatGPT CLI login used by annotation. No inference."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import child_environment, resolve_codex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "login"))
    parser.add_argument("--codex", help="Optional explicit CLI executable.")
    args = parser.parse_args()
    codex = resolve_codex(args.codex)
    print("Annotation CLI:", codex, flush=True)
    command = [str(codex), "login"]
    if args.action == "status":
        command.append("status")
    return subprocess.run(command, env=child_environment()).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
