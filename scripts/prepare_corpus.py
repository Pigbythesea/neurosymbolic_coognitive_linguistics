"""Prepare every released reading story for annotation from its real text and timing."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.corpus import build_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/deniz")
    parser.add_argument("--contract", type=Path, default=ROOT / "data/processed/deniz/contract.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/corpus")
    args = parser.parse_args()
    build_corpus(args.raw_root, args.contract, args.output)
