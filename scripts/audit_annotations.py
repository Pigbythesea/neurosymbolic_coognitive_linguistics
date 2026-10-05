"""Audit completed real annotations and prepare source-grounded human review records."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.audit import annotate_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/annotation.json")
    args = parser.parse_args()
    annotate_audit(ROOT, args.config)
