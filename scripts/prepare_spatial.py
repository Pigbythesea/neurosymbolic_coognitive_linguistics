"""Acquire the pinned atlas and prepare all real Deniz native-voxel parcels."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.spatial import prepare_spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spatial.json")
    parser.add_argument("--download", action="store_true", help="Download two small atlas label files from the pinned official repository.")
    args = parser.parse_args()
    output = prepare_spatial(ROOT, args.config, download=args.download)
    print("SPATIAL PREPARATION COMPLETE:", output)


if __name__ == "__main__":
    main()
