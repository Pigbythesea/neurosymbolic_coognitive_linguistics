"""Build actual story alignment and nested story-level split manifests."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.temporal import build_alignment

if __name__ == "__main__":
    build_alignment(ROOT, ROOT / "configs/alignment.json")
