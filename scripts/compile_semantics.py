"""Compile a real saved annotation snapshot without contacting an LLM or the cluster."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.io import read_json
from neurosym.semantics import compile_semantics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/semantics.json")
    parser.add_argument("--require-complete", action="store_true", help="Return status 2 when actual annotation coverage is incomplete.")
    args = parser.parse_args()
    build = compile_semantics(ROOT, args.config)
    report = read_json(build / "report.json")
    print("SEMANTIC BUILD:", build)
    print("REPORT:", build / "report.html")
    print({"compiled_units": report["counts"]["units"], "expected_units": report["expected_units"],
           "corpus_coverage_complete": report["corpus_coverage_complete"], "llm_calls": 0,
           "semantic_accuracy_measured": False})
    return 2 if args.require_complete and not report["corpus_coverage_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
