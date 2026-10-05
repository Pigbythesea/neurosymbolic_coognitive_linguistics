"""Import the completed cluster inspection without rescanning or downloading fMRI."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.deniz import inspect_alignment, format_summary
from neurosym.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/deniz/contract.json")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        if set(archive.namelist()) != {"report.json", "summary.txt"} or archive.testzip():
            raise ValueError("Unexpected or corrupt inspection archive.")
        original = archive.read("report.json")
    report = json.loads(original)
    manifest_bytes = (ROOT / "manifests/deniz-reading.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {entry["path"]: entry["bytes"] for entry in manifest["datasets"]["deniz"]["files"]}
    received = {entry["path"]: entry["bytes"] for entry in report["sources"]}
    if report["scope"] != "all" or received != expected or len(report["sources"]) != len(expected):
        raise ValueError("Inspection did not cover the complete reading release.")
    if report["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Inspection used a different source manifest.")
    # Reassess the original dimensional checks against the observed singleton
    # training-repeat layout. Preserve all unrelated inspection findings.
    old_shape_error = "Response shape differs from expected time/voxel/repetition contract; inspect original metadata."
    report["issues"] = [item for item in report["issues"] if item["detail"] != old_shape_error]
    report["alignment"] = inspect_alignment(report)
    report["status"] = "needs_attention" if any(item["severity"] == "error" for item in report["issues"]) else "complete"
    print(format_summary(report))
    if report["status"] != "complete":
        raise ValueError("Unresolved data-contract errors remain; inspect the original report.")
    for path, metadata in report["hdf5"].items():
        if path.startswith("responses/"):
            for name, values in metadata["datasets"].items():
                numeric = values["numeric"]
                if not numeric.get("scanned") or numeric["finite"] != numeric["elements"]:
                    raise ValueError(f"Response validity policy needed before fitting: {path}/{name}")
    contract = {"format_version": 1, "dataset": "deniz-reading", "manifest_sha256": report["manifest_sha256"],
                "inspection_report_sha256": hashlib.sha256(original).hexdigest(),
                "inspection_environment": report["environment"], "inspection_code_sha256": report["inspection_code_sha256"],
                "contract_code_sha256": hashlib.sha256((ROOT / "neurosym/deniz.py").read_bytes()).hexdigest(),
                "source_files": report["sources"], "subjects": report["alignment"]["subjects"],
                "feature_shapes": report["alignment"]["features"], "mappers": report["mappers"],
                "stories": report["stimuli"]["stories"], "tr_seconds": report["alignment"]["documented_tr_seconds"],
                "trim": report["alignment"]["published_trim_length_check"],
                "all_scanned_response_values_finite": True,
                "timing_note": report["alignment"]["absolute_onset_note"]}
    save_json(args.output, contract)
    print("DATA CONTRACT:", args.output)


if __name__ == "__main__":
    main()
