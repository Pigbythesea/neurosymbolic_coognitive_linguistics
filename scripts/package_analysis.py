"""Package isolated analysis code plus a COMPLETE real semantic/spatial snapshot."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.analysis_data import AnalysisData
from neurosym.io import object_hash, read_json

MODULES = ["__init__", "io", "dataset", "extraction_inputs", "model_registry", "temporal", "extraction",
           "model_features", "graphs", "semantics", "semantic_queries", "semantic_records", "semantic_features",
           "analysis_data", "analysis_runs", "spatial", "decoders", "decoder_fit", "decoder_batch", "decoder_minibatch", "protocol", "study_jobs", "study_reports", "grounding_checks", "encoding", "encoding_support", "geometry",
           "reviewed_archive", "reviewed_graph", "reviewed_queries", "reviewed_compile", "experiment_plan", "compute", "storage", "execution", "pca", "runtime",
           "trace_store", "trace_geometry", "decoder_stages", "fit_reuse"]
EXECUTION_SCRIPTS = ['scripts/run_analysis.py', 'scripts/verify_compute.py', 'scripts/submit_analysis.py',
                     'scripts/run_manifest_array.py', 'scripts/run_analysis_array.sbatch',
                     'scripts/verify_trace_tables.py', 'scripts/adopt_protocol2_fits.py', 'scripts/cleanup_retired_decoder_tests.py']


def main():
    # No fallback to old annotations, partial catalogs, missing data or fabricated arrays.
    data = AnalysisData(ROOT)
    verification = read_json(ROOT / "artifacts/analysis-verification.json")
    if verification["status"] != "verified" or verification["semantic_build_hash"] != data.semantics.build_hash:
        raise ValueError("Verify the FINAL compiled annotation build before packaging; the old snapshot's receipt is insufficient.")
    if verification.get("verified_code_sha256", {}).get("scripts/verify_analysis.py") != hashlib.sha256((ROOT / "scripts/verify_analysis.py").read_bytes()).hexdigest():
        raise ValueError("Analysis verification procedure changed since its receipt.")
    support = read_json(ROOT / "artifacts/encoding-support-verification.json")
    if (support["status"] != "verified" or support["semantic_build_hash"] != data.semantics.build_hash or
            support.get("config_hash") != object_hash(data.config) or verification.get("config_hash") != object_hash(data.config) or
            support.get("verification_script_sha256") != hashlib.sha256((ROOT / "scripts/verify_encoding_support.py").read_bytes()).hexdigest()):
        raise ValueError("Verify comparison support on the complete real corpus with the current analysis configuration.")
    reviewed = read_json(ROOT / "artifacts/reviewed-semantics-verification.json")
    from neurosym.experiment_plan import experiment_plan
    plan = experiment_plan(data)
    if (verification.get("experiment_definition_hash") != plan["definition_hash"] or
            read_json(ROOT / "artifacts/experiment-plan.json") != plan):
        raise ValueError("Experiment definitions changed since verification; resolve and verify the current plan.")
    if reviewed["status"] != "verified" or reviewed["semantic_build_hash"] != data.semantics.build_hash:
        raise ValueError("Full accepted-corpus scope verification is required before packaging.")
    if reviewed.get("verification_script_sha256") != hashlib.sha256((ROOT / "scripts/verify_reviewed_semantics.py").read_bytes()).hexdigest():
        raise ValueError("Reviewed-corpus verification procedure changed since its receipt.")
    from neurosym.execution import code_identity, execution_manifest
    traces = read_json(ROOT / 'artifacts/trace-verification-cpu.json')
    if (traces.get('status') != 'verified' or not traces.get('complete_samples') or
            not traces.get('geometry_verified') or traces.get('code') != code_identity()):
        raise ValueError('Verify compact traces and exact geometry aggregation against the real fitted audit before packaging.')
    compute = read_json(ROOT / 'artifacts/compute-verification-cpu.json')
    jobs = execution_manifest(data)
    if (compute.get('status') != 'verified' or compute.get('device') != 'cpu' or
            compute.get('semantic_build_hash') != data.semantics.build_hash or compute.get('code') != code_identity() or
            compute.get('config_hash') != object_hash(data.config) or
            compute.get('compute_config_hash') != object_hash(jobs['resources']) or
            compute.get('execution_manifest_hash') != jobs['content_hash'] or
            compute.get('verification_script_sha256') != hashlib.sha256((ROOT / 'scripts/verify_compute.py').read_bytes()).hexdigest()):
        raise ValueError('Verify compute equivalence and the job inventory on the current complete build before packaging.')
    jobs_path = ROOT / 'artifacts/execution' / jobs['content_hash'] / 'manifest.json'
    if read_json(jobs_path) != jobs:
        raise ValueError('Packaged execution manifest differs from verified definitions.')
    code = {f"neurosym/{name}.py": (ROOT / "neurosym" / (name + ".py")).read_bytes() for name in MODULES}
    code.update({name: (ROOT / name).read_bytes() for name in EXECUTION_SCRIPTS})
    for name, body in code.items():
        if name.endswith('.py'):
            ast.parse(body, filename=name, feature_version=(3, 11))
        if verification.get("verified_code_sha256", {}).get(name) != hashlib.sha256(body).hexdigest():
            raise ValueError("Analysis source changed since real-data verification: " + name)
    code_hash = object_hash({k: hashlib.sha256(v).hexdigest() for k, v in code.items()})
    code_root = "analysis_code/" + code_hash
    contents = {code_root + "/" + k: v for k, v in code.items()}
    shared = ["configs/analysis.json", "configs/extraction.json", "configs/alignment.json", "configs/spatial.json",
              "requirements/analysis.txt", "scripts/run_analysis.sbatch", "docs/analysis.md", "docs/analysis_environment.md",
              "data/processed/deniz/contract.json", "artifacts/analysis-verification.json",
              "artifacts/reviewed-semantics-verification.json", "artifacts/encoding-support-verification.json",
              "configs/semantics.json", "configs/experiments.json", "artifacts/experiment-plan.json",
              "manifests/frozen-models.lock.json", "docs/experiment_definitions.md",
              "docs/reviewed_downstream.md", "docs/encoding_support.md", 'docs/compute.md', 'configs/compute.json',
               'artifacts/compute-verification-cpu.json', 'artifacts/execution/latest.json', 'docs/CLUSTER_STATUS.md',
              'docs/ENGINEERING_HANDOFF.md', 'docs/SCIENTIFIC_STATUS_HANDOFF.md', 'artifacts/trace-verification-cpu.json',
              'scripts/qualify_trace.sbatch', 'scripts/maintain_qualified_fits.sbatch',
              jobs_path.relative_to(ROOT).as_posix()]
    for name in shared:
        contents[name] = (ROOT / name).read_bytes()
    folders = [data.semantics.build, data.spatial.path, ROOT / "data/atlases/schaefer200",
               ROOT / "data/processed/corpus", ROOT / "data/processed/alignment"]
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file():
                contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for folder in [ROOT / data.config["semantics"], ROOT / data.config["spatial"]]:
        contents[(folder / "latest.json").relative_to(ROOT).as_posix()] = (folder / "latest.json").read_bytes()
    for name, body in contents.items():
        if name.endswith(".sbatch") and b"\r\n" in body:
            raise ValueError("Slurm script must use LF newlines: " + name)
    manifest = {"format_version": 1, "code_root": code_root, "semantic_build_hash": data.semantics.build_hash,
                "files": {name: hashlib.sha256(body).hexdigest() for name, body in sorted(contents.items())}}
    destination = ROOT / "artifacts/analysis-source.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as file:
        for name, body in contents.items():
            file.writestr(name, body)
        file.writestr("BUNDLE.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as file:
        if file.testzip() is not None or any(hashlib.sha256(file.read(n)).hexdigest() != d for n, d in manifest["files"].items()):
            raise ValueError("Analysis bundle CRC/content verification failed.")
    for name in ("install_analysis_bundle.py", "setup_analysis.sbatch", "update_analysis.sbatch"):
        shutil.copyfile(ROOT / "scripts" / name, ROOT / "artifacts" / name)
    transfer_files = ['artifacts/analysis-source.zip', 'artifacts/install_analysis_bundle.py',
                      'artifacts/update_analysis.sbatch', 'scripts/cluster_env.sh']
    transfer = {'format_version': 1, 'semantic_build_hash': data.semantics.build_hash, 'code_root': code_root,
                'manifest_hash': jobs['content_hash'],
                'files': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in transfer_files}}
    (ROOT / 'artifacts/analysis-transfer.json').write_text(json.dumps(transfer, indent=2) + '\n', encoding='utf-8')
    print("ANALYSIS BUNDLE:", len(contents), "files;", destination.stat().st_size, "bytes")
    print("SHA256:", hashlib.sha256(destination.read_bytes()).hexdigest())
    print("Analysis code is isolated by content hash; no annotation runner, prompts, keys or model weights are included.")


if __name__ == "__main__":
    main()
