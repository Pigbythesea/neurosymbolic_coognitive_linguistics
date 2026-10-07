"""Run explicit actual-data analyses. No remote connections, installs or fake inputs."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("NEUROSYM_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT))

from neurosym.analysis_data import AnalysisData
from neurosym.analysis_runs import write_report
from neurosym.io import object_hash, read_json
from neurosym.runtime import deadline, YieldRequested


def preflight(root, config_path=None, build=None):
    cfg = read_json(config_path or root / "configs/analysis.json")
    issues, facts, data = [], {}, None
    try:
        data = AnalysisData(root, config_path, build, allow_partial=True)
        facts.update(semantic_build=str(data.semantics.build), partial_annotations=data.partial,
                     coverage=data.semantics.identity["coverage"], parcels=len(data.spatial.names))
        if data.partial:
            issues.append("Annotation snapshot is partial; final fitting requires the completed build.")
        from neurosym.experiment_plan import experiment_plan
        plan = experiment_plan(data)
        facts["experiment_definition_hash"] = plan["definition_hash"]
        if not any(d["group"] == "BC" for d in data.semantics.definitions.values()):
            issues.append("Compiled semantics predate the matched binding control; run the complete local preparation.")
    except (ValueError, FileNotFoundError) as error:
        issues.append(str(error))
    contract = read_json(root / cfg["contract"])
    responses = [s for s in contract["source_files"] if s["path"].startswith("responses/")]
    facts["complete_response_files"] = sum((root / cfg["raw"] / s["path"]).is_file() and
                                           (root / cfg["raw"] / s["path"]).stat().st_size == s["bytes"] for s in responses)
    facts["expected_response_files"] = len(responses)
    if facts["complete_response_files"] != len(responses):
        issues.append("Full measured fMRI files are not all present here. Use the verified cluster data for fitting.")
    extraction = read_json(root / "configs/extraction.json")
    facts["aligned_model_receipts"] = [str(p.parent.parent.name) for p in (root / extraction["output"]).glob("*/aligned/complete.json")]
    facts["models"] = {}
    if data is not None:
        import h5py
        lock = read_json(root / extraction["model_lock"])
        for model in extraction["models"]:
            name, count = model["id"], 0
            try:
                reader = data.model(name)
                source_run = read_json(reader.source / "run.json")
                if source_run["model_id"] != name or source_run["model_lock_hash"] != lock["content_hash"]:
                    raise ValueError("Extraction belongs to another model lock.")
                expected = next(m for m in lock["models"] if m["id"] == name)
                for story in data.semantics.story_ids:
                    timing = read_json(root / extraction["alignment"] / "stories" / (story + ".json"))
                    for aligned in (False, True):
                        path = reader.source / "aligned" / (story + ".h5") if aligned else reader.source / (story + ".h5")
                        with h5py.File(path, "r") as file:
                            run_hash = object_hash(reader.aligned_run) if aligned else reader.aligned_run["source_run_hash"]
                            if not file.attrs["complete"] or file.attrs["run_hash"] != run_hash:
                                raise ValueError("Incomplete/mismatched hidden-state file: " + str(path))
                            for kind in ("words", "units"):
                                shape = file[kind]["features"].shape if aligned else file[kind].shape
                                rows = timing["raw_response_rows"] if aligned else len(timing[kind]["bin_indices"])
                                if shape != (expected["num_hidden_layers"] + 1, rows, expected["hidden_size"]):
                                    raise ValueError("Hidden-state layer/event/coordinate inventory differs from the pinned inputs.")
                            if aligned and (file["response_raw_indices"][()].tolist() != timing["response_raw_indices"] or
                                            file["fir_delays_trs"][()].tolist() != timing["fir_delays_trs"]):
                                raise ValueError("Hidden-state alignment uses another response clock.")
                        count += 1
                facts["models"][name] = {"status": "metadata_verified", "files": count}
            except (FileNotFoundError, OSError, ValueError, KeyError) as error:
                facts["models"][name] = {"status": "unavailable", "files_checked": count, "reason": str(error)}
                issues.append("Frozen model not ready: " + name)
    facts["source_verification"] = "Response presence/byte sizes and model file headers/provenance; does not rehash full arrays or run scientific fits."
    return {"status": "ready" if not issues else "blocked_for_final_execution", "facts": facts, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build", type=Path, help="Explicit immutable semantic build; otherwise current configured output/latest.json")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly label an incomplete-annotation development analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("plan", help="Resolve experiment definitions, pinned model layers and partitions without running fits")
    storage = sub.add_parser('cache-status', help='Read cache budget/reservations and filesystem free space; no deletion')
    storage.add_argument('--scan', action='store_true', help='Also total cache, fit, build and artifact files; may take time')
    numeric = sub.add_parser('verify-compute', help='Verify optimized numerics on real released text features and accepted queries')
    numeric.add_argument('--device', default='cpu')
    jobs = sub.add_parser('jobs', help='Write the complete fit/preparation job manifest without submitting anything')
    jobs.add_argument('--phase', choices=['development', 'final'], default='development')
    jobs.add_argument('--encoding-device', choices=['cpu', 'cuda'], default='cuda')
    jobs.add_argument('--decoder-device', choices=['cpu', 'cuda'], default='cuda')
    worker = sub.add_parser('job', help='Execute one manifest job on an allocated compute node')
    worker.add_argument('manifest', type=Path)
    worker.add_argument('--index', type=int, required=True)
    worker.add_argument('--allow-final', action='store_true')
    grouped = sub.add_parser('worker', help='Execute a related manifest fit panel with shared resident data')
    grouped.add_argument('manifest', type=Path)
    grouped.add_argument('--index', type=int, required=True)
    grouped.add_argument('--allow-final', action='store_true')
    paired = sub.add_parser("compare-encoding", help="Paired effects with identical declared training/test support")
    paired.add_argument("first", type=Path)
    paired.add_argument("second", type=Path)
    paired.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--permutations", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=11)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--maps", action="store_true", help="Common-atlas grounding-map stability instead of RDM association")
    compare.add_argument('--device', default='cpu', help='FP64 matrix products and batched RDM permutations on cpu or cuda')
    panel = sub.add_parser('geometry-panel', help='Run a JSON list of complete geometry options, sharing parent arrays')
    panel.add_argument('options', type=Path)
    for command in ("encoding", "decoder", "geometry", 'prepare-decoder'):
        run = sub.add_parser(command)
        run.add_argument("--fold", required=True, help="0..9, final, or an explicit multistory holdout such as stories:story_01,story_02")
        run.add_argument("--subject")
        run.add_argument("--model")
        run.add_argument("--layer", type=int)
        run.add_argument("--seed", type=int, default=11)
        if command in {"decoder", "geometry", 'prepare-decoder'}:
            run.add_argument("--modality", choices=["brain", "model"], required=True)
            run.add_argument("--reviewed-only", action="store_true")
        if command == "encoding":
            run.add_argument('--device', default='cpu', help='cpu or cuda[:index]; encoding remains float64')
            run.add_argument("--groups", nargs="+", required=True)
            support = run.add_mutually_exclusive_group(required=True)
            support.add_argument("--comparison", help="Named comparison in config: same value for baseline and augmented runs")
            support.add_argument("--mask-groups", nargs="+", help="Explicit UNION of feature groups across all compared conditions")
            run.add_argument("--mask-policy", choices=["comparison", "all-groups"], default="comparison")
            run.add_argument("--mask-model", action="append", dest="mask_models", metavar="MODEL_ID:LAYER",
                             help="Repeat for all compared frozen states, identically on baseline and model conditions")
        if command == "decoder":
            run.add_argument("--family", choices=["prior", "linear", "mlp", "structured"], required=True)
            run.add_argument("--device", default="cpu")
            run.add_argument('--require-prepared', action='store_true', help='Require completed observation preparation')
            run.add_argument("--retrain-null", action="store_true")
            run.add_argument("--composition-keys", type=Path, help="JSON list of preselected exact configuration hashes; whole training stories are purged")
        if command == "geometry":
            run.add_argument('--device', default='cpu', help='FP64 RDM products and compact encoding reconstruction on cpu or cuda')
            run.add_argument("--view", choices=["native", "grounding", "latent", "latent-passage", "encoding-implied"], required=True,
                             help="latent uses occurrence-matched queries; latent-passage averages all source-unit queries")
            run.add_argument("--kind", choices=["concept", "predicate", "role", "configuration", "discourse", "reference", "state_update", "literal", "identity", "scope", "qualification"], required=True)
            run.add_argument("--from-run", type=Path)
            run.add_argument("--component", help="Encoding feature-group contribution, otherwise total predicted response")
            run.add_argument("--min-stories", type=int, default=1, help="Per-fold test sets contain one story; cross-context reliability is reported unavailable there")
            run.add_argument("--scope-mode", choices=["pooled", "scoped"], default="pooled", help="Pool with saved scope provenance, or require identical symbolic scope per semantic item")
            run.add_argument("--residualize-presentation", action="store_true", help="Native geometry: training-fitted presentation-feature residuals, saved separately from raw native geometry")
    args = parser.parse_args()
    if args.command == 'cache-status':
        from neurosym.storage import storage_report
        print(json.dumps(storage_report(ROOT, read_json(ROOT / 'configs/compute.json')['storage'], scan=args.scan), indent=2))
        return
    if args.command == 'jobs':
        from neurosym.execution import write_manifest
        path = write_manifest(AnalysisData(ROOT, args.config, args.build), phase=args.phase,
                              encoding_device=args.encoding_device, decoder_device=args.decoder_device)
        print('EXECUTION MANIFEST (NOT SUBMITTED):', path)
        return
    if args.command == "plan":
        from neurosym.experiment_plan import experiment_plan
        report = experiment_plan(AnalysisData(ROOT, args.config, args.build))
        write_report(ROOT / "artifacts/experiment-plan.json", report)
        print("EXPERIMENT DEFINITIONS RESOLVED:", report["definition_hash"])
        return
    if args.command == "preflight":
        report = preflight(ROOT, args.config, args.build)
        write_report(ROOT / "artifacts/analysis-preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ready" else 1
    if os.name != "nt" and (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith("login")):
        raise RuntimeError("Cluster processing requires an allocated compute node, never a login node.")
    if args.command == 'verify-compute':
        command = [sys.executable, '-I', '-B', '-u', str(Path(__file__).with_name('verify_compute.py')), '--device', args.device]
        if args.build:
            command += ['--build', str(args.build)]
        if args.config:
            raise ValueError('Compute verification uses the pinned configs/analysis.json.')
        os.execv(sys.executable, command)
    if args.command in {'job', 'worker'}:
        from neurosym.execution import run_job, run_worker
        path = (run_job if args.command == 'job' else run_worker)(ROOT, args.manifest, args.index, allow_final=args.allow_final)
        print('JOB COMPLETE:', path)
        return
    if args.command == "compare":
        from neurosym.geometry import compare_geometry, grounding_stability
        report = (grounding_stability if args.maps else compare_geometry)(args.first, args.second, permutations=args.permutations, seed=args.seed, device=args.device)
        write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return
    if args.command == "compare-encoding":
        from neurosym.encoding import compare_encoding
        report = compare_encoding(args.first, args.second)
        write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return
    if args.command == 'geometry-panel':
        from neurosym.geometry import run_geometry_panel
        paths = run_geometry_panel(AnalysisData(ROOT, args.config, args.build), read_json(args.options))
        print('GEOMETRY PANEL COMPLETE:', *paths, sep='\n')
        return
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in {"command", "config", "build", "allow_partial"}}
    if args.command == "decoder" and args.composition_keys:
        options["composition_keys"] = read_json(args.composition_keys)
        if not isinstance(options["composition_keys"], list):
            parser.error("Composition keys file must contain a JSON list.")
    if args.command == "encoding" and not args.subject:
        parser.error("Encoding needs --subject.")
    if args.command in {"decoder", "geometry", 'prepare-decoder'}:
        if args.modality == "brain" and not args.subject:
            parser.error("Brain observations need --subject.")
        if args.modality == "model" and (not args.model or args.layer is None or args.subject):
            parser.error("Model observations need --model and --layer and no --subject.")
    if args.command == "geometry" and args.view != "native" and args.from_run is None:
        parser.error("Derived geometries need the completed --from-run directory.")
    if args.command == "geometry" and args.residualize_presentation and args.view != "native":
        parser.error("Presentation residualization is supported for native observations only.")
    data = AnalysisData(ROOT, args.config, args.build, allow_partial=args.allow_partial,
                        require_prepared=getattr(args, 'require_prepared', False))
    if args.command == "encoding":
        from neurosym.encoding import run_encoding
        path = run_encoding(data, options)
    elif args.command == "decoder":
        from neurosym.decoder_fit import run_decoder
        path = run_decoder(data, options)
    elif args.command == 'prepare-decoder':
        from neurosym.decoder_fit import prepare_decoder
        path = prepare_decoder(data, options)
    else:
        from neurosym.geometry import run_geometry
        path = run_geometry(data, options)
    print("ANALYSIS COMPLETE:", path)


if __name__ == "__main__":
    deadline.install()
    try:
        sys.exit(main())
    except YieldRequested as error:
        print('ANALYSIS YIELDED:', error, flush=True)
        sys.exit(75)
