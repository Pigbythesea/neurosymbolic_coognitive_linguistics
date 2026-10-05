"""Run explicit actual-data analyses. No remote connections, installs or fake inputs."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("NEUROSYM_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT))

from neurosym.analysis_data import AnalysisData
from neurosym.analysis_runs import write_report
from neurosym.io import read_json


def preflight(root, config_path=None, build=None):
    cfg = read_json(config_path or root / "configs/analysis.json")
    issues, facts = [], {}
    try:
        data = AnalysisData(root, config_path, build, allow_partial=True)
        facts.update(semantic_build=str(data.semantics.build), partial_annotations=data.partial,
                     coverage=data.semantics.identity["coverage"], parcels=len(data.spatial.names))
        if data.partial:
            issues.append("Annotation snapshot is partial; final fitting requires the completed build.")
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
    facts["source_verification"] = "Response presence/byte sizes and available receipts; does not rehash 25 GB or run scientific fits."
    return {"status": "ready" if not issues else "blocked_for_final_execution", "facts": facts, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build", type=Path, help="Explicit immutable semantic build; otherwise current configured output/latest.json")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly label an incomplete-annotation development analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    compare = sub.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--permutations", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=11)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--maps", action="store_true", help="Common-atlas grounding-map stability instead of RDM association")
    for command in ("encoding", "decoder", "geometry"):
        run = sub.add_parser(command)
        run.add_argument("--fold", required=True, help="0..9, final, or an explicit multistory holdout such as stories:story_01,story_02")
        run.add_argument("--subject")
        run.add_argument("--model")
        run.add_argument("--layer", type=int)
        run.add_argument("--seed", type=int, default=11)
        if command in {"decoder", "geometry"}:
            run.add_argument("--modality", choices=["brain", "model"], required=True)
            run.add_argument("--reviewed-only", action="store_true")
        if command == "encoding":
            run.add_argument("--groups", nargs="+", required=True)
        if command == "decoder":
            run.add_argument("--family", choices=["prior", "linear", "mlp", "structured"], required=True)
            run.add_argument("--device", default="cpu")
            run.add_argument("--retrain-null", action="store_true")
            run.add_argument("--composition-keys", type=Path, help="JSON list of preselected exact configuration hashes; whole training stories are purged")
        if command == "geometry":
            run.add_argument("--view", choices=["native", "grounding", "latent", "encoding-implied"], required=True)
            run.add_argument("--kind", choices=["concept", "predicate", "role", "configuration", "discourse", "reference", "state_update", "literal", "identity", "scope", "qualification"], required=True)
            run.add_argument("--from-run", type=Path)
            run.add_argument("--component", help="Encoding feature-group contribution, otherwise total predicted response")
            run.add_argument("--min-stories", type=int, default=1, help="Per-fold test sets contain one story; cross-context reliability is reported unavailable there")
            run.add_argument("--scope-mode", choices=["pooled", "scoped"], default="pooled", help="Pool with saved scope provenance, or require identical symbolic scope per semantic item")
            run.add_argument("--residualize-presentation", action="store_true", help="Native geometry: training-fitted presentation-feature residuals, saved separately from raw native geometry")
    args = parser.parse_args()
    if args.command == "preflight":
        report = preflight(ROOT, args.config, args.build)
        write_report(ROOT / "artifacts/analysis-preflight.json", report)
        print(json.dumps(report, indent=2))
        return
    if args.command == "compare":
        from neurosym.geometry import compare_geometry, grounding_stability
        report = (grounding_stability if args.maps else compare_geometry)(args.first, args.second, permutations=args.permutations, seed=args.seed)
        write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return
    if os.name != "nt" and (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith("login")):
        raise RuntimeError("Cluster fits must run on an allocated compute node, never a login node.")
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in {"command", "config", "build", "allow_partial"}}
    if args.command == "decoder" and args.composition_keys:
        options["composition_keys"] = read_json(args.composition_keys)
        if not isinstance(options["composition_keys"], list):
            parser.error("Composition keys file must contain a JSON list.")
    if args.command == "encoding" and not args.subject:
        parser.error("Encoding needs --subject.")
    if args.command in {"decoder", "geometry"}:
        if args.modality == "brain" and not args.subject:
            parser.error("Brain observations need --subject.")
        if args.modality == "model" and (not args.model or args.layer is None or args.subject):
            parser.error("Model observations need --model and --layer and no --subject.")
    if args.command == "geometry" and args.view != "native" and args.from_run is None:
        parser.error("Derived geometries need the completed --from-run directory.")
    if args.command == "geometry" and args.residualize_presentation and args.view != "native":
        parser.error("Presentation residualization is supported for native observations only.")
    data = AnalysisData(ROOT, args.config, args.build, allow_partial=args.allow_partial)
    if args.command == "encoding":
        from neurosym.encoding import run_encoding
        path = run_encoding(data, options)
    elif args.command == "decoder":
        from neurosym.decoder_fit import run_decoder
        path = run_decoder(data, options)
    else:
        from neurosym.geometry import run_geometry
        path = run_geometry(data, options)
    print("ANALYSIS COMPLETE:", path)


if __name__ == "__main__":
    main()
