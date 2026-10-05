"""Check actual analysis dependencies; this is not a scientific validation run."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    versions, problems = {}, []
    requirements = ROOT / "requirements" / "analysis.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            module = importlib.import_module(name)
            actual = importlib.metadata.version(name)
            versions[name] = actual
            if actual.split("+")[0] != expected:
                problems.append(f"{name}: expected {expected}, found {actual}")
            if name == "torch" and sys.platform == "win32" and module.version.cuda is not None:
                problems.append("Expected CPU PyTorch in the local analysis environment.")
        except (ImportError, OSError, RuntimeError) as error:
            problems.append(f"{name}: {type(error).__name__}: {error}")
    if sys.version_info < (3, 11):
        problems.append("Analysis requires Python 3.11 or later.")
    if sys.platform == "win32" and Path(sys.prefix).resolve() != (ROOT / ".venv-analysis").resolve():
        problems.append("Run this check with .venv-analysis, separate from the running annotator.")
    report = {"status": "needs_setup" if problems else "ready", "python": sys.version,
              "executable": sys.executable, "versions": versions, "problems": problems,
              "scope": "Dependency imports and versions only; no scientific fits or synthetic data.",
              "scientific_validation_executed": False}
    destination = ROOT / "artifacts" / "analysis-environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(report, indent=2), flush=True)
    print("REPORT:", destination, flush=True)
    if problems:
        return 2
    print("ANALYSIS ENVIRONMENT READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
