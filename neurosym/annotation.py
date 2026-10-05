"""Resumable, prefix-only Codex annotation using the user's existing ChatGPT login."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .graphs import GraphDelta, GraphHistory, graph_schema
from .annotation_repair import normalize_graph
from .io import immutable_json, object_hash, read_json, save_json


DISABLED_FEATURES = ("shell_tool", "unified_exec", "apps", "plugins", "browser_use", "browser_use_external",
                     "computer_use", "in_app_browser", "code_mode_host", "multi_agent", "multi_agent_v2",
                     "image_generation", "view_image", "memories", "hooks", "skill_search", "goals", "fast_mode")

# Codex 0.160.0 reports this as an error item before turn.started even though
# disabling the host is intentional and generation can complete successfully.
DISABLED_CODE_MODE_DIAGNOSTIC = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; "
    "enable `features.code_mode_host` and install `codex-code-mode-host`."
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_run(output: Path, identity: dict) -> dict:
    """Preserve run provenance across explicitly recorded implementation fixes.

    Configuration, prompts, corpus, schema, and graph validation stay immutable.
    A reviewed implementation revision is recorded separately, so existing unit
    hashes and raw request provenance never need to be rewritten.
    """
    path = output / "run.json"
    if not path.exists():
        if any(output.iterdir()):
            raise ValueError("A new annotation run needs an empty output directory; old artifacts will not be adopted: " + str(output))
        immutable_json(path, identity)
        return identity
    previous = read_json(path)
    if previous == identity:
        return previous
    implementation = {"runner_code_sha256", "normalizer_code_sha256", "jobs_code_sha256",
                      "runtime_policy_hash", "repair_schema_hash", "catalog_code_sha256",
                      "catalog_jobs_code_sha256", "grounding_checks_code_sha256", "annotation_protocol",
                      "source_anchored_code_sha256", "source_anchored_jobs_sha256"}
    if ({k: v for k, v in previous.items() if k not in implementation}
            != {k: v for k, v in identity.items() if k not in implementation}):
        raise ValueError("Annotation inputs/configuration changed; existing run cannot be resumed.")
    revision_path = output / "runner-revisions" / (object_hash(identity) + ".json")
    if not revision_path.is_file():
        raise ValueError("Runner changed without a recorded compatible revision: " + str(revision_path))
    revision = read_json(revision_path)
    if revision.get("base_run_hash") != object_hash(previous) or revision.get("identity") != identity or not revision.get("reason"):
        raise ValueError("Invalid compatible runner revision: " + str(revision_path))
    return previous


def annotation_passes(config: dict, pass_name: str | None = None) -> list[str]:
    if config["format_version"] != 1 or config["reasoning_effort"] not in ("none", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Invalid annotation configuration.")
    registered = config["passes"]
    defaults = config.get("default_passes", registered)
    for names in (registered, defaults):
        if (not isinstance(names, list) or not names or any(name not in ("primary", "blind") for name in names)
                or len(names) != len(set(names))):
            raise ValueError("Annotation passes must be nonempty, unique lists of primary/blind.")
    if not set(defaults).issubset(registered):
        raise ValueError("Default annotation pass is not registered.")
    selected = [pass_name] if pass_name else defaults
    if not set(selected).issubset(registered):
        raise ValueError("Unregistered annotation pass.")
    return selected


def resolve_codex(explicit: str | None = None) -> Path:
    specified = explicit or os.environ.get("ANNOTATION_CODEX")
    if specified:
        path = Path(specified)
        if not path.is_file():
            raise ValueError(f"ANNOTATION_CODEX does not point to an executable: {path}")
        return path.resolve()
    installed = shutil.which("codex.exe" if os.name == "nt" else "codex")
    if installed:
        return Path(installed).resolve()
    matches = sorted((Path.home() / ".vscode/extensions").glob("openai.chatgpt-*/bin/windows-x86_64/codex.exe"),
                     key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if matches:
        return matches[0].resolve()
    raise ValueError("Codex CLI not found. Set ANNOTATION_CODEX to your installed codex.exe.")


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        env.pop(key, None)
    # Preserve saved ChatGPT authentication; never copy or read its token file.
    return env


def validate_cli(codex: Path) -> dict:
    metadata = {}
    for label, arguments in (("version", ["--version"]), ("login", ["login", "status"]),
                             ("help", ["exec", "--help"]), ("features", ["features", "list"])):
        result = subprocess.run([str(codex), *arguments], env=child_environment(), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30)
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(f"Codex {label} check failed:\n{output}")
        metadata[label] = output.strip()
    if "logged in using chatgpt" not in metadata["login"].casefold():
        raise RuntimeError("Annotation requires the existing ChatGPT login. Run codex login; no API fallback will be used.")
    for flag in ("--ignore-user-config", "--output-schema", "--ephemeral", "--json"):
        if flag not in metadata["help"]:
            raise RuntimeError("Installed Codex lacks required annotation option: " + flag)
    available = {line.split()[0] for line in metadata["features"].splitlines() if line.split()}
    missing = set(DISABLED_FEATURES) - available
    if missing:
        raise RuntimeError("Codex feature controls changed; inspect before running annotations: " + ", ".join(sorted(missing)))
    return {"executable": str(codex), "version": metadata["version"], "authentication": "ChatGPT",
            "packages": {name: importlib.metadata.version(name) for name in ("pydantic", "pydantic-core")}}


def verified_story(corpus: Path, entry: dict) -> dict:
    story = read_json(corpus / entry["path"])
    content = {key: value for key, value in story.items() if key != "content_hash"}
    if object_hash(content) != entry["content_hash"] or story["content_hash"] != entry["content_hash"]:
        raise ValueError("Corpus story changed: " + entry["id"])
    return story


def compact_ledger(history: GraphHistory) -> dict:
    # Retain every prior identity and span; remove only redundant provenance and
    # numerical confidence fields. This is not a truncation of discourse history.
    return {
        "entities": [{"id": item.id, "label": item.label, "concept": item.concept, "kind": item.kind}
                     for item in history.entities.values()],
        "mentions": [{"id": item.id, "entity_id": item.entity_id, "span": item.span.model_dump()}
                     for item in history.mentions.values()],
        "events": [{"id": item.id, "predicate": item.predicate, "trigger": item.trigger.model_dump(),
                    "arguments": [a.model_dump() for a in item.arguments], "polarity": item.polarity,
                    "status": item.status, "scope_parent_id": item.scope_parent_id}
                   for item in history.events.values()],
        "relations": [{"type": item.type, "source_event": item.source_event,
                       "target_event": item.target_event, "status": item.status}
                      for item in history.relations.values()],
        "entity_updates": history.updates,
    }


def make_prompt(instructions: str, story: dict, unit: dict, history: GraphHistory) -> str:
    text = story["text"]
    pieces = []
    cursor = 0
    for word in story["words"][:unit["end_token"]]:
        pieces.extend((text[cursor:word["char_start"]], f"[{word['index']}]", text[word["char_start"]:word["char_end"]]))
        cursor = word["char_end"]
    pieces.append(text[cursor:unit["prefix_char_end"]])
    payload = {"story_id": story["story_id"], "unit_id": unit["id"],
               "current_unit_token_span": [unit["start_token"], unit["end_token"]],
               "indexed_prefix_text": "".join(pieces), "prior_ledger": compact_ledger(history)}
    return instructions + "\n\nANNOTATION INPUT (quoted data):\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def codex_command(codex: Path, config: dict, workspace: Path, schema: Path, output: Path) -> list[str]:
    command = [str(codex), "exec", "--ignore-user-config", "--model", config["model"],
               "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check", "--cd", str(workspace),
               "--output-schema", str(schema), "--output-last-message", str(output), "--json", "--color", "never",
               "-c", 'model_reasoning_effort="' + config["reasoning_effort"] + '"',
               "-c", 'web_search="disabled"']
    for feature in DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.append("-")
    return command


def terminate_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def check_events(path: Path) -> dict:
    usage = []
    startup_diagnostics = []
    started = False
    completed = False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") in ("error", "turn.failed"):
                raise RuntimeError("Codex reported: " + json.dumps(event, ensure_ascii=False) + "; log: " + str(path))
            if event.get("type") == "turn.started":
                started = True
            if event.get("type") == "turn.completed":
                if not started:
                    raise RuntimeError("Codex reported completion without a started turn: " + str(path))
                completed = True
                usage.append(event.get("usage", {}))
            item = event.get("item", {})
            if (not started and event.get("type") == "item.completed" and item.get("type") == "error"
                    and item.get("message") == DISABLED_CODE_MODE_DIAGNOSTIC):
                startup_diagnostics.append(item["message"])
                continue
            if event.get("type", "").startswith("item.") and item.get("type") not in ("agent_message", "reasoning"):
                raise RuntimeError(f"Annotation emitted unexpected item type {item.get('type')!r}: "
                                   f"{item.get('message', '')}. Inspect {path}")
    if not completed:
        raise RuntimeError("Codex did not report a completed turn: " + str(path))
    return {"reported_usage": usage, "startup_diagnostics": startup_diagnostics}


def invoke_codex(command: list[str], prompt: str, log: Path, timeout: int,
                 heartbeat=None, heartbeat_seconds: int = 30) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    stderr = log.with_suffix(".stderr.txt")
    process = None
    started = time.monotonic()
    try:
        with log.open("wb") as output, stderr.open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=errors,
                                       env=child_environment(), start_new_session=os.name != "nt",
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            first_wait = True
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.communicate(input=prompt.encode("utf-8") if first_wait else None,
                                        timeout=min(heartbeat_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    first_wait = False
                    if time.monotonic() - started >= timeout:
                        raise
                    if heartbeat is not None:
                        heartbeat(time.monotonic() - started)
            if process.returncode:
                detail = stderr.read_text(encoding="utf-8", errors="replace")[-1800:]
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") in ("error", "turn.failed"):
                        detail += "\n" + json.dumps(event, ensure_ascii=False)
                raise RuntimeError(f"Codex exited {process.returncode}; valid units are retained.\n{detail}\nLog: {log}")
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Codex exceeded {timeout} seconds. Valid units are retained; inspect {log}.") from error
    finally:
        if process is not None:
            terminate_child(process)
    return {**check_events(log), "wall_seconds": time.monotonic() - started}


def implementation_identity(root: Path, config: dict, index: dict, instructions: str, schema: dict) -> dict:
    from .annotation_catalog import PROTOCOL
    identity = {"format_version": 1, "configuration": config, "corpus_hash": index["content_hash"],
            "schema_hash": object_hash(schema), "prompt_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
            "runner_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "normalizer_code_sha256": hashlib.sha256((root / "neurosym/annotation_repair.py").read_bytes()).hexdigest(),
            "jobs_code_sha256": hashlib.sha256((root / "neurosym/annotation_jobs.py").read_bytes()).hexdigest(),
            "catalog_code_sha256": hashlib.sha256((root / "neurosym/annotation_catalog.py").read_bytes()).hexdigest(),
            "catalog_jobs_code_sha256": hashlib.sha256((root / "neurosym/annotation_catalog_jobs.py").read_bytes()).hexdigest(),
            "grounding_checks_code_sha256": hashlib.sha256((root / "neurosym/annotation_grounding_checks.py").read_bytes()).hexdigest(),
            "annotation_protocol": config.get("protocol", PROTOCOL),
            "runtime_policy_hash": object_hash(read_json(root / "configs/annotation-runtime.json")),
            "repair_schema_hash": object_hash(read_json(root / "schemas/graph-repair.schema.json")),
            "graph_code_sha256": hashlib.sha256((root / "neurosym/graphs.py").read_bytes()).hexdigest()}
    if config.get("protocol") == "source-anchored-v2":
        identity["source_anchored_code_sha256"] = hashlib.sha256((root / "neurosym/annotation_anchored.py").read_bytes()).hexdigest()
        identity["source_anchored_jobs_sha256"] = hashlib.sha256((root / "neurosym/annotation_anchored_jobs.py").read_bytes()).hexdigest()
    if config.get("protocol") == "joint-source-v3":
        for name in ("annotation_joint.py", "annotation_joint_jobs.py", "annotation_anchored.py"):
            identity[name + "_sha256"] = hashlib.sha256((root / "neurosym" / name).read_bytes()).hexdigest()
    if config.get("protocol") in ("joint-source-v4", "joint-source-v5"):
        for name in ("annotation_source.py", "annotation_source_jobs.py", "annotation_completion.py", "annotation_reuse.py",
                     "annotation_joint.py", "annotation_anchored.py", "deniz.py"):
            identity[name + "_sha256"] = hashlib.sha256((root / "neurosym" / name).read_bytes()).hexdigest()
    return identity


def annotate(root: Path, config_path: Path, *, codex_path: str | None = None, pass_name: str | None = None) -> dict:
    from .annotation_jobs import run_jobs, validate_policy
    from .annotation_catalog_jobs import CatalogStoryJob
    from .annotation_catalog import PROTOCOL
    from .annotation_repair import repair_schema

    config = read_json(config_path)
    protocol = config.get("protocol", PROTOCOL)
    job_type = CatalogStoryJob
    if protocol in ("joint-source-v3", "joint-source-v4", "joint-source-v5"):
        if protocol in ("joint-source-v4", "joint-source-v5"):
            from .annotation_source_jobs import SourceStoryJob
            job_type = SourceStoryJob
        else:
            from .annotation_joint_jobs import JointStoryJob
            job_type = JointStoryJob
        if type(config.get("annotation_attempts")) is not int or config["annotation_attempts"] < 1:
            raise ValueError("Joint annotation needs a positive annotation_attempts budget.")
    elif protocol == "source-anchored-v2":
        from .annotation_anchored_jobs import AnchoredStoryJob
        job_type = AnchoredStoryJob
        for field in ("grounding_attempts", "binding_attempts"):
            if type(config.get(field)) is not int or not 1 <= config[field] <= 3:
                raise ValueError("Source-anchored configuration requires 1..3 attempts for " + field)
    elif protocol != PROTOCOL:
        raise ValueError("Unknown annotation protocol: " + protocol)
    passes = annotation_passes(config, pass_name)
    corpus = (root / config["corpus"]).resolve()
    output = (root / config["output"]).resolve()
    index = read_json(corpus / "index.json")
    if index["content_hash"] != object_hash({k: v for k, v in index.items() if k != "content_hash"}):
        raise ValueError("Corpus index changed.")
    instructions = (root / config["prompt"]).read_text(encoding="utf-8")
    schema = graph_schema()
    schema_path = (root / config["schema"]).resolve()
    if read_json(schema_path) != schema:
        raise ValueError("Exported graph schema differs from the current code.")
    if read_json(root / "schemas/graph-repair.schema.json") != repair_schema():
        raise ValueError("Exported repair schema differs from current code; run scripts/export_graph_schema.py.")
    policy = read_json(root / "configs/annotation-runtime.json")
    validate_policy(policy)
    implementation = implementation_identity(root, config, index, instructions, schema)
    output.mkdir(parents=True, exist_ok=True)
    identity = initialize_run(output, implementation)
    workspace = root / "artifacts/annotation-context"
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise ValueError("Annotation working directory must stay empty: " + str(workspace))
    lock = output / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_utc": utcnow()}))
    except FileExistsError as error:
        raise RuntimeError("An annotation run is active or was terminated abruptly. Inspect " + str(lock)) from error
    try:
        codex = resolve_codex(codex_path)
        cli = validate_cli(codex)
        print(f"ANNOTATION protocol={protocol} model={config['model']} reasoning={config['reasoning_effort']} "
              f"passes={','.join(passes)} policy={policy}", flush=True)
        return run_jobs(output=output, corpus=corpus, config=config, index=index, identity=identity,
                        implementation_hash=object_hash(implementation), instructions=instructions, codex=codex,
                        cli=cli, workspace=workspace, graph_schema_path=schema_path,
                        repair_schema_path=root / "schemas/graph-repair.schema.json", policy=policy, passes=passes,
                        job_type=job_type)
    finally:
        lock.unlink(missing_ok=True)
