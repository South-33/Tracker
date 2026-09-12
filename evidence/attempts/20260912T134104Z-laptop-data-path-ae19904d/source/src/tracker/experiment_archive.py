"""Save executable source before an experiment and its outcome afterward.

Small evidence is versioned; datasets, caches and checkpoints stay local. A
snapshot is evidence of what ran, not a claim that it can run without its inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def inside(root: Path, path: str | Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Experiment path leaves the repository: {path}")
    return resolved


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def start_experiment(root: Path, name: str, script: str | Path, command: list[str],
                     ledger: dict, output: str | Path) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Experiment name must contain lowercase letters, digits, - or _")
    root = root.resolve()
    script = inside(root, script)
    output = inside(root, output)
    if not script.is_file():
        raise FileNotFoundError(script)
    now = datetime.now(timezone.utc)
    directory = root / "evidence" / "attempts" / f"{now:%Y%m%dT%H%M%SZ}-{name}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    sources = {script}
    sources.update(path for path in script.parent.rglob("*.py") if "__pycache__" not in path.parts)
    for folder in ("src", "scripts", "experiments"):
        sources.update(path for path in (root / folder).rglob("*.py") if "__pycache__" not in path.parts)
    for filename in ("pyproject.toml", "requirements-laptop.txt", "AGENTS.md", "GOAL.md", "tracker.md", "research-cycle.json"):
        if (root / filename).is_file():
            sources.add(root / filename)
    source_hashes = {}
    for source in sorted(sources):
        source = inside(root, source)
        relative = source.relative_to(root)
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative.as_posix()] = digest(destination)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    record = {
        "version": 1, "name": name, "cycle": ledger.get("cycle"),
        "started_at": now.isoformat(), "status": "running", "git_revision": revision,
        "command": command, "script": script.relative_to(root).as_posix(),
        "expected_output": output.relative_to(root).as_posix(),
        "source_sha256": source_hashes, "ledger": ledger,
        "reproduction": "Restore the source snapshot at repository root and supply the ledger's exact local input hashes. External dependencies/data/checkpoints are not bundled.",
    }
    path = directory / "record.json"
    _write(path, record)
    return path


def finish_experiment(root: Path, record_path: Path, returncode: int | None,
                      error: str | None = None) -> dict:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = returncode
    record["status"] = "complete" if returncode == 0 and error is None else "failed"
    if error:
        record["error"] = error
    output = inside(root, record["expected_output"])
    if output.is_file():
        record["output_sha256"] = digest(output)
        record["output_bytes"] = output.stat().st_size
        if output.suffix == ".json" and output.stat().st_size <= 2 * 1024 * 1024:
            try:
                json.loads(output.read_text(encoding="utf-8"))
                shutil.copy2(output, record_path.parent / "result.json")
                record["archived_result"] = "result.json"
            except (ValueError, UnicodeError) as exc:
                record["status"] = "invalid_result"
                record["error"] = str(exc)
    elif record["status"] == "complete":
        record["status"] = "missing_result"
    _write(record_path, record)
    return record
