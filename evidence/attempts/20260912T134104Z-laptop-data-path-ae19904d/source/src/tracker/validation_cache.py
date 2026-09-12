"""Reuse unit-test results only while executable inputs and dependencies match.

Dynamic ledger/data checks still run in lab doctor for every research launch.
Explicit lab test always executes the suite.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def test_fingerprint(root: Path, environment: Path | None = None) -> str:
    environment = Path(sys.prefix) if environment is None else environment
    digest = hashlib.sha256()
    digest.update(json.dumps({"python": sys.version, "executable": sys.executable,
                              "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
                              "pythonpath": os.environ.get("PYTHONPATH", "")}, sort_keys=True).encode())
    paths = set()
    for folder in ("src", "tests", "scripts", "experiments", "third_party"):
        paths.update(path for path in (root / folder).rglob("*.py") if "__pycache__" not in path.parts and ".git" not in path.parts)
    for name in ("pyproject.toml", "requirements-laptop.txt", "pytest.ini", "conftest.py"):
        if (root / name).is_file():
            paths.add(root / name)
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    # A dependency reinstall, upgrade, or editable-package change invalidates
    # the cache. Do not import torch just to determine whether tests are cached.
    package_roots = [environment / "Lib/site-packages", *environment.glob("lib/python*/site-packages")]
    for packages in package_roots:
        metadata = [*packages.glob("*.dist-info/METADATA"), *packages.glob("*.pth")]
        for path in sorted(metadata):
            stat = path.stat()
            digest.update(str(path).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def passing_result(root: Path, fingerprint: str) -> dict | None:
    try:
        record = json.loads((root / "runs/.lab-tests.json").read_text(encoding="utf-8"))
        return record if record.get("fingerprint") == fingerprint and record.get("returncode") == 0 else None
    except (OSError, ValueError):
        return None
