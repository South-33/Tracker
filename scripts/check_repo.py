"""Fast repository sanity check for the temporal-tracker research loop."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_PYTHON = (
    ROOT / "src" / "tracker",
    ROOT / "scripts",
    ROOT / "tests",
)
FORBIDDEN_ACTIVE_TERMS = (
    "ultralytics.trackers",
    "KalmanFilter",
    "BYTETracker",
    "BOTSORT",
)


def run(*command: str) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def fail(message: str) -> None:
    raise SystemExit(f"repo check failed: {message}")


def check_docs() -> None:
    markdown = sorted(path.name for path in ROOT.glob("*.md"))
    expected = ["AGENTS.md", "tracker.md"]
    if markdown != expected:
        fail(f"top-level Markdown must be exactly {expected}; found {markdown}")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    tracker = (ROOT / "tracker.md").read_text(encoding="utf-8")
    if "tracker.md" not in agents:
        fail("AGENTS.md must direct agents to tracker.md")
    if "Research loop" not in tracker or "Established evidence" not in tracker:
        fail("tracker.md is missing the stable research-loop/evidence structure")


def check_active_code() -> None:
    legacy_files = (
        ROOT / "src" / "tracker" / "causal.py",
        ROOT / "src" / "tracker" / "runtime.py",
        ROOT / "src" / "tracker" / "system.py",
        ROOT / "src" / "tracker" / "association.py",
    )
    present = [path.name for path in legacy_files if path.exists()]
    if present:
        fail(f"legacy tracker modules returned to active source: {present}")

    for folder in (ROOT / "src" / "tracker",):
        for path in folder.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for term in FORBIDDEN_ACTIVE_TERMS:
                if term in text:
                    fail(f"{path.relative_to(ROOT)} contains forbidden active term {term!r}")


def check_branch() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
    ).strip()
    if branch != "main":
        fail(f"research work must stay on main, not {branch!r}")


def check_generated_clutter() -> None:
    stale = []
    for root in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        stale.extend(root.rglob("__pycache__"))
    if stale:
        fail(
            "remove stale __pycache__ directories before checkpointing: "
            + ", ".join(str(path.relative_to(ROOT)) for path in stale)
        )


def check_syntax() -> None:
    for root in ACTIVE_PYTHON:
        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            try:
                compile(source, str(path), "exec")
            except SyntaxError as error:
                fail(f"syntax error in {path.relative_to(ROOT)}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Also require a clean Git working tree (use after committing).",
    )
    args = parser.parse_args()

    check_branch()
    check_docs()
    check_active_code()
    check_generated_clutter()
    check_syntax()

    run(sys.executable, "-B", "-m", "unittest", "tests.test_core")
    run("git", "diff", "--check")

    if args.clean:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
        if status:
            fail("working tree is not clean")

    print("repo check passed")


if __name__ == "__main__":
    main()
