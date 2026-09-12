"""Small guardrail for the active research-cycle budget.

The scientific reasoning lives in tracker.md. This module keeps the next decision
obvious, counts decision-changing experiments, and prevents quiet probe creep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_PROBE_KINDS = ("mechanism", "reality", "break-it")
VALID_STATUSES = (
    "probing", "decision_required", "big_run_required", "big_run_running", "evaluation_required",
    "complete", "killed",
)


def default_ledger_path() -> Path:
    return Path(__file__).resolve().parents[2] / "research-cycle.json"


def load_ledger(path: str | Path | None = None) -> dict[str, Any]:
    ledger_path = Path(path) if path is not None else default_ledger_path()
    # Accept both plain UTF-8 and the BOM-prefixed UTF-8 that Windows
    # PowerShell may emit when touching the ledger. The ledger is human-edited
    # research state, so a harmless encoding marker must not block the lab.
    data = json.loads(ledger_path.read_text(encoding="utf-8-sig"))
    validate_ledger(data)
    return data


def validate_ledger(data: dict[str, Any]) -> None:
    if data.get("version") != 1:
        raise ValueError("research-cycle.json must use version 1")
    if data.get("status") not in VALID_STATUSES:
        raise ValueError(f"invalid cycle status: {data.get('status')!r}")
    budget = data.get("probe_budget")
    if not isinstance(budget, int) or not 0 <= budget <= len(VALID_PROBE_KINDS):
        raise ValueError(f"probe_budget must be between 0 and {len(VALID_PROBE_KINDS)}")

    probes = data.get("probes")
    if not isinstance(probes, list):
        raise ValueError("probes must be a list")
    if len(probes) > budget:
        raise ValueError("probe budget exceeded")

    kinds: set[str] = set()
    for probe in probes:
        kind = probe.get("kind")
        if kind not in VALID_PROBE_KINDS:
            raise ValueError(f"invalid probe kind: {kind!r}")
        if kind in kinds:
            raise ValueError(f"duplicate probe kind: {kind}")
        kinds.add(kind)

    active_probe = data.get("probe")
    if active_probe:
        active_kind = active_probe.get("kind")
        if data["status"] != "probing":
            raise ValueError("active probe requires probing status")
        if active_kind not in VALID_PROBE_KINDS:
            raise ValueError(f"invalid active probe kind: {active_kind!r}")
        if active_kind in kinds:
            raise ValueError(f"active probe kind already completed: {active_kind}")
        if len(probes) >= budget:
            raise ValueError("active probe exceeds probe budget")

    if data["status"] == "probing" and len(probes) >= budget:
        raise ValueError("probing is inconsistent with an exhausted probe budget")
    small_budget = data.get("exploration_budget", 3)
    if not isinstance(small_budget, int) or not 1 <= small_budget <= 3:
        raise ValueError("exploration_budget must be between 1 and 3")
    diagnostics = data.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        raise ValueError("diagnostics must be a list")
    if data["status"] == "probing" and exploration_used(data) >= small_budget:
        raise ValueError("probing is inconsistent with an exhausted exploration budget")


def exploration_used(data: dict[str, Any]) -> int:
    """Train-only diagnostics count too; checkpoint evaluations within a big run do not."""
    return len(data.get("probes", [])) + len(data.get("diagnostics", []))


def action_allowed(data: dict[str, Any], action: str, probe_kind: str | None = None) -> tuple[bool, str]:
    status = data["status"]
    if action in {"probe", "diagnose"} and exploration_used(data) >= data.get("exploration_budget", 3):
        return False, "Exploration budget spent. Commit to a substantial run or document a new direction; another small variant is not allowed."
    if action == "diagnose":
        return (status == "probing", "Train-only experiment allowed" if status == "probing" else f"cycle status is {status}; decide the next substantial action first")
    if action == "probe":
        if status != "probing":
            return False, f"cycle status is {status}; another probe is out of contract"
        if probe_kind not in VALID_PROBE_KINDS:
            return False, f"probe kind must be one of {', '.join(VALID_PROBE_KINDS)}"
        active = data.get("probe")
        if active and active.get("kind") != probe_kind:
            return False, f"{active.get('kind')} probe is already active; finish it before another experiment"
        used = {probe["kind"] for probe in data["probes"]}
        if probe_kind in used:
            return False, f"{probe_kind} probe already used"
        if len(data["probes"]) >= data["probe_budget"]:
            return False, "probe budget exhausted"
        return True, f"{probe_kind} probe allowed"

    if action == "big-run":
        if status not in {"big_run_required", "big_run_running"}:
            return False, f"cycle status is {status}; big run is not the active action"
        return True, "big run is the active action"

    raise ValueError(f"unknown action: {action}")


def status_text(data: dict[str, Any]) -> str:
    probes = ", ".join(
        f"{probe['kind']}:{probe.get('decision', probe.get('status', 'done'))}"
        for probe in data["probes"]
    ) or "none"
    remaining_slots = data["probe_budget"] - len(data["probes"])
    big = data.get("big_run", {})
    lines = [
        f"cycle: {data['cycle']}",
        f"status: {data['status']}",
        f"formal probes used: {len(data['probes'])}/{data['probe_budget']} ({probes})",
        f"optional probe slots available: {remaining_slots}",
        f"jump claim: {data['jump_claim']}",
        f"exploratory experiments used: {exploration_used(data)}/{data.get('exploration_budget', 3)} (includes train-only diagnostics)",
    ]
    if data.get("incumbent") is None:
        lines.append("validated incumbent: none; do not report historical oracle/component scores as tracker progress")
    else:
        lines.append(f"validated incumbent: {data['incumbent']}")
    if data.get("next_action"):
        lines.append(f"priority: {data['next_action']}")
    development = data.get("development", {})
    active_development = development.get("active_experiment")
    if active_development:
        lines.append(
            f"active train-only development: {active_development.get('name', 'unspecified')} "
            f"({active_development.get('scope', 'train-only')})"
        )
        if active_development.get("question"):
            lines.append(f"active question: {active_development['question']}")
    active_probe = data.get("probe")
    if active_probe:
        lines.append(
            f"active formal probe: {active_probe.get('kind', 'unspecified')} "
            f"({active_probe.get('scope', active_probe.get('split', 'unspecified'))})"
        )
        if active_probe.get("question"):
            lines.append(f"active question: {active_probe['question']}")
    if big:
        lines.append(f"big run: {big.get('status', 'unspecified')}")
        manifest = big.get("manifest", {})
        initialize = big.get("initialize", {})
        lines.append(f"manifest: {manifest.get('path', manifest)}")
        lines.append(f"initialize: {initialize.get('path', initialize)}")
    evaluation = data.get("evaluation")
    if evaluation:
        lines.append(f"evaluation: {evaluation.get('status', 'unspecified')} ({evaluation.get('protocol', 'unspecified')})")
        if data["status"] == "evaluation_required" and evaluation.get("status") != "complete":
            lines.append("next action: EVALUATE THE FROZEN MECHANISM ON NON-GATE DEV BEFORE TUNING OR NEW IDEAS.")
        elif data["status"] == "evaluation_required" and evaluation.get("status") == "complete":
            lines.append("next action: compare the frozen system to the incumbent. Scale a learning mechanism, fix a confirmed defect, or change the causal hypothesis; do not drift into micro-tuning.")
    if data["status"] == "probing" and not active_probe and not active_development:
        remaining = [kind for kind in VALID_PROBE_KINDS if kind not in {probe["kind"] for probe in data["probes"]}]
        lines.append(
            "next action: decide from current evidence. Use another formal probe only if it can change that decision; "
            + (f"available roles: {', '.join(remaining)}." if remaining else "no probe roles remain.")
        )
    elif data["status"] == "probing" and (active_probe or active_development):
        if active_development and active_development.get("status") == "complete" and not active_probe:
            lines.append(
                "next action: interpret/archive the completed train-only experiment, clear it from active development, "
                "then decide the next question. Do not start another variant before that cleanup."
            )
        else:
            lines.append("next action: finish and interpret the active experiment. Do not start another variant in parallel.")
    if data["status"] == "decision_required":
        lines.append("next action: choose and declare a substantial run/bet or a reasoned change of direction. Do not reset the counter to continue the same small probes.")
    if data["status"] == "big_run_required":
        lines.append("next action: run the declared substantial experiment through its planned compute and causal evaluation. No additional exploratory variants.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=default_ledger_path())
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    check = sub.add_parser("check", help="check whether a research action is allowed")
    check.add_argument("action", choices=("probe", "diagnose", "big-run"))
    check.add_argument("--kind", choices=VALID_PROBE_KINDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_ledger(args.ledger)
    if args.command in {None, "status"}:
        print(status_text(data))
        return 0
    allowed, message = action_allowed(data, args.action, args.kind)
    print(message)
    return 0 if allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
