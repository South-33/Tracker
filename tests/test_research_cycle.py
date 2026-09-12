import copy
import json

import pytest

from tracker.research_cycle import VALID_PROBE_KINDS, action_allowed, load_ledger, status_text, validate_ledger


def test_current_cycle_obeys_action_contract():
    data = load_ledger()
    used = {probe["kind"] for probe in data["probes"]}
    active = data.get("probe", {}).get("kind")
    for kind in VALID_PROBE_KINDS:
        allowed, _ = action_allowed(data, "probe", kind)
        assert allowed == (
            data["status"] == "probing"
            and kind not in used
            and len(data["probes"]) < data["probe_budget"]
            and (active is None or active == kind)
        )
    big_run_allowed, _ = action_allowed(data, "big-run")
    assert big_run_allowed == (data["status"] in {"big_run_required", "big_run_running"})


def test_duplicate_probe_kinds_are_invalid():
    data = load_ledger()
    broken = copy.deepcopy(data)
    broken.pop("probe", None)
    broken["probe_budget"] = 3
    broken["status"] = "killed"
    broken["probes"] = [
        {"kind": VALID_PROBE_KINDS[0], "status": "complete"},
        {"kind": VALID_PROBE_KINDS[0], "status": "complete"},
    ]
    with pytest.raises(ValueError, match="duplicate probe kind"):
        validate_ledger(broken)


def test_probing_cannot_hide_an_exhausted_budget():
    data = load_ledger()
    broken = copy.deepcopy(data)
    broken.pop("probe", None)
    broken.pop("probe", None)
    kinds = VALID_PROBE_KINDS
    broken["probes"] = [
        {"kind": kind, "status": "complete"}
        for kind in kinds[:broken["probe_budget"]]
    ]
    broken["status"] = "probing"
    with pytest.raises(ValueError, match="exhausted probe budget"):
        validate_ledger(broken)


def test_exhausted_budget_has_explicit_decision_state():
    data = load_ledger()
    broken = copy.deepcopy(data)
    broken.pop("probe", None)
    broken.pop("probe", None)
    broken["probes"] = [
        {"kind": kind, "status": "complete"}
        for kind in VALID_PROBE_KINDS[:broken["probe_budget"]]
    ]
    broken["status"] = "decision_required"
    validate_ledger(broken)


def test_decision_can_be_required_before_budget_is_spent():
    data = load_ledger()
    early = copy.deepcopy(data)
    early.pop("probe", None)
    early.pop("probe", None)
    early["probes"] = []
    early["status"] = "decision_required"
    validate_ledger(early)


def test_probe_budget_cannot_exceed_available_roles():
    data = load_ledger()
    broken = copy.deepcopy(data)
    broken["probe_budget"] = len(VALID_PROBE_KINDS) + 1
    with pytest.raises(ValueError, match="probe_budget"):
        validate_ledger(broken)


def test_load_ledger_accepts_utf8_bom(tmp_path):
    source = load_ledger()
    ledger = tmp_path / "research-cycle.json"
    ledger.write_text(json.dumps(source, indent=2), encoding="utf-8-sig")
    assert load_ledger(ledger)["cycle"] == source["cycle"]


def test_completed_development_status_requests_archive_not_more_training():
    data = load_ledger()
    complete = copy.deepcopy(data)
    complete.pop("probe", None)
    complete["status"] = "probing"
    complete.setdefault("development", {})["active_experiment"] = {
        "name": "finished-train-only-test",
        "status": "complete",
        "scope": "train-only",
        "question": "Did it work?",
    }
    text = status_text(complete)
    assert "interpret/archive" in text
    assert "finish and interpret" not in text
