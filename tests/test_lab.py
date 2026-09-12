from pathlib import Path
import copy
import json
from types import SimpleNamespace
from tracker.lab import big_run_input_issues, probe_input_issues, context_issues, development_issues, diagnostic_issues, progress_text, source_lock_issues, sync_cycle_marker
from tracker.research_cycle import load_ledger


def test_current_big_run_inputs_are_frozen_and_clean():
    assert big_run_input_issues(load_ledger()) == []


def test_current_probe_inputs_are_frozen_and_clean():
    assert probe_input_issues(load_ledger()) == []


def test_current_development_inputs_are_frozen_and_clean():
    assert development_issues(load_ledger()) == []


def test_probe_input_guard_checks_optional_frozen_weights(tmp_path, monkeypatch):
    from tracker import lab
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "manifest.json"
    weights = tmp_path / "adapter.pt"
    script = tmp_path / "probe.py"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text('{"records": [{"sequence": "dev1", "split": "dev"}]}', encoding="utf-8")
    weights.write_bytes(b"adapter")
    script.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    ledger = {
        "promotion": {"gate_sequence": "gate"},
        "probe": {
            "checkpoint": {"path": "checkpoint.pt", "sha256": lab.sha256(checkpoint)},
            "manifest": {"path": "manifest.json", "sha256": lab.sha256(manifest)},
            "weights": {"path": "adapter.pt", "sha256": lab.sha256(weights)},
            "split": "dev",
            "script": "probe.py",
        },
    }
    assert probe_input_issues(ledger) == []
    weights.write_bytes(b"changed")
    assert any("probe weights hash changed" in issue for issue in probe_input_issues(ledger))


def test_probe_command_refuses_existing_output():
    source = (Path(__file__).resolve().parents[1] / "src/tracker/lab.py").read_text(encoding="utf-8")
    probe = source.split("def command_probe", 1)[1].split("def command_eval", 1)[0]
    assert "refusing to rerun completed probe output" in probe


def test_doctor_rejects_bad_curriculum_mix_when_big_run_exists():
    ledger = copy.deepcopy(load_ledger())
    if "big_run" not in ledger or "curriculum" not in ledger["big_run"]:
        return
    ledger["big_run"]["curriculum"]["ordinary_probability"] = .9
    issues = big_run_input_issues(ledger)
    assert any("curriculum probabilities" in issue for issue in issues)


def test_progress_text_has_zero_state_for_fresh_output(tmp_path):
    ledger = copy.deepcopy(load_ledger())
    if "big_run" not in ledger:
        return
    ledger["big_run"]["output"] = str(tmp_path / "missing")
    assert progress_text(ledger["big_run"]) == [
        f"progress: 0/{ledger['big_run']['training']['steps']}"
    ]


def test_progress_text_understands_local_recovery(tmp_path, monkeypatch):
    from tracker import lab
    big = {
        "output": "run",
        "training": {"steps": 100, "feature_cache": "cache"},
    }
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "step": 10, "epoch": 2, "sequence": "s1",
        "accepted_write_precision": .99, "visible_write_recall": .6,
        "local_safe_acceptance": .8, "local_hazard_veto_recall": .9,
        "recovery_safe_acceptance": .7, "recovery_hazard_veto_recall": .95,
        "wrong_write_rate": .001, "clip_strict_survival": .95,
    }
    (run / "loss.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    text = "\n".join(progress_text(big))
    assert "local_veto=0.900" in text
    assert "recovery_veto=0.950" in text
    assert "fresh-start epoch 2" in text
    assert "precision=0.990" in text


def test_current_context_is_clean():
    assert context_issues(load_ledger()) == []


def test_context_rejects_status_drift_but_allows_normal_docs(tmp_path):
    ledger = copy.deepcopy(load_ledger())
    (tmp_path / "tracker.md").write_text("<!-- cycle-status: complete -->\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("stale\n", encoding="utf-8")
    issues = context_issues(ledger, tmp_path)
    assert not any("README" in issue for issue in issues)
    assert any("cycle marker disagrees" in issue for issue in issues)


def test_cycle_marker_sync_repairs_mechanical_drift(tmp_path):
    ledger = copy.deepcopy(load_ledger())
    (tmp_path / "tracker.md").write_text("# Tracker\n\n<!-- cycle-status: killed -->\n", encoding="utf-8")
    assert sync_cycle_marker(ledger, tmp_path)
    assert f"<!-- cycle-status: {ledger['status']} -->" in (tmp_path / "tracker.md").read_text(encoding="utf-8")



def test_current_source_lock_is_clean():
    assert source_lock_issues(load_ledger()) == []


def test_source_lock_rejects_drift(tmp_path):
    ledger = {"source_lock": {"mechanism.py": "0" * 64}}
    (tmp_path / "mechanism.py").write_text("changed\n", encoding="utf-8")
    issues = source_lock_issues(ledger, tmp_path)
    assert any("source drift" in issue for issue in issues)


def test_doctor_aggregation_mentions_source_lock():
    source = (Path(__file__).resolve().parents[1] / "src/tracker/lab.py").read_text(encoding="utf-8")
    doctor = source.split("def command_doctor", 1)[1].split("def command_test", 1)[0]
    assert "source_lock_issues(ledger)" in doctor


def test_resource_guard_keeps_headroom_and_caps_library_threads():
    from tracker import lab
    assert lab.ACTIVE_CPU_MAX_PERCENT < lab.IDLE_CPU_MAX_PERCENT
    assert lab.ACTIVE_CPU_RESUME_PERCENT < lab.ACTIVE_CPU_MAX_PERCENT
    assert lab.IDLE_CPU_RESUME_PERCENT < lab.IDLE_CPU_MAX_PERCENT
    assert lab.ACTIVE_CPU_CORES <= 2
    assert lab.IDLE_CPU_CORES >= lab.ACTIVE_CPU_CORES
    env = lab.managed_environment()
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        assert env[name] == str(lab.CPU_THREAD_BUDGET)


def test_resource_profile_expands_only_after_user_is_idle():
    from tracker import lab
    active = lab.resource_profile(0.0)
    idle = lab.resource_profile(lab.USER_IDLE_BOOST_SECONDS + 1.0)
    unknown = lab.resource_profile(None)
    assert active["mode"] == "interactive"
    assert unknown["mode"] == "interactive"
    assert idle["mode"] == "idle"
    assert idle["cores"] >= active["cores"]
    assert idle["max_cpu"] > active["max_cpu"]


def test_resource_guard_pauses_and_resumes_on_system_load(monkeypatch, tmp_path):
    from tracker import lab

    class FakeProcess:
        pid = 1234
        returncode = 0

        def __init__(self):
            self.polls = iter((None, None, 0))

        def poll(self):
            return next(self.polls, 0)

        def terminate(self):
            self.returncode = -1

    fake = FakeProcess()
    monkeypatch.setattr(lab.subprocess, "Popen", lambda *args, **kwargs: fake)
    monkeypatch.setattr(lab, "_set_research_priority", lambda process, profile: None)
    events = []
    monkeypatch.setattr(lab, "_suspend_research", lambda process: events.append("suspend"))
    monkeypatch.setattr(lab, "_resume_research", lambda process: events.append("resume"))
    monkeypatch.setattr(lab, "user_idle_seconds", lambda: 0.0)
    readings = iter((0.0, 95.0, 40.0))
    monkeypatch.setattr(lab.psutil, "cpu_percent", lambda interval=None: next(readings))

    result = lab.run_managed(["python", "probe.py"], cwd=tmp_path)
    assert result.returncode == 0
    assert events == ["suspend", "resume"]


def test_diagnostic_guard_accepts_only_train_no0096_scripts(tmp_path):
    directory = tmp_path / "runs" / "_diag_fresh-context"
    directory.mkdir(parents=True)
    script = directory / "probe.py"
    script.write_text(
        "MANIFEST = 'data/manifest-train-full-dense-no0096.json'\n"
        "load_sequences(MANIFEST, 'train')\n",
        encoding="utf-8",
    )
    assert diagnostic_issues("fresh-context", tmp_path) == []

    script.write_text(
        "MANIFEST = 'data/manifest-train-full-dense-no0096.json'\n"
        "load_sequences(MANIFEST, 'dev')\n",
        encoding="utf-8",
    )
    issues = diagnostic_issues("fresh-context", tmp_path)
    assert any("train split" in issue for issue in issues)

    script.write_text(
        "MANIFEST = 'data/manifest-train-full-dense-no0096.json'\n"
        "# dancetrack0096 must stay untouched\n"
        "load_sequences(MANIFEST, 'train')\n",
        encoding="utf-8",
    )
    assert diagnostic_issues("fresh-context", tmp_path) == []

    script.write_text(
        "MANIFEST = 'data/manifest-train-full-dense-no0096.json'\n"
        "GATE = 'dancetrack0096'\n"
        "load_sequences(MANIFEST, 'train')\n",
        encoding="utf-8",
    )
    issues = diagnostic_issues("fresh-context", tmp_path)
    assert any("architecture gate" in issue for issue in issues)


def test_diagnostic_guard_rejects_invalid_name_and_missing_script(tmp_path):
    assert diagnostic_issues("../escape", tmp_path)
    assert diagnostic_issues("missing", tmp_path)


def test_diagnostic_guard_allows_learned_train_only_development(tmp_path):
    directory = tmp_path / "runs" / "_diag_training"
    directory.mkdir(parents=True)
    script = directory / "probe.py"
    script.write_text(
        "MANIFEST = 'data/manifest-train-full-dense-no0096.json'\n"
        "load_sequences(MANIFEST, 'train')\n"
        "optimizer = torch.optim.AdamW(model.parameters())\n"
        "loss.backward()\n",
        encoding="utf-8",
    )
    assert diagnostic_issues("training", tmp_path) == []


def test_diagnose_source_locks_and_completes_nonlearned_active_experiment(tmp_path, monkeypatch):
    from tracker import lab

    directory = tmp_path / "runs" / "_diag_ceiling"
    directory.mkdir(parents=True)
    script = directory / "probe.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    ledger = {
        "development": {
            "active_experiment": {
                "name": "ceiling",
                "status": "planned",
                "script_sha256": lab.sha256(script),
            }
        }
    }
    written = []
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "diagnostic_issues", lambda name: [])
    monkeypatch.setattr(lab, "learned_experiment_signals", lambda text: [])
    monkeypatch.setattr(lab, "load_ledger", lambda: copy.deepcopy(ledger))
    monkeypatch.setattr(lab, "atomic_write_ledger", lambda value: written.append(copy.deepcopy(value)))
    monkeypatch.setattr(lab, "command_doctor", lambda require_idle_gpu=False: 0)
    monkeypatch.setattr(lab, "command_test", lambda: 0)
    monkeypatch.setattr(lab.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, args=args[0]))

    assert lab.command_diagnose("ceiling", "cpu") == 0
    assert written[-1]["development"]["active_experiment"]["status"] == "complete"

    script.write_text("print('changed')\n", encoding="utf-8")
    written.clear()
    assert lab.command_diagnose("ceiling", "cpu") == 2
    assert written == []
