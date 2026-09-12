"""One front door for the research loop: inspect cheaply, count learned experiments, then promote or kill."""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import statistics
import re
import psutil

from .research_cycle import VALID_PROBE_KINDS, load_ledger, status_text


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "research-cycle.json"
CPU_POLL_SECONDS = 0.5
USER_IDLE_BOOST_SECONDS = 90.0
ACTIVE_CPU_MAX_PERCENT = 65.0
ACTIVE_CPU_RESUME_PERCENT = 45.0
ACTIVE_CPU_CORES = max(1, min(2, os.cpu_count() or 2))
IDLE_CPU_MAX_PERCENT = 92.0
IDLE_CPU_RESUME_PERCENT = 80.0
IDLE_CPU_CORES = max(ACTIVE_CPU_CORES, min(8, max(1, (os.cpu_count() or 2) - 2)))
CPU_THREAD_BUDGET = IDLE_CPU_CORES


def managed_environment() -> dict[str, str]:
    """Bound numerical libraries; affinity narrows them further while the user is active."""
    env = os.environ.copy()
    threads = str(CPU_THREAD_BUDGET)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = threads
    return env


def user_idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input on Windows, otherwise unknown."""
    if os.name != "nt":
        return None

    class LastInputInfo(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LastInputInfo()
    info.cbSize = ctypes.sizeof(info)
    try:
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        now = ctypes.windll.kernel32.GetTickCount()
    except (AttributeError, OSError):
        return None
    return max(0.0, float((now - info.dwTime) & 0xFFFFFFFF) / 1000.0)


def resource_profile(idle_seconds: float | None) -> dict[str, float | int | str]:
    """Choose an interactive or idle budget without requiring configuration."""
    if idle_seconds is not None and idle_seconds >= USER_IDLE_BOOST_SECONDS:
        return {
            "mode": "idle",
            "cores": IDLE_CPU_CORES,
            "max_cpu": IDLE_CPU_MAX_PERCENT,
            "resume_cpu": IDLE_CPU_RESUME_PERCENT,
        }
    return {
        "mode": "interactive",
        "cores": ACTIVE_CPU_CORES,
        "max_cpu": ACTIVE_CPU_MAX_PERCENT,
        "resume_cpu": ACTIVE_CPU_RESUME_PERCENT,
    }


def _process_tree(process: subprocess.Popen) -> list[psutil.Process]:
    try:
        root = psutil.Process(process.pid)
        return [root, *root.children(recursive=True)]
    except (psutil.Error, ProcessLookupError):
        return []


def _set_research_priority(process: subprocess.Popen, profile: dict[str, float | int | str]) -> None:
    """Prefer interactive apps, then widen the research slice only while idle."""
    for item in _process_tree(process):
        try:
            if os.name == "nt":
                priority = (
                    psutil.BELOW_NORMAL_PRIORITY_CLASS
                    if profile["mode"] == "idle"
                    else psutil.IDLE_PRIORITY_CLASS
                )
                item.nice(priority)
                logical = psutil.cpu_count(logical=True) or 1
                width = min(int(profile["cores"]), logical)
                item.cpu_affinity(list(range(logical - width, logical)))
                try:
                    item.ionice(psutil.IOPRIO_VERYLOW)
                except (AttributeError, psutil.Error, PermissionError):
                    pass
            else:
                item.nice(10 if profile["mode"] == "idle" else 19)
        except (psutil.Error, PermissionError):
            pass


def _suspend_research(process: subprocess.Popen) -> None:
    # Children first prevents a busy worker from escaping while its parent sleeps.
    for item in reversed(_process_tree(process)):
        try:
            item.suspend()
        except psutil.Error:
            pass


def _resume_research(process: subprocess.Popen) -> None:
    for item in _process_tree(process):
        try:
            item.resume()
        except psutil.Error:
            pass


def run_managed(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    """Run lab work politely and adapt to whether the machine is being used."""
    process = subprocess.Popen(args, cwd=cwd, env=managed_environment())
    profile = resource_profile(user_idle_seconds())
    _set_research_priority(process, profile)
    psutil.cpu_percent(interval=None)
    suspended = False
    announced_pause = False
    last_mode = str(profile["mode"])
    print(
        f"resource guard: {last_mode} mode, {profile['cores']} cores, "
        f"CPU target < {profile['max_cpu']:.0f}%",
        flush=True,
    )
    try:
        while process.poll() is None:
            total_cpu = psutil.cpu_percent(interval=CPU_POLL_SECONDS)
            profile = resource_profile(user_idle_seconds())
            mode = str(profile["mode"])
            if mode != last_mode:
                last_mode = mode
                print(
                    f"resource guard: switched to {mode} mode, {profile['cores']} cores, "
                    f"CPU target < {profile['max_cpu']:.0f}%",
                    flush=True,
                )
            if not suspended:
                _set_research_priority(process, profile)

            if not suspended and total_cpu >= float(profile["max_cpu"]):
                _suspend_research(process)
                suspended = True
                if not announced_pause:
                    print(
                        f"resource guard: pausing research at {total_cpu:.0f}% system CPU",
                        flush=True,
                    )
                    announced_pause = True
            elif suspended and total_cpu <= float(profile["resume_cpu"]):
                _resume_research(process)
                suspended = False
                announced_pause = False
                _set_research_priority(process, profile)
                print(
                    f"resource guard: resuming research at {total_cpu:.0f}% system CPU",
                    flush=True,
                )
        return subprocess.CompletedProcess(args, int(process.returncode))
    except BaseException:
        if suspended:
            _resume_research(process)
        if process.poll() is None:
            process.terminate()
        raise
    finally:
        if suspended:
            _resume_research(process)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def big_run_input_issues(ledger: dict) -> list[str]:
    big = ledger.get("big_run")
    if not big:
        return []
    issues = []
    for label in ("initialize", "manifest"):
        spec = big[label]
        path = ROOT / spec["path"]
        if not path.is_file():
            issues.append(f"missing {label}: {spec['path']}")
            continue
        actual = sha256(path)
        if actual != spec["sha256"]:
            issues.append(
                f"{label} hash changed: expected {spec['sha256']}, got {actual}"
            )

    manifest_path = ROOT / big["manifest"]["path"]
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_sequences = {
            record["sequence"] for record in manifest["records"]
            if record.get("split") == "train"
        }
        if "dancetrack0096" in train_sequences:
            issues.append("dancetrack0096 leaked into the training manifest")
    else:
        train_sequences = set()

    for label in ("hard_events", "gap_events"):
        spec = big.get(label)
        if not spec:
            continue
        path = ROOT / spec["path"]
        if not path.is_file():
            issues.append(f"missing {label}: {spec['path']}")
            continue
        actual = sha256(path)
        if actual != spec["sha256"]:
            issues.append(f"{label} hash changed: expected {spec['sha256']}, got {actual}")
            continue
        events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
        event_sequences = {event.get("sequence") for event in events}
        if "dancetrack0096" in event_sequences:
            issues.append(f"dancetrack0096 leaked into {label}")
        outside = sorted(sequence for sequence in event_sequences if sequence and sequence not in train_sequences)
        if outside:
            issues.append(f"{label} references non-training sequences: {', '.join(outside)}")

    mix = big.get("curriculum")
    if mix:
        total = sum(float(mix[key]) for key in (
            "hard_probability", "gap_probability", "ordinary_probability"
        ))
        if abs(total - 1.0) > 1e-8:
            issues.append(f"curriculum probabilities sum to {total}, expected 1.0")
    training = big["training"]
    if "unroll" in training and not 32 <= int(training["unroll"]) <= 48:
        issues.append("frozen recurrent big-run unroll must be 32-48 frames")
    if int(big["architecture"].get("candidate_count", 0)) <= 0:
        issues.append("big-run candidate_count must be positive")
    script = big.get("script")
    if not script or not (ROOT / script).is_file():
        issues.append(f"missing big-run script: {script}")
    return issues




def probe_input_issues(ledger: dict) -> list[str]:
    probe = ledger.get("probe")
    if not probe:
        return []
    issues = []
    labels = ["checkpoint", "manifest"]
    if probe.get("weights"):
        labels.append("weights")
    for label in labels:
        spec = probe.get(label, {})
        path = ROOT / spec.get("path", "")
        if not path.is_file():
            issues.append(f"missing probe {label}: {spec.get('path')}")
            continue
        actual = sha256(path)
        if actual != spec.get("sha256"):
            issues.append(f"probe {label} hash changed: expected {spec.get('sha256')}, got {actual}")
    split = probe.get("split")
    if split != "dev":
        issues.append("formal architecture probes must stay on non-gate dev split")
    script = probe.get("script")
    if not script or not (ROOT / script).is_file():
        issues.append(f"missing probe script: {script}")
    elif probe.get("script_sha256") and sha256(ROOT / script) != probe["script_sha256"]:
        issues.append("probe script hash changed")
    manifest_path = ROOT / probe.get("manifest", {}).get("path", "")
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
        dev = {row.get("sequence") for row in records if row.get("split") == "dev"}
        gate = ledger.get("promotion", {}).get("gate_sequence", "dancetrack0096")
        if gate in dev:
            issues.append(f"gate sequence {gate} leaked into probe dev split")
        if not dev:
            issues.append("probe manifest has no dev sequences")
    return issues

def evaluation_issues(ledger: dict) -> list[str]:
    evaluation = ledger.get("evaluation")
    if ledger.get("status") == "evaluation_required" and not evaluation:
        return ["evaluation_required cycle has no frozen evaluation protocol"]
    if not evaluation:
        return []
    issues = []
    for label in ("checkpoint", "manifest"):
        spec = evaluation.get(label, {})
        path = ROOT / spec.get("path", "")
        if not path.is_file():
            issues.append(f"missing evaluation {label}: {spec.get('path')}")
            continue
        actual = sha256(path)
        if actual != spec.get("sha256"):
            issues.append(
                f"evaluation {label} hash changed: expected {spec.get('sha256')}, got {actual}"
            )
    if evaluation.get("split") != "dev":
        issues.append("mechanism evaluation must stay on non-gate dev split")
    script = evaluation.get("script")
    if not script or not (ROOT / script).is_file():
        issues.append(f"missing evaluation script: {script}")
    manifest = evaluation.get("manifest", {})
    big = ledger.get("big_run", {})
    if big and manifest != big.get("manifest"):
        issues.append("evaluation manifest must be the frozen clean training/dev manifest")
    manifest_path = ROOT / manifest.get("path", "")
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
        dev = {row.get("sequence") for row in records if row.get("split") == "dev"}
        gate = big.get("promotion", {}).get("gate_sequence", "dancetrack0096")
        if gate in dev:
            issues.append(f"gate sequence {gate} leaked into dev evaluation split")
        if not dev:
            issues.append("evaluation manifest has no dev sequences")
    return issues


def development_issues(ledger: dict) -> list[str]:
    """Keep the single active train-only development question reproducible."""
    active = ledger.get("development", {}).get("active_experiment")
    if not active:
        return []
    issues = []
    script = ROOT / active.get("script", "")
    if not script.is_file():
        issues.append(f"missing active development script: {active.get('script')}")
        return issues
    expected = active.get("script_sha256")
    if expected:
        actual = sha256(script)
        if actual != expected:
            issues.append(f"active development script hash changed: expected {expected}, got {actual}")
    name = active.get("name")
    if name:
        issues.extend(diagnostic_issues(name))
    if not active.get("question"):
        issues.append("active development must state the decision question")
    return issues

def source_lock_issues(ledger: dict, root: Path = ROOT) -> list[str]:
    """Refuse resume/evaluation if frozen mechanism source changed."""
    issues = []
    for relative, expected in ledger.get("source_lock", {}).items():
        path = root / relative
        if not path.is_file():
            issues.append(f"missing source-locked file: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            issues.append(f"source drift: {relative}; expected {expected}, got {actual}")
    return issues


def context_issues(ledger: dict, root: Path = ROOT) -> list[str]:
    """Catch files/text that can poison the next agent's working context."""
    issues = []
    if not (root / "AGENTS.md").is_file():
        issues.append("missing AGENTS.md project working rules")

    tracker = root / "tracker.md"
    if not tracker.is_file():
        issues.append("missing tracker.md scientific source of truth")
        return issues
    text = tracker.read_text(encoding="utf-8")
    marker = f"<!-- cycle-status: {ledger['status']} -->"
    if marker not in text:
        issues.append(
            f"tracker.md cycle marker disagrees with ledger; expected {marker}"
        )
    if re.search(r"\*\*Status:", text):
        issues.append("tracker.md duplicates mutable cycle status in prose; keep status only in the cycle marker/ledger")
    if tracker.stat().st_size > 24 * 1024:
        issues.append(
            f"tracker.md is {tracker.stat().st_size / 1024:.1f} KiB; compress stale context below 24 KiB"
        )
    return issues


def sync_cycle_marker(ledger: dict, root: Path = ROOT) -> bool:
    """Repair the tracker status marker from the ledger; this is bookkeeping, not science."""
    tracker = root / "tracker.md"
    if not tracker.is_file():
        return False
    text = tracker.read_text(encoding="utf-8")
    desired = f"<!-- cycle-status: {ledger['status']} -->"
    updated, count = re.subn(r"<!-- cycle-status: [^>]+ -->", desired, text, count=1)
    if count and updated != text:
        tracker.write_text(updated, encoding="utf-8")
        return True
    return False


def active_gpu_processes(min_sm=15) -> list[dict]:
    """Return processes doing material GPU work, ignoring incidental desktop rendering."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    active = []
    for line in result.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            pid = int(fields[1])
            sm = int(fields[3]) if fields[3] != "-" else 0
        except ValueError:
            continue
        if pid == os.getpid() or sm < min_sm:
            continue
        active.append({"pid": pid, "sm": sm, "command": fields[-1]})
    return sorted(active, key=lambda row: row["sm"], reverse=True)


def progress_rows(big: dict) -> list[dict]:
    path = ROOT / big["output"] / "loss.jsonl"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The trainer appends one complete JSON object at a time. Ignore a
                # partially visible final line if status races an append.
                continue
    return rows


def progress_text(big: dict, window: int = 50) -> list[str]:
    rows = progress_rows(big)
    total = int(big["training"]["steps"])
    if not rows:
        return [f"progress: 0/{total}"]
    recent = rows[-window:]
    mean = lambda key: statistics.fmean(float(row[key]) for row in recent if row.get(key) is not None)
    lines = [f"progress: {int(rows[-1]['step'])}/{total}"]
    if "recovery_hazard_veto_recall" in recent[-1]:
        lines.append(
            f"recent {len(recent)}: write_precision={mean('accepted_write_precision'):.3f} "
            f"write_recall={mean('visible_write_recall'):.3f} "
            f"local_accept={mean('local_safe_acceptance'):.3f} "
            f"local_veto={mean('local_hazard_veto_recall'):.3f} "
            f"recovery_accept={mean('recovery_safe_acceptance'):.3f} "
            f"recovery_veto={mean('recovery_hazard_veto_recall'):.3f} "
            f"wrong={mean('wrong_write_rate'):.5f} "
            f"strict={mean('clip_strict_survival'):.3f}"
        )
        latest_epoch = int(rows[-1].get("epoch", -1))
        epoch_rows = [row for row in rows if int(row.get("epoch", -2)) == latest_epoch]
        first_by_sequence = []
        for sequence in dict.fromkeys(row.get("sequence") for row in epoch_rows):
            sequence_rows = [row for row in epoch_rows if row.get("sequence") == sequence]
            first_by_sequence.extend(sequence_rows[:2])
        if first_by_sequence:
            fresh_mean = lambda key: statistics.fmean(
                float(row[key]) for row in first_by_sequence if row.get(key) is not None
            )
            lines.append(
                f"fresh-start epoch {latest_epoch}: n={len(first_by_sequence)} "
                f"precision={fresh_mean('accepted_write_precision'):.3f} "
                f"recall={fresh_mean('visible_write_recall'):.3f} "
                f"strict={fresh_mean('clip_strict_survival'):.3f}"
            )
    elif "visible_branch_coverage" in recent[-1]:
        nonzero = sum(float(row["clip_strict_survival"]) > 0 for row in recent)
        lines.append(
            f"recent {len(recent)}: branch={mean('visible_branch_coverage'):.3f} "
            f"top1={mean('visible_top1_accuracy'):.3f} "
            f"noobs={mean('absent_branch_noobs'):.3f} "
            f"strict={mean('clip_strict_survival'):.3f} "
            f"strict_nonzero={nonzero}/{len(recent)}"
        )
    elif "visible_action_accuracy" in recent[-1]:
        lines.append(
            f"recent {len(recent)}: action={mean('visible_action_accuracy'):.3f} "
            f"write_precision={mean('accepted_write_precision'):.3f} "
            f"write_recall={mean('visible_write_recall'):.3f} "
            f"wrong={mean('wrong_write_rate'):.3f} "
            f"defer={mean('absent_defer_accuracy'):.3f} "
            f"strict={mean('clip_strict_survival'):.3f}"
        )
    elif "hazard_veto_recall" in recent[-1]:
        useful = sum(float(row["visible_commit_recall"]) > .55 for row in recent)
        wrong = sum(float(row["wrong_write_rate"]) > 0 for row in recent)
        lines.append(
            f"recent {len(recent)}: write_precision={mean('accepted_write_precision'):.3f} "
            f"write_recall={mean('visible_commit_recall'):.3f} "
            f"local={mean('local_link_coverage'):.3f} "
            f"safe_accept={mean('safe_link_acceptance'):.3f} "
            f"hazard_veto={mean('hazard_veto_recall'):.3f} "
            f"hazard_rate={mean('hazard_link_rate'):.3f} "
            f"wrong={mean('wrong_write_rate'):.5f} "
            f"strict={mean('clip_strict_survival'):.3f} "
            f"useful={useful}/{len(recent)} wrong_clips={wrong}/{len(recent)}"
        )
    elif "existing_visible_accuracy" in recent[-1]:
        lines.append(
            f"recent {len(recent)}: existing={mean('existing_visible_accuracy'):.3f} "
            f"absent={mean('existing_absent_accuracy'):.3f} "
            f"proposal={mean('proposal_accuracy'):.3f} "
            f"new={mean('new_recall'):.3f} bg={mean('background_accuracy'):.3f}"
        )
    cache = ROOT / big["training"]["feature_cache"]
    if cache.is_dir():
        files = [path for path in cache.rglob("*.pt") if path.is_file()]
        bytes_used = sum(path.stat().st_size for path in files)
        free = shutil.disk_usage(ROOT).free
        lines.append(
            f"feature cache: {len(files)} frames, {bytes_used / 2**30:.1f} GiB; "
            f"disk free {free / 2**30:.1f} GiB"
        )
    return lines


def atomic_write_ledger(data: dict) -> None:
    temporary = LEDGER.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER)
    tracker = ROOT / "tracker.md"
    if tracker.is_file():
        text = tracker.read_text(encoding="utf-8")
        text = re.sub(r"<!-- cycle-status: [^>]+ -->", f"<!-- cycle-status: {data['status']} -->", text, count=1)
        tracker.write_text(text, encoding="utf-8")


def command_status() -> int:
    ledger = load_ledger()
    sync_cycle_marker(ledger)
    print(status_text(ledger))
    big = ledger.get("big_run")
    if big:
        for line in progress_text(big):
            print(line)
    return 0


def command_doctor(require_idle_gpu=False) -> int:
    ledger = load_ledger()
    sync_cycle_marker(ledger)
    issues = (
        big_run_input_issues(ledger)
        + probe_input_issues(ledger)
        + evaluation_issues(ledger)
        + development_issues(ledger)
        + source_lock_issues(ledger)
        + context_issues(ledger)
    )
    active = active_gpu_processes()
    if active:
        description = ", ".join(
            f"{row['command']} pid={row['pid']} sm={row['sm']}%" for row in active
        )
        message = f"GPU active: {description}"
        if require_idle_gpu:
            issues.append(message)
        else:
            print(f"NOTE {message}")
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 2
    print("OK frozen inputs, hashes, source lock, no0096 boundary, curriculum, architecture, evaluation protocol, context hygiene")
    if not active:
        print("OK GPU appears idle")
    return 0


def command_test() -> int:
    result = run_managed([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    return int(result.returncode)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def learned_experiment_signals(text: str) -> list[str]:
    """Identify scripts that learn parameters and therefore consume a formal probe slot."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["script does not parse"]
    signals = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name.endswith(".backward") or name in {"torch.autograd.grad", "autograd.grad"}:
            signals.add("gradient computation")
        if name.startswith("torch.optim.") or name.startswith("optim."):
            signals.add("optimizer construction")
        if name.endswith("requires_grad_") and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is True:
            signals.add("trainable parameters")
        if name in {"torch.nn.Parameter", "nn.Parameter"}:
            signals.add("trainable parameter creation")
    return sorted(signals)


def diagnostic_issues(name: str, root: Path = ROOT) -> list[str]:
    """Keep train-only scratch work inside the contamination boundary."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        return [f"invalid diagnostic name: {name!r}"]
    script = root / "runs" / f"_diag_{name}" / "probe.py"
    if not script.is_file():
        return [f"missing diagnostic script: {script.relative_to(root)}"]
    text = script.read_text(encoding="utf-8")
    issues = []
    if "manifest-train-full-dense-no0096.json" not in text:
        issues.append("diagnostic must use the frozen train-only no0096 manifest")
    try:
        tree = ast.parse(text)
        literal_strings = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
    except SyntaxError:
        literal_strings = []
        issues.append("diagnostic script does not parse")
    if any("dancetrack0096" in value for value in literal_strings):
        issues.append("diagnostic executable strings reference the architecture gate dancetrack0096")
    if not re.search(r"load_sequences\([^)]*['\"]train['\"]\)", text, re.S):
        issues.append("diagnostic must load only the train split")
    return issues


def command_diagnose(name: str, device: str) -> int:
    issues = diagnostic_issues(name)
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 2
    ledger = load_ledger()
    script = ROOT / "runs" / f"_diag_{name}" / "probe.py"
    learned = learned_experiment_signals(script.read_text(encoding="utf-8"))
    active = ledger.get("development", {}).get("active_experiment")
    if learned:
        if not active or active.get("name") != name:
            print(
                "FAIL learned train-only development needs one matching development.active_experiment "
                "with a question before launch"
            )
            return 2
    if active and active.get("name") == name:
        expected = active.get("script_sha256")
        actual = sha256(script)
        if expected and expected != actual:
            print(f"FAIL active development script hash changed: expected {expected}, got {actual}")
            return 2
        if not expected:
            active["script_sha256"] = actual
            atomic_write_ledger(ledger)
    elif active and active.get("status") == "running" and active.get("name") != name:
        print(f"FAIL active development {active.get('name')} is already running; finish it first")
        return 2
    if command_doctor(require_idle_gpu=str(device).lower().startswith("cuda")):
        print("diagnostic not started; fix doctor failures first")
        return 2
    if command_test():
        print("diagnostic not started; tests failed")
        return 2
    result = run_managed([sys.executable, str(script), "--device", device], cwd=ROOT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    finished = load_ledger()
    active = finished.get("development", {}).get("active_experiment")
    if active and active.get("name") == name:
        active["status"] = "complete"
        atomic_write_ledger(finished)
    print("diagnostic complete; interpret/archive the result, then clear the active development and scratch")
    return 0


def record_probe_result(ledger: dict, probe: dict) -> dict:
    """Record one frozen probe output and consume exactly one experiment slot."""
    output = ROOT / probe["output"]
    if not output.is_file():
        raise FileNotFoundError(f"probe completed without required output: {probe['output']}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    kind = probe["kind"]
    if any(row["kind"] == kind for row in ledger["probes"]):
        raise ValueError(f"probe kind already recorded: {kind}")
    completed = {
        "kind": kind,
        "status": "complete",
        "question": probe.get("question", ""),
        "scope": probe.get("scope", probe.get("split", "")),
        "output": probe["output"],
        "decision": payload.get("decision", "measured"),
    }
    ledger["probes"].append(completed)
    ledger.pop("probe", None)
    if len(ledger["probes"]) >= ledger["probe_budget"]:
        ledger["status"] = "decision_required"
    atomic_write_ledger(ledger)
    return completed


def command_record_probe() -> int:
    """Record an already-running/adopted probe without rerunning it."""
    ledger = load_ledger()
    probe = ledger.get("probe")
    if not probe:
        print("no active probe to record")
        return 2
    issues = probe_input_issues(ledger)
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 2
    completed = record_probe_result(ledger, probe)
    remaining = ledger["probe_budget"] - len(ledger["probes"])
    print(f"probe recorded: {completed['kind']} -> {completed['decision']}; {remaining} slots remain")
    return 0




def command_probe(kind: str, device: str) -> int:
    ledger = load_ledger()
    from .research_cycle import action_allowed
    allowed, message = action_allowed(ledger, "probe", kind)
    if not allowed:
        print(message)
        return 2
    probe = ledger.get("probe", {})
    if probe.get("kind") != kind:
        print(f"frozen probe is {probe.get('kind')}; refusing {kind}")
        return 2
    output = ROOT / probe.get("output", "")
    if output.is_file():
        print(
            f"refusing to rerun completed probe output {probe['output']}; "
            "interpret and record the result in research-cycle.json first"
        )
        return 2
    if command_doctor(require_idle_gpu=True):
        print("probe not started; fix doctor failures first")
        return 2
    if command_test():
        print("probe not started; tests failed")
        return 2
    result = run_managed([sys.executable, probe["script"], "--device", device], cwd=ROOT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    finished = load_ledger()
    completed = record_probe_result(finished, finished["probe"])
    print(
        f"probe recorded automatically: {kind} -> {completed['decision']}; "
        f"{finished['probe_budget'] - len(finished['probes'])} slots remain"
    )
    return 0

def command_eval(device: str) -> int:
    ledger = load_ledger()
    if ledger["status"] != "evaluation_required":
        print(f"refusing evaluation while cycle status is {ledger['status']}")
        return 2
    evaluation = ledger.get("evaluation", {})
    if evaluation.get("status") == "complete" and (ROOT / evaluation["output"]).is_file():
        print(f"evaluation already complete: {evaluation['output']}")
        return 0
    if command_doctor(require_idle_gpu=True):
        print("evaluation not started; fix doctor failures first")
        return 2
    if command_test():
        print("evaluation not started; tests failed")
        return 2
    evaluation["status"] = "running"
    atomic_write_ledger(ledger)
    try:
        result = run_managed(
            [sys.executable, evaluation["script"], "--device", device], cwd=ROOT,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
    except BaseException:
        failed = load_ledger()
        failed["evaluation"]["status"] = "required"
        atomic_write_ledger(failed)
        raise
    finished = load_ledger()
    finished["evaluation"]["status"] = "complete"
    atomic_write_ledger(finished)
    print(f"mechanism evaluation complete: {finished['evaluation']['output']}")
    return 0

def command_run(device: str) -> int:
    ledger = load_ledger()
    if not ledger.get("big_run") or ledger["status"] not in {"big_run_required", "big_run_running"}:
        print(f"refusing run while cycle status is {ledger['status']}")
        return 2
    if command_doctor(require_idle_gpu=True):
        print("run not started; fix doctor failures first")
        return 2
    if command_test():
        print("run not started; tests failed")
        return 2

    previous = copy.deepcopy(ledger)
    ledger["status"] = "big_run_running"
    ledger["big_run"]["status"] = "running"
    atomic_write_ledger(ledger)
    try:
        result = run_managed(
            [sys.executable, ledger["big_run"]["script"], "--device", device],
            cwd=ROOT,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
    except BaseException:
        previous["status"] = "big_run_required"
        previous["big_run"]["status"] = "required"
        atomic_write_ledger(previous)
        raise

    finished = load_ledger()
    finished["status"] = "evaluation_required"
    finished["big_run"]["status"] = "complete"
    template = finished["big_run"].get("post_run_evaluation")
    if template:
        evaluation = copy.deepcopy(template)
        checkpoint = ROOT / evaluation["checkpoint"]["path"]
        evaluation["checkpoint"]["sha256"] = sha256(checkpoint)
        evaluation["status"] = "required"
        finished["evaluation"] = evaluation
    atomic_write_ledger(finished)
    print("big run complete; cycle advanced to evaluation_required")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the research contract and current progress")
    sub.add_parser("doctor", help="check frozen inputs, contamination, and GPU activity")
    sub.add_parser("test", help="run the small active test suite")
    diagnose = sub.add_parser("diagnose", help="run one guarded train-only diagnostic under runs/_diag_<name>/probe.py")
    diagnose.add_argument("name")
    diagnose.add_argument("--device", default="cuda")
    probe = sub.add_parser("probe", help="run one frozen decision-changing experiment and account for it automatically")
    probe.add_argument("--kind", required=True, choices=VALID_PROBE_KINDS)
    probe.add_argument("--device", default="cuda")
    sub.add_parser("record-probe", help="record the frozen output of an already-running active probe")
    evaluate = sub.add_parser("eval", help="run the frozen uncut non-gate mechanism evaluation")
    evaluate.add_argument("--device", default="cuda")
    run = sub.add_parser("run", help="resume or launch the one allowed big run")
    run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "status":
        return command_status()
    if args.command == "doctor":
        return command_doctor()
    if args.command == "test":
        return command_test()
    if args.command == "diagnose":
        return command_diagnose(args.name, args.device)
    if args.command == "probe":
        return command_probe(args.kind, args.device)
    if args.command == "record-probe":
        return command_record_probe()
    if args.command == "eval":
        return command_eval(args.device)
    return command_run(args.device)


if __name__ == "__main__":
    raise SystemExit(main())
