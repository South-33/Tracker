# Tracker

Research toward one stateful person tracker with persistent anonymous IDs through crossings, occlusion and short exits. Laptop training uses an RTX 4060; the deployment target is Orin Nano Super 8 GB at >=15 FPS. **No current model has passed acceptance.**

Start with [GOAL.md](GOAL.md) for the goal and [tracker.md](tracker.md) for current evidence. Agents also read [AGENTS.md](AGENTS.md).

```powershell
Set-Location D:\Project\tracker
.\lab.ps1 status
.\lab.ps1 doctor
.\lab.ps1 test
```

The existing environment is `.venv\Scripts\python.exe`. Data and weights remain local and ignored. The setup pins and upstream patching live in `pyproject.toml`, `requirements-laptop.txt` and `scripts/bootstrap.py`; do not reinstall working GPU dependencies routinely.

| Task | Command |
|---|---|
| Run the declared train-only experiment | `.\lab.ps1 diagnose <name>` |
| Run a declared formal dev experiment | `.\lab.ps1 probe --kind mechanism` (also `reality`, `break-it`) |
| Launch/resume the declared substantial run | `.\lab.ps1 run` |
| Evaluate the declared trained mechanism | `.\lab.ps1 eval` |
| Score saved MOT predictions | `.\lab.ps1 score --run runs/<replay> --manifest data/<manifest>.json` |
| Verify/time annotation preparation | `.\lab.ps1 benchmark-data` (CPU only, training annotations) |

Commands require the matching plan in `research-cycle.json`; `status` states the next action. A diagnostic's source belongs in `experiments/<name>/probe.py`, accepts `--device`, and writes the plan's explicit JSON output. The old `runs/_diag_<name>/probe.py` location is still readable for migration. All exploratory diagnostics count toward the same three-experiment limit. Tests and scoring existing predictions do not consume a scientific probe.

To declare a diagnostic, set cycle `status` to `probing` and set `development.active_experiment` to a record like this. Fill in a real question, useful expected gain and bounded compute before launching; the lab records the source hash automatically.

```json
{
  "name": "your-experiment",
  "script": "experiments/your-experiment/probe.py",
  "output": "runs/your-experiment/result.json",
  "status": "planned",
  "question": "Which specific uncertainty changes the substantial bet?",
  "expected_system_gain": "Why this could move the tracker toward 80%",
  "compute_budget": "Planned examples/rollout duration/steps and laptop wall time",
  "decision_rule": "The observed result that would justify scaling or changing direction"
}
```

After three exploratory results, `diagnose` and `probe` refuse another small run. Declare the substantial `big_run` plan and its post-run evaluation in the ledger, or record why a different hypothesis deserves a new cycle. The harness validates the plan fields; do not reset a counter to retry the same question.

`score` expects `<sequence>.txt` in MOT format and `run.json` with `sequence`, `frames`, exact `manifest_sha256`, and checkpoint/source provenance. It reports official HOTA/IDF1 plus version-2 coverage diagnostics. Predict every frame in a full causal replay; an empty prediction frame is still evaluated. Sequence-prefix results are labeled as such.

| Folder | Responsibility |
|---|---|
| `src/tracker/` | Reusable detector, state, data, evaluation and lab code |
| `experiments/` | Current executable research recipes; create only when declaring a run |
| `scripts/` | Dataset/bootstrap and official evaluation entry points |
| `tests/` | Behavioral regression checks |
| `evidence/attempts/` | Automatically saved source/config snapshots and outcomes, including failures |
| `evidence/legacy/` | Historical results with limited reproducibility; consult on demand |
| `runs/`, `data/`, `weights/`, `third_party/` | Ignored local artifacts and dependencies |

The lab saves source before research execution and copies small JSON results afterward. Commit these records with the decision. Before research launches it reuses a passing unit suite only when Python source, upstream code and dependency metadata still match; dynamic input/ledger checks run every time. Explicit `lab test` always reruns the suite. `status` reads a bounded log tail and does not walk the feature cache. Heavy jobs inherit the activity-aware laptop resource guard. This repository is not a deployed SmartCampus integration.
