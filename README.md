# Tracker

Small causal person tracker:

~~~text
frame + bounded state -> boxes + confidence + anonymous IDs + updated state
~~~

Architecture:

~~~text
YOLO26n -> 64D ROI appearance -> motion/GMC -> guarded association -> IDs
~~~

Work directly on main. Use Git commits as checkpoints. Do not create normal
research branches or extra worktrees.

## Data roles

src/tracker/splits.py is the only source of truth.

| Role | Sequences | Use |
|---|---|---|
| TRAIN | 0001, 0002, 0006, 0008, 0015 | update learned weights |
| CALIBRATION | 0012 | thresholds and hypothesis selection |
| DEV | 0016, 0020 | normal online hill-climb |
| CONSUMED_HOLDOUT | 0004, 0005, 0007, 0010, 0096 | historical evidence only |
| RESERVED_HOLDOUT | 0082, 0083 | next frozen-candidate final test |

Never tune on either holdout group. The consumed holdout has already been seen.
The local reserved sequences are only partial 120-frame slices, so do not score
them until the complete official sequences are installed.

## 1. Train

Only these commands update learned weights:

~~~powershell
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe scripts/train_owner.py
~~~

Training scripts enforce TRAIN. Validation uses CALIBRATION only.

## 2. Develop

scripts/track.py is the research runner. It accepts only CALIBRATION and DEV.

Frozen incumbent on DEV:

~~~powershell
.\.venv\Scripts\python.exe scripts/run_tracker.py dancetrack0016 dancetrack0020 --tracker runs/person-tracker.pt --output runs/dev-current
.\.venv\Scripts\python.exe scripts/benchmark_suite.py runs/dev-current --split dev
~~~

Calibration is separate:

~~~powershell
.\.venv\Scripts\python.exe scripts/track.py dancetrack0012 --head runs/identity-head-v2/head.pt --owner-head runs/owner-head-v1/head.pt --new-track-threshold 0.45 --track-low-threshold 0.05 --detector-confidence 0.05 --gmc-max-corners 100 --output runs/calibration-current
.\.venv\Scripts\python.exe scripts/benchmark_suite.py runs/calibration-current --split calibration
~~~

Do not combine calibration with the headline DEV score.

For a broad **research diagnostic** across calibration + dev (medium + sparse +
very dense), put all three prediction files in one run folder and use:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_suite.py RUN_FOLDER --split research
~~~

This is useful for understanding behavior across scene types, but it is not a
fresh final score.

Detector threshold changes such as `--detector-confidence 0.05` or
`--track-low-threshold 0.05` are experiments. Do not call them the incumbent
until they beat the frozen model on DEV without using holdout feedback.

Current broad research picture:

| Density | Frozen tracker HOTA | Stock YOLO + BoT-SORT |
|---|---:|---:|
| Sparse | **33.33** | 32.79 |
| Medium | **25.39** | 23.61 |
| Dense | **34.11** | 31.71 |
| Macro | **30.94** | 29.37 |

So the tracker is better across all three research density levels, with the
largest gain in dense crowds. The consumed historical holdout also improves in
macro HOTA (**13.11 vs 12.24**), but its sparse/medium clips are dominated by
very low detector recall and must not guide new tuning.

Next benchmark expansion should stay separate from training:

- MOT17 for general pedestrian tracking across different cameras/scenes.
- CrowdHuman for detector recall and heavy-occlusion stress.

## 3. Freeze and final

runs/person-tracker.pt is the previous frozen artifact. Its old five-sequence
holdout result is historical evidence only.

For the next candidate, complete RESERVED_HOLDOUT first, freeze the candidate,
then score it with:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_suite.py RUN_FOLDER --split final --confirm-final
~~~

Historical reproduction only:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_suite.py runs/protected-final --split historical --confirm-historical
~~~

Never use that historical report to choose a new change.

## 4. Deployment benchmark

Speed testing is separate from model selection:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_tracker.py dancetrack0020 --tracker runs/person-tracker.pt --warmup 30 --min-fps 15 --max-memory-gib 8 --output runs/orin-benchmark.json
~~~

On Orin Nano Super:

~~~bash
python3 scripts/orin_acceptance.py dancetrack0020 --tracker runs/person-tracker.pt --min-fps 15 --max-memory-gib 8 --output runs/orin-benchmark.json
~~~

## Script map

- scripts/train.py: train identity appearance head.
- scripts/train_owner.py: train owner-continuity scorer.
- scripts/track.py: research tracking on calibration/dev only.
- scripts/benchmark_suite.py: quality report for an existing run.
- scripts/package_tracker.py: freeze a validated candidate.
- scripts/run_tracker.py: run a packaged tracker.
- scripts/benchmark_tracker.py: throughput and memory benchmark.
- scripts/orin_acceptance.py: Jetson hardware acceptance.
- scripts/evaluate.py: low-level metrics used by benchmark_suite.

tracker.md is the detailed research notebook. README.md is the operating guide.
