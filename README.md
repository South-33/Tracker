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
| CONSUMED_HOLDOUT | 0004, 0005, 0007, 0010, 0096, 0014, 0019, 0035, 0047, 0063, 0073, 0077, 0081, 0090, 0097 | historical/final evidence only |
| RESERVED_HOLDOUT | none | define a new sealed set before the next final comparison |

Never tune on consumed holdout data. The 10-sequence final set was evaluated on
2026-10-01 and is now permanently consumed.

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

Current tracker settings use a 0.05 detector/recovery floor, 0.45 birth
threshold, 100-corner GMC, guarded learned association, and one frame of
output-only Kalman coast only when at least 10 tracks are active.

Fresh 10-sequence final comparison:

| Metric | Current tracker | Stock YOLO + BoT-SORT |
|---|---:|---:|
| Macro HOTA | **31.84** | 29.21 |
| Macro AssA | **18.51** | 15.65 |
| Macro IDF1 | **31.79** | 28.11 |
| Macro recall | **74.97%** | 73.76% |
| Macro precision | 89.65% | **90.25%** |
| ID switches / 1k frames | **140.6** | 158.3 |

The gain survives all density buckets: HOTA is **35.01 vs 34.38** sparse,
**31.43 vs 28.22** medium, and **28.41 vs 25.77** dense. Sparse identity is not
uniformly better, so do not treat the model as solved; the strongest
generalization gain is still association in medium/dense scenes.

Next benchmark expansion should stay separate from training:

- MOT17 for general pedestrian tracking across different cameras/scenes.
- CrowdHuman for detector recall and heavy-occlusion stress.

## 3. Freeze and final

`runs/person-tracker.pt` is the promoted canonical artifact.

Rebuild it from the current validated heads/config:

~~~powershell
.\.venv\Scripts\python.exe scripts/package_tracker.py
~~~

Before the **next** final comparison, define a new RESERVED_HOLDOUT in
`src/tracker/splits.py` before running inference or metrics. Then freeze the
candidate and score it once with:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_suite.py RUN_FOLDER --split final --confirm-final
~~~

The current 10-sequence final outputs live under
`runs/final-fresh-current-complete` and
`runs/final-fresh-baseline-complete`. They are consumed evidence and must not
guide new tuning.

Reproduce that report with:

~~~powershell
.\.venv\Scripts\python.exe scripts/benchmark_suite.py runs/final-fresh-current-complete --split historical-final --confirm-historical
~~~

The older five-sequence holdout is separately reproducible with
`--split historical-legacy --confirm-historical`.

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
