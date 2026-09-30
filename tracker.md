# Tracker

## Goal

Build one small, causal neural person tracker that does detection and tracking in one model.

```text
current frame + bounded memory
    -> person boxes + confidence + anonymous track continuity + updated memory
```

We do not need real identities. We only need to keep the same anonymous person together while visible, through crossings/occlusion, and for a bounded time after they disappear so they can be recovered on re-entry. If someone is gone beyond the memory horizon, creating a new track is fine.

Target: one camera at **>=15 FPS on Orin Nano Super 8 GB**.

## Current bet

Start from official pretrained **YOLO26n**. Do not retrain basic vision from scratch yet.

The current diagnostic model freezes YOLO26n and trains only a small identity embedding from detector features:

```text
YOLO26n
  -> person box + confidence
  -> 64D tracking embedding
```

BoT-SORT is currently the online association harness, not the intended final architecture. It lets us test whether the learned feature contains useful tracking signal without confounding that question with a new lifecycle, motion model, or assignment algorithm.

Sequence identity labels train front/side/back/partial views of the same person to stay compatible while different people separate. The head is trained directly on post-NMS detector boxes with symmetric identity retrieval. Appearance is only computed on dense frames by default.

Do not add recurrent/transformer memory, learned association matrices, or full-detector fine-tuning until this simpler representation/association split is understood.

## Data

- **PersonPath22**: main varied real-world tracking/video data.
- **DanceTrack**: hard motion, crossings and similar-looking people.
- **CrowdHuman and/or COCO person batches**: preserve strong person detection during video fine-tuning.

Local bootstrap data is ready. We have all official PersonPath22 visible annotations/splits plus a CRC-verified 5-video training starter slice chosen for crowding, occlusion and many identities (432 MB). Existing DanceTrack data stays available. Expand PersonPath22 only after the first temporal model shows signal instead of blocking iteration on the full 7.66 GB core-train archive.

## Baseline to beat

Before training the new model, run **YOLO26n + BoT-SORT** on the same frozen development videos and record exact settings here.

Development videos:

```text
dancetrack0016
dancetrack0020
```

Frozen baseline, 2026-09-30:

```text
YOLO26n + Ultralytics BoT-SORT
Ultralytics 8.4.166
640 px, person class only, conf=0.1, iou=0.7
official botsort.yaml defaults, ReID disabled
dev: dancetrack0016 + dancetrack0020 (2,746 frames)

HOTA       32.20   <-- primary number to beat
AssA       19.39
DetA       53.94
IDF1       30.32
ID switches 397
Recall     66.67%
Precision  96.03%

Laptop RTX 4060:
model + tracker             ~31.3 FPS
including JPEG frame reads  ~22.6 FPS
```

The baseline is much weaker at association than detection, which directly supports testing temporal embeddings/context. Do not celebrate a tiny HOTA win; report the full metric set. If our simple model clears this default baseline, run BoT-SORT with ReID enabled as a stronger second reference.

## Current experiment and evidence

### Rejected learned matcher

The first learned-memory probe froze YOLO26n and trained a 64D embedding, learned association head, and absence head. Its held-out choice accuracy rose substantially, but the online tracker regressed badly:

```text
dev: dancetrack0016 + dancetrack0020

official baseline              HOTA 32.20  AssA 19.39  IDF1 30.32  IDSW 397
learned matcher                HOTA 27.84  AssA 14.35  IDF1 26.09  IDSW 1326
```

The learned matcher is rejected. Do not revive it by simply training longer.

### Stronger reference: native YOLO features

Ultralytics 8.4.166 can feed native detector features into BoT-SORT ReID with `model: auto`. On the same two dev sequences this was worse than the frozen no-ReID baseline:

```text
YOLO26n + BoT-SORT native ReID
HOTA 31.01  AssA 17.76  DetA 54.50  IDF1 29.87  IDSW 582
```

So useful appearance information is not available for free from the raw detector feature path.

### Learned 64D identity feature

The active code now trains only the 64D embedding with symmetric identity retrieval on post-NMS detections. Training alternates the five-video PersonPath22 starter slice with complete DanceTrack train sequences `0001/0002/0006/0008/0015`; DanceTrack `0012` is held out. A 500-step probe improved held-out top-1 identity retrieval from **42.63% to 57.67%**.

The final reproducibility rerun was made from clean commit `a094e28` with `repo_dirty: false`. Its checkpoint SHA-256 is `b0fbba5b59cac80c52f584f5328a9c937a9cc184e69a94b47730c1907972c3fc`. The learned embedding tensors are bit-for-bit identical to the earlier head used for the tracking metrics below, so those metrics apply to this clean checkpoint without a second tracking run.

The learned feature is passed to BoT-SORT only when a frame has at least 10 detections. Two comparisons matter:

```text
dev: dancetrack0016 + dancetrack0020

                              HOTA   AssA   DetA   IDF1  IDSW  Frag  Recall  Precision
official model.track baseline 32.20  19.39  53.94  30.32   397   499   66.67     96.03
manual path, no ReID          31.77  18.87  54.01  30.30   427   523   67.25     95.47
manual path, learned 64D      33.12  20.06  55.20  32.14   554   608   69.03     94.45
```

The manual no-ReID row is the causal ablation for the learned feature because it uses the identical square-letterbox detector path and direct BoT-SORT API. Relative to that controlled baseline, the 64D feature adds **+1.35 HOTA, +1.19 AssA, and +1.84 IDF1**. The representation is therefore useful.

It is **not promoted as the incumbent yet**. ID switches rise from 427 to 554 and fragments from 523 to 608. The hard `0020` sequence shows the tradeoff most clearly:

```text
manual no-ReID: HOTA 30.66  AssA 25.38  IDF1 34.05  IDSW 322  Frag 411
learned 64D:    HOTA 32.79  AssA 27.56  IDF1 37.22  IDSW 452  Frag 496
```

On `0016`, learned appearance is almost neutral and slightly reduces switches. The hard sequence is therefore the useful discriminator.

A seeded **untrained** 64D projection is an important control. On `0020` it already reaches HOTA **32.56**, AssA **27.30**, IDF1 **37.10**, with 466 switches and 522 fragments. The trained head reaches HOTA **32.79**, AssA **27.56**, IDF1 **37.22**, with 452 switches and 496 fragments. Training is helping identity stability, but only modestly. Most of the current gain comes from giving BoT-SORT a compact detector-derived appearance feature at all, not from the present retrieval objective.

Two focused follow-ups did not earn their place. Raising retrieval temperature from `0.1` to `1.0` spread cosine scores out but reduced held-out retrieval and `0020` tracking (HOTA 32.63 vs 32.79). Restricting appearance to detections overlapping another box at IoU >=0.3 reduced `0020` switches from 452 to 428 and raised AssA from 27.56 to 27.79, but lowered HOTA to 32.66 and hurt DetA/recall/precision. Keep the simpler temperature-`0.1`, dense-frame gate as the active probe.

### Next research question

Keep the official no-ReID baseline as the incumbent. Keep the simple 64D identity head as the active candidate, but require future embedding changes to beat a seeded random-projection control as well as no-ReID. The next high-information work is to reduce the switch/fragment penalty while making the learned representation earn a larger margin over random projection. Prefer focused experiments over a larger neural matcher. In particular:

1. isolate where ReID changes BoT-SORT assignments on `0020`, especially the new switches;
2. train against harder same-frame/crossing impostors and measure the margin over the random projection;
3. test only conservative association/lifecycle changes justified by those failure cases;
4. only then consider learned assignment, temporal memory, or detector fine-tuning.

Do not touch the untouched DanceTrack sequences until a dev candidate is clearly better across the full metric set.

### Reproduction commands

```powershell
# Official baseline
.\.venv\Scripts\python.exe scripts/baseline.py dancetrack0016 dancetrack0020
.\.venv\Scripts\python.exe scripts/evaluate.py runs/yolo26n-botsort dancetrack0016 dancetrack0020

# Strong native-feature reference
.\.venv\Scripts\python.exe scripts/baseline.py dancetrack0016 dancetrack0020 --reid --output runs/yolo26n-botsort-reid
.\.venv\Scripts\python.exe scripts/evaluate.py runs/yolo26n-botsort-reid dancetrack0016 dancetrack0020

# Train the active 64D identity head
.\.venv\Scripts\python.exe scripts/train.py --steps 500 --temperature 0.1 --output runs/identity-head-v2

# Controlled identical-path baseline
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --disable-reid --output runs/manual-botsort-no-reid
.\.venv\Scripts\python.exe scripts/evaluate.py runs/manual-botsort-no-reid dancetrack0016 dancetrack0020

# Learned feature candidate
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --head runs/identity-head-v2/head.pt --output runs/identity-v2-track
.\.venv\Scripts\python.exe scripts/evaluate.py runs/identity-v2-track dancetrack0016 dancetrack0020
```

## Research loop

The current bet is the incumbent, not something we are obligated to keep modifying. Finish and measure the simplest version before adding capability.

The agent has broad freedom to investigate. It may read literature and implementations, inspect and relabel data, build focused probes, change the representation or training objective, simplify or rewrite code, or conclude that the incumbent should remain untouched. The constraint is evidence, not conservatism: explore widely, but only promote changes that earn their place.

1. **Measure the incumbent.** Reproduce the frozen baseline, train/evaluate the current bet, and keep the same development videos and settings so changes are attributable.
2. **Find a real weakness.** Use HOTA/AssA/DetA/IDF1, ID switches, runtime, and targeted failure review. Decide whether the bottleneck is detection, representation, association, memory, motion, data, or evaluation before proposing a fix.
3. **Run the cheapest discriminating experiment.** Prefer an ablation or narrow change that can prove/disprove the idea. Research papers/implementations when the mechanism is unclear. Do not add complexity because it sounds plausible.
4. **Promote only meaningful wins.** A candidate must improve the relevant tracking capability without hiding a material detection/runtime regression. Tiny metric noise is not enough. If it is not clearly better, keep the incumbent and remove the candidate machinery.
5. **Check that the new capability is real.** Ablate the added memory/context. If removing it barely changes tracking, the model did not learn the intended behavior. Once a candidate is frozen on development, compare against the stronger BoT-SORT+ReID reference and then evaluate the untouched sequences exactly once.
6. **Repeat only when evidence gives a reason.** New data, a reproducible failure family, or a credible research insight can justify another change. If there is no evidence-backed improvement to make, do not touch the model.

The goal is not continuous code churn. It is to preserve the strongest known tracker while taking high-information shots at real capability jumps. Success means the one-network model approaches or beats strong tracking references while keeping useful detection quality and a credible path to >=15 FPS on Nano.

## Working rules

- Keep this repo small. Git history is the archive.
- Delete dead experiments instead of maintaining legacy compatibility.
- Prefer one obvious path over frameworks and abstraction.
- Keep tests only where a silent bug would invalidate training or evaluation.
- Raw datasets live in `data/`, generated runs in `runs/`, and weights in `weights/`; all stay untracked.
- Never claim a result without exact model settings, tracker settings, evaluated sequences and metrics.
