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

The current diagnostic path freezes YOLO26n, ROI-pools its multi-scale detector features at each person box, and optionally compresses them through a small 64D identity head:

```text
YOLO26n frozen pyramid
  -> person box + confidence
  -> ROI-pooled appearance
  -> optional 64D tracking embedding
```

BoT-SORT is currently the online association harness, not the intended final architecture. It lets us test whether the learned feature contains useful tracking signal without confounding that question with a new lifecycle, motion model, or assignment algorithm.

Sequence identity labels train front/side/back/partial views of the same person to stay compatible while different people separate. The head is trained directly on post-NMS detector boxes with symmetric identity retrieval. Appearance is only computed on dense frames by default.

The controls now show that ROI-pooled pretrained appearance is already strong before identity training. Treat the 64D training as an incremental refinement, not the source of the whole appearance gain.

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

The manual no-ReID row is the causal ablation for the full appearance path because it uses the identical square-letterbox detector path and direct BoT-SORT API. Relative to that controlled baseline, the trained 64D feature adds **+1.35 HOTA, +1.19 AssA, and +1.84 IDF1**.

It is **not promoted as the incumbent yet**. ID switches rise from 427 to 554 and fragments from 523 to 608. The hard `0020` sequence shows the tradeoff most clearly:

```text
manual no-ReID: HOTA 30.66  AssA 25.38  IDF1 34.05  IDSW 322  Frag 411
learned 64D:    HOTA 32.79  AssA 27.56  IDF1 37.22  IDSW 452  Frag 496
```

On `0016`, learned appearance is almost neutral and slightly reduces switches. The hard sequence is therefore the useful discriminator.

A raw-feature control makes that conclusion sharper. On the hard `0020` sequence:

```text
                              HOTA   AssA   IDF1  IDSW  Frag
manual path, no ReID          30.66  25.38  34.05   322   411
raw normalized ROI-pooled     32.43  27.04  35.56   434   494
seeded random 64D projection  32.56  27.30  37.10   466   522
trained 64D projection        32.79  27.56  37.22   452   496
```

Most of the capability jump comes from exposing ROI-pooled YOLO appearance features to BoT-SORT. A seeded **untrained** 64D projection improves them a little further. Training the 64D head adds a smaller but repeatable gain over that random projection, about **+0.23 HOTA and +0.27 AssA**, while reducing switches from 466 to 452 and fragments from 522 to 496. Keep the trained head because it is the best of these simple variants, but do not attribute the entire appearance gain to the retrieval objective.

Two focused follow-ups did not earn their place. Raising retrieval temperature from `0.1` to `1.0` reduced held-out top-1 retrieval from **57.67% to 49.68%** and reduced `0020` tracking HOTA from 32.79 to 32.63. Restricting appearance to detections overlapping another box at IoU >=0.3 reduced `0020` switches from 452 to 428 and raised AssA from 27.56 to 27.79, but lowered HOTA to 32.66 and hurt DetA/recall/precision. Keep the simpler temperature-`0.1`, dense-frame gate as the active probe.

### Lifecycle correction

Targeted switch review on `0020` showed that much of the appearance regression was track-birth churn in crowded frames, not just pairwise identity swaps. The simplest useful fix was therefore lifecycle, not a larger embedding model: keep ByteTrack-style low-score recovery for existing tracks, but require a stronger score to start a new track.

Sweeping only `new_track_thresh` on `0020` with the frozen learned feature found `0.45` as the best HOTA point:

```text
new_track_thresh   HOTA   AssA   IDF1  IDSW  Frag  Recall
0.25               32.79  27.56  37.22   452   496   53.28
0.35               32.77  28.24  38.29   344   439   51.37
0.40               32.73  28.69  37.59   286   403   50.14
0.45               33.69  31.21  39.40   236   376   48.72
0.50               33.17  31.19  39.36   219   353   47.02
```

Across both development sequences, the global `0.45` rule gives the strongest current dev candidate:

```text
                                      HOTA   AssA   DetA   IDF1  IDSW  Frag  Recall  Precision
official model.track baseline         32.20  19.39  53.94  30.32   397   499   66.67     96.03
manual no-ReID, new-track 0.45        33.24  21.18  52.40  33.25   289   458   64.73     97.27
learned 64D, new-track 0.45           33.58  21.19  53.44  33.60   323   486   66.14     96.44
```

The lifecycle rule is doing most of the work. At the same `0.45` threshold, learned appearance adds **+0.34 HOTA, +0.35 IDF1, and +1.41 recall points**, but also adds 34 switches and 28 fragments. Keep the learned feature because it still gives the highest HOTA/IDF1 while recovering recall, but treat lifecycle as the dominant current bottleneck.

Three follow-ups were rejected rather than accumulated: confidence-gating appearance at 0.35/0.40/0.45 lowered `0020` HOTA versus the ungated 0.45 candidate; applying the stricter birth threshold only on dense/ReID-active frames produced 33.69 HOTA but worse IDF1 and more switches than the global rule; and a short explicit hard-negative-margin training probe behaved almost identically to the existing low-temperature retrieval loss.

TrackTrack (CVPR 2025) was also checked because its track-aware initialization explicitly targets spurious births. Stock Ultralytics defaults were too conservative for this detector on the two-sequence dev set (HOTA **26.23**, recall **48.66%**). A fair `0020` control with only its confidence thresholds aligned to our validated regime (`high=0.25`, `low=0.10`, `new=0.45`) reached HOTA **31.37**, AssA **26.95**, IDF1 **37.04**, IDSW **201**, recall **48.00%**. That is still clearly below the learned-64D + global-0.45 candidate on the same sequence, so do not spend more time tuning TrackTrack here.

The first detection-centric lifecycle probe exposed a more important supervision bug. Training from arbitrary frame pairs is invalid for NEWBORN semantics because the real tracker processes every intervening frame and retains bounded history. On held-out DanceTrack `0012`, **97.2%** of identities that a sampled pair would label NEWBORN had actually been seen within the previous 30 frames. Pair-based lifecycle training is therefore rejected.

Using sequential oracle memory with a 30-frame / roughly 1.5-second horizon fixes the label semantics. On `0012`, detector-backed current detections split into **10,738 KNOWN / 34 NEWBORN / 3,755 DROP**, and frozen cosine retrieval picks the correct remembered identity for **88.33%** of KNOWN detections. DanceTrack train `0001` has almost no true lifecycle births (**4,735 / 8 / 3,047**), while the five-video PersonPath starter slice contributes **11,785 / 729 / 8,454** and about **87.17%** known-ID cosine retrieval, so PersonPath is the useful lifecycle supervision source.

A deliberately tiny two-stage classifier on six sequential-memory cues (confidence, max appearance similarity, max IoU, nearest normalized center distance, current-frame crowding, and age of the best appearance match) reaches **85.48%** semantic accuracy on held-out `0012`: KNOWN **89.38%**, NEWBORN **35.29%**, DROP **74.75%**. Conditional NEWBORN-vs-DROP accuracy is **92.77%** when the example is already known to be non-KNOWN. The remaining hard decision is therefore KNOWN-vs-NEWBORN, not false-positive rejection.

Preserving candidate alignment fixes much of that ambiguity. A tiny pair scorer trained on four sequential PersonPath videos from appearance cosine, relative box geometry, IoU, memory age, detector confidence, and current-frame crowding improves held-out `0012` remembered-ID selection from **88.33%** for cosine alone to **94.37%**. Calibrating its max match score on the fifth PersonPath video raises true NEWBORN recognition to **58.82%** and KNOWN recognition to **91.66%**, but DROP recognition falls to **44.18%**. This is useful evidence: candidate-aligned memory is worth keeping, while a single match threshold is not sufficient lifecycle logic.

Later online probes show that this offline gain is not enough by itself. A distilled bounded-memory student using the candidate-aligned cues runs at about **32.7 FPS** on the laptop but reaches only **29.36 HOTA**, **29.60 IDF1**, **584** switches, and **48.59%** recall on `0020`; an ambiguity-margin variant is worse. A classifier trained directly on the unmatched low-confidence birth candidates emitted by the online tracker also fails to generalize to `0012`: its best balanced accuracy is only **52.04%**, with **6.90%** true-birth recall. Do not promote either path.

Two simpler lifecycle recovery rules were also rejected. Requiring extra observations before admitting low-confidence births raises `0020` recall to **51.51%** but falls to HOTA **32.97** with **313** switches. Transplanting only TrackTrack's track-aware birth suppression into the stronger BoT-SORT path reaches HOTA **33.22** but returns to **396** switches. Both recover detections by giving back too much identity stability.

Detector-head adaptation was checked after the memory student failed online. A three-epoch Detect-head-only YOLO26n fine-tune on **5,194** training frames improves held-out `0012` raw detection metrics, reaching about mAP50 **0.865**, mAP50-95 **0.533**, and recall **75.14%**. The errors are less tracker-friendly, however. With no-ReID BoT-SORT and the same `0.45` lifecycle rule on `0020`, the adapted detector gives HOTA **28.44**, IDF1 **30.93**, **624** switches, recall **56.26%**, precision **80.40%** at detector confidence 0.10. Raising detector confidence to 0.30 still gives only HOTA **25.55**, IDF1 **29.66**, **539** switches, recall **47.05%**, precision **87.14%**. The original detector at the same lifecycle setting remains HOTA **33.17**, IDF1 **38.89**, **201** switches, recall **46.43%**, precision **95.97%**. Reject this detector adaptation and do not spend more work calibrating the same objective.

### Closed-loop association finding

Training the candidate scorer on real BoT-SORT predicted states rather than oracle memory improves held-out `0012` candidate selection from cosine **68.69%** to **77.22%**. That first live-state metric was still too permissive: about **20%** of queries contained multiple fragmented BoT-SORT tracks for the same GT person, so choosing any same-GT track could count as correct even when it changed the public ID.

Changing the target to **preserve the previous public owner track** makes the continuity problem explicit. On held-out `0012`:

```text
cosine owner selection        81.07%
live-state pair scorer        95.02%
owner-aware 11-cue scorer     95.58%
```

The owner-aware scorer adds only tracklet length and active/lost state to the existing pair cues. With score margin >=2 it covers **91.8%** of owner queries at **98.35%** accuracy. That strong one-step result still does not transfer directly through closed-loop tracking:

```text
0020 variant                              HOTA   AssA   IDF1  IDSW  Frag
incumbent learned64D + BoT-SORT 0.45     33.69  31.21  39.40   236   376
owner appearance pruning                  33.45  30.97  39.17   236   368
owner full-cost bonus 0.05                32.81  29.52  38.39   246   375
```

An exact first-association replay shows why pairwise accuracy is insufficient. The incumbent Hungarian assignment preserves the previous public owner on **90.01%** of held-out `0012` targets. A tiny rank bonus of **0.005**, selected on the five training sequences, improves that frozen-trajectory metric to **90.93%**. Yet the same first-association-only bonus reaches only **32.15 HOTA / 239 IDSW** online on `0020`.

The switch diagnostic confirms that association still matters: among **349** public-owner changes on `0020`, the previous owner remains in the first-association candidate pool in **300** cases, and the owner scorer prefers it in **135** cases versus **60** for cosine. But changing those decisions perturbs later memory and assignment states, so one-step improvements can still reduce end-to-end HOTA.

The conclusion is stronger than “use a larger pair scorer”: **teacher-forced or frozen-trajectory association accuracy is not a reliable promotion metric**. Small assignment changes alter the future memory distribution. Do not spend more time tuning static score blends inside BoT-SORT.

### Next research question

Freeze the global `new_track_thresh=0.45` learned-feature configuration as the current dev candidate. Do not spend more iterations tuning that scalar. The evidence now says the next capability to learn should be **lifecycle**, especially deciding when a weak unmatched detection deserves to become persistent memory without sacrificing ByteTrack's useful low-score continuation.

Single-frame and pairwise memory cues have now been pushed far enough. The next probe should be trained on **self-generated rollout states**, not oracle memory or a frozen teacher trajectory. Use the cached detector outputs and 64D features so this is cheap enough to iterate. Scheduled sampling or dataset aggregation is the right complexity class: keep the memory state and scorer small, but expose them during training to the mistakes and duplicate states they create themselves.

Only add a recurrent track state if rollout-trained last-state memory is still insufficient. The immediate objective is to close the train/inference state-distribution gap, not to add sequence-model capacity. Promotion requires a full online `0020` rollout that beats the current `0.45` candidate; one-step assignment accuracy alone no longer counts as evidence.

The learned appearance head remains an auxiliary input/control, not proof that a larger ReID model is needed. Any new representation work must still beat raw pooled features and the seeded random projection.

Keep `0096` and `0004/0005/0007/0010` untouched while the lifecycle module is being selected on development.

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

# Cheap appearance controls on the hard development sequence
.\.venv\Scripts\python.exe scripts/track.py dancetrack0020 --feature-mode raw --output runs/raw-pooled-feature-0020
.\.venv\Scripts\python.exe scripts/track.py dancetrack0020 --feature-mode random --output runs/random-identity-track
.\.venv\Scripts\python.exe scripts/evaluate.py runs/raw-pooled-feature-0020 dancetrack0020
.\.venv\Scripts\python.exe scripts/evaluate.py runs/random-identity-track dancetrack0020

# Current dev candidate
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --head runs/identity-head-v2/head.pt --new-track-threshold 0.45 --output runs/identity-v2-new045
.\.venv\Scripts\python.exe scripts/evaluate.py runs/identity-v2-new045 dancetrack0016 dancetrack0020

# Lifecycle-only control
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --disable-reid --new-track-threshold 0.45 --output runs/manual-no-reid-new045
.\.venv\Scripts\python.exe scripts/evaluate.py runs/manual-no-reid-new045 dancetrack0016 dancetrack0020
```

## Research loop

The current bet is the incumbent, not something we are obligated to keep modifying. Finish and measure the simplest version before adding capability.

The agent has broad freedom to investigate. It may read literature and implementations, inspect and relabel data, build focused probes, change the representation or training objective, simplify or rewrite code, or conclude that the incumbent should remain untouched. The constraint is evidence, not conservatism: explore widely, but only promote changes that earn their place.

1. **Measure the incumbent.** Reproduce the frozen baseline, train/evaluate the current bet, and keep the same development videos and settings so changes are attributable.
2. **Find a real weakness.** Use HOTA/AssA/DetA/IDF1, ID switches, runtime, and targeted failure review. Decide whether the bottleneck is detection, representation, association, memory, motion, data, or evaluation before proposing a fix.
3. **Run the cheapest discriminating experiment.** Prefer an ablation or narrow change that can prove/disprove the idea. Research papers/implementations when the mechanism is unclear. Do not add complexity because it sounds plausible.
4. **Promote only meaningful wins.** A candidate must improve the relevant tracking capability without hiding a material detection/runtime regression. Tiny metric noise is not enough. If it is not clearly better, keep the incumbent and remove the candidate machinery.
5. **Check that the new capability is real.** Ablate the added memory/context. If removing it barely changes tracking, the model did not learn the intended behavior. Once a candidate is frozen on development, compare against the stronger BoT-SORT+ReID reference and then evaluate the untouched sequences exactly once.
6. **Housekeep and checkpoint.** Remove dead scratch paths, keep generated artifacts untracked, commit coherent evidence-backed changes, and push useful checkpoints/results so the remote branch stays a recoverable handoff.
7. **Repeat when evidence gives a reason.** New data, a reproducible failure family, or a credible research insight can justify another change.

The goal is not continuous code churn. It is to preserve the strongest known tracker while taking high-information shots at real capability jumps. Success means the one-network model approaches or beats strong tracking references while keeping useful detection quality and a credible path to >=15 FPS on Nano.

## Working rules

- Keep this repo small. Git history is the archive.
- Delete dead experiments instead of maintaining legacy compatibility.
- Prefer one obvious path over frameworks and abstraction.
- Keep tests only where a silent bug would invalidate training or evaluation.
- Raw datasets live in `data/`, generated runs in `runs/`, and weights in `weights/`; all stay untracked.
- Never claim a result without exact model settings, tracker settings, evaluated sequences and metrics.
