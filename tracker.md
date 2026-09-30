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

Across both development sequences, the global `0.45` rule was the strongest dev candidate before guarded assignment learning:

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

An exact first-association replay shows why pairwise accuracy is insufficient. The incumbent Hungarian assignment preserves the previous public owner on **89.34%** of held-out `0012` targets. A per-detection normalized assignment residual selected on the five training sequences improves that frozen-trajectory metric to **92.01%**, but a large residual (`alpha=0.2`) destabilizes rollout and falls to **30.37 HOTA / 321 IDSW** on `0020`. One round of on-policy data collection and retraining does not rescue it: the owner scorer moves only from **95.78%** to **95.82%** on its induced `0012` states and the corresponding rollout falls further to **29.52 HOTA / 354 IDSW**.

The useful part is much smaller. Keeping the same owner scorer but reducing the assignment residual to **0.005**, with the value chosen on the five training sequences, reproducibly gives on `0020`:

```text
variant                         HOTA   AssA   IDF1  IDSW  Frag  Recall  Precision
incumbent 0.45                 33.69  31.21  39.40   236   376   48.72      94.29
owner residual alpha=0.005     33.75  31.26  39.69   228   363   48.63      94.39
```

This is a real dense-sequence identity-continuity gain, but it does not earn promotion across both exposed development sequences. On `0016`, HOTA changes **33.12 -> 33.00** and IDSW **87 -> 88**. Combined `0016+0020`, the residual gives **33.564 HOTA / 21.158 AssA / 33.757 IDF1 / 316 IDSW / 473 Frag**, versus the current candidate's roughly **33.578 HOTA / 21.195 AssA / 33.601 IDF1 / 323 IDSW / 486 Frag**. The residual reduces switches and fragments but does not improve the primary combined HOTA/AssA result, so keep it as evidence rather than the default.

Trying to preserve BoT-SORT's matchable-edge set does not help: the candidate-graph-preserving `alpha=0.005` version falls to **33.53 HOTA / 272 IDSW** on `0020`. A density gate selected only on the five training sequences is also rejected; training prefers applying the tiny residual everywhere it is available.

The switch diagnostic confirms that association still matters: among **349** public-owner changes on `0020`, the previous owner remains in the first-association candidate pool in **300** cases, and the owner scorer prefers it in **135** cases versus **60** for cosine. But changing those decisions perturbs later memory and assignment states, so one-step improvements can still reduce end-to-end HOTA.

The conclusion is stronger than "use a larger pair scorer": **teacher-forced or frozen-trajectory association accuracy is not a reliable promotion metric**. Small assignment changes alter the future memory distribution, and the semantic KNOWN/NEWBORN/DROP decision remains entangled with assignment. Do not spend more time tuning unconstrained static score blends inside BoT-SORT.

The useful exception is a **guarded assignment tie-break**, which keeps BoT-SORT as the primary policy. The learned owner scorer may propose a different first-association Hungarian solution, but that proposal is accepted only when all of the following hold:

- every learned edge was already below BoT-SORT's normal match threshold,
- the learned assignment has exactly the same number of matches as the incumbent assignment,
- the learned assignment raises the *original BoT-SORT cost* by at most **0.00025 per match on average**.

This tiny ambiguity budget was checked on held-out `0012`: owner preservation improves from **90.01%** to **90.61%** while changing only **11.3%** of first-association frames. Unlike the larger residual policies, this conservative rule survives online rollout.

A fresh reproducible owner-continuity head trained by `scripts/train_owner.py` on the five DanceTrack training sequences reaches **95.98%** owner-selection accuracy on held-out `0012` using **18,904** training queries and **7,951** validation queries. With that freshly generated checkpoint, the guarded tracker gives:

```text
development result                         HOTA   AssA   DetA   IDF1  IDSW  Frag  Recall  Precision
learned64D + birth 0.45                  33.58  21.19  53.44  33.60   323   486   66.14      96.44
+ guarded owner-continuity tie-break      33.83  21.47  53.54  34.17   326   483   66.15      96.53
```

On the hard `0020` sequence alone, the fresh guarded candidate reaches **34.10 HOTA / 31.80 AssA / 40.43 IDF1 / 239 IDSW / 373 Frag**, versus **33.69 / 31.21 / 39.40 / 236 / 376** for the previous candidate. The gain is therefore not switch-count optimization: it trades three extra switches for better overall association quality and IDF1 while holding recall effectively flat.

The tracked implementation reproduces the scratch result through `scripts/track.py`. On the laptop manual path it runs at about **22.1 FPS** on `0020` and **18.4 FPS** on `0016`. This keeps a credible runtime path, but the >=15 FPS Orin Nano target is still unverified.

### Next research question

Promote the **learned64D + global birth 0.45 + guarded owner-continuity tie-break** as the current development candidate. The guard matters: unconstrained learned assignment remains rejected.

Subsequent architectural probes rule out several tempting ways to extend that candidate:

- **Identity-only oracle headroom is real.** Replaying the guarded tracker on cached `0020` gives about **34.14 HOTA**. Keeping exactly those emitted boxes and replacing only public IDs with a GT-matched 30-frame oracle raises HOTA to **38.83**, AssA to **39.90**, and IDF1 to **49.12**. A global identity oracle reaches **44.48 HOTA / 52.14 AssA / 64.44 IDF1**. The remaining gap is therefore not explained by detection alone.
- **More spatial appearance is not the answer.** Full raw `2x2` ROI layout improves the hard `0020` representation control, but reaches only about **33.70 HOTA combined** on `0016+0020`; the structured per-cell 512D projection reaches about **33.72**, and guarded variants remain below the **33.83** incumbent. Spatial layout contains signal but does not solve continuity.
- **A recurrent hidden track state is actively harmful closed-loop.** A 32D GRU slot model trained on oracle identity histories reaches **94.31%** held-out known-ID choice accuracy, but a causal `0020` rollout collapses to **26.01 HOTA / 24.93 IDF1 / 904 IDSW**. Teacher-forced state accuracy is again not a promotion metric.
- **Joint existing-ID / NEW / DROP classification is not enough.** Tiny joint-softmax variants retain good known-ID selection but held-out true-birth recall stays around **32%**. Candidate-set context does not fix lifecycle semantics.
- **Detached public-ID repair does not earn its place.** A pair scorer trained on incumbent output streams reaches about **96.35%** held-out identity selection under clean/teacher-forced memory and can identify a small high-confidence correction tail, but causal and frame-local output overlays leave `0012` essentially unchanged at about **25.70 HOTA**. Restart-only relinking is too rare to matter on `0020`.
- **Appearance-update rate is already near its useful point.** In the guarded cached `0020` replay, the current EMA `alpha=0.90` gives **34.14 HOTA**. Slower updates regress (`0.95 -> 33.21`, `0.98 -> 32.84`, frozen `1.0 -> 29.29`), while faster updates also regress (`0.8 -> 33.30`, `0.7 -> 33.16`, `0.5 -> 32.25`). State corruption is not an EMA-rate tuning problem.
- **History summaries are too lossy for corruption training.** A four-observation history MLP previously improved held-out invalid-edge owner selection from **33.3% to 37.7%**, but its guarded online rollout regressed. Inspired by MOTIP's trajectory random-occlusion/random-switch training, adding synthetic corruption to the same 22-cue history MLP does not transfer: light corruption reaches **35.5%** invalid-edge accuracy and stronger corruption **32.6%**, both below the unaugmented model.
- **The bounded-memory horizon was too short.** On the fixed guarded-`0020` output boxes, the GT public-ID oracle rises monotonically from **24.93 HOTA at 1 frame**, **30.47 at 4**, **34.97 at 8**, **37.17 at 15**, **38.83 at 30**, **41.41 at 60**, **42.45 at 90**, and **43.82 at 120**. The global oracle is **44.48**. A 120-frame bound captures most of the identity-only ceiling while remaining finite.
- **The current appearance representation cannot exploit that longer horizon.** On held-out `0012`, last-observation retrieval with the learned 64D feature is about **93.8% at gap 1**, **29.6% at 2-4**, **22.5% at 5-10**, **10.8% at 11-30**, **3.1% at 31-60**, and effectively zero beyond 60. Raw 448D pooled YOLO features and raw 2x2 spatial features are no better at long gaps, so the information loss is not just the 64D projection.
- **A gap-aware residual metric transform does not recover the missing identity signal.** Training a tiny residual transform on temporally separated positives and same-frame negatives nudges held-out `2-4` retrieval from **29.6% to 32.2%**, leaves `5-10` unchanged, and slightly worsens `11-30`. Reject reweighting the same 64D representation.
- **Generic crop appearance and a trained projection also fail.** ImageNet MobileNetV3-Small crop features slightly improve one mid-gap bucket but are worse on already-drifted identities. A 576->128 projection trained with uniformly sampled temporal gap buckets collapses gap-1 retrieval from about **90.6% to 79.3%** while ending at only **26.5% / 19.8% / 9.5%** for `2-4 / 5-10 / 11-30`. Reject this branch.
- **Specialist OSNet x0.25 is useful but still not enough to justify a second always-on backbone.** The cached MSMT17-pretrained model improves 120-frame restart-event top-1 retrieval on `0012` from about **30.8%** for YOLO appearance to **50.0%** when the correct lost identity exists in the archive. However a learned 11-cue NEW-vs-return gate reaches only about **40.4%** top-1 on known returns, and high-precision gates cover just one or two events. A causal local-ID archive using full-resolution OSNet features gives only a small output-layer gain at the frozen `0.05` stay bias: `0012` moves **25.7006 -> 25.8764 HOTA** and `0020` **34.1009 -> 34.1293**, while switches worsen. A zero-initialized residual metric preserves OSNet's short-gap retrieval and lifts held-out `11-30` retrieval from **18.9% to 20.3%**, but the corresponding causal overlay still reaches only **34.1236 HOTA / 246 IDSW** on `0020`. The OSNet extractor alone runs at about **22.9 FPS** on the laptop, so using it every frame would also consume too much of the deployment budget for the observed benefit.
- **A corruption-trained MOTIP-lite decoder does not solve the state problem.** A one-layer, 48-ID-token, 30-frame decoder trained with history dropout and deliberate ID-token swaps reaches only **89.2%** held-out existing-ID accuracy, **35.2%** accuracy on already-wrong local identities, and collapses to **2.7%** at 11-30-frame gaps. This is not strong enough to justify closed-loop rollout.

The identity-memory search is now saturated enough to stop adding branches. The guarded owner tracker remains the strongest reproducible system, and the remaining project gap is architectural: it still uses BoT-SORT as a research harness even though the target is one small causal tracker. The next work should therefore **consolidate the proven pieces into one explicit bounded-state tracking model** rather than search for another public-ID overlay or ReID feature:

1. keep YOLO26n detection and the trained 64D ROI appearance head;
2. keep the validated `0.45` birth rule and the guarded owner-continuity scorer;
3. make the bounded track state and update step explicit in project code, with one causal `step(frame, memory) -> boxes/confidence/IDs/new_memory` interface;
4. reproduce the guarded incumbent on `0016+0020` before changing any association behavior;
5. only after equivalence, profile/export that single path for the >=15 FPS Orin Nano target.

The first two consolidation steps are complete. `src/tracker/runtime.py` exposes perception + online association through one causal `step(frame)` interface, and `src/tracker/causal.py` now owns the actual tracking policy: confidence splitting, bounded lost-track memory, prediction orchestration, GMC application, two-stage association, birth/removal lifecycle, appearance costs, and guarded owner tie-breaking. Low-level Kalman/track/assignment primitives are still reused from Ultralytics for numerical compatibility, but BoT-SORT no longer owns the guarded policy. Cached replay is **exact** on both development sequences: `0020` has **10,391 rows with zero ID mismatches, zero box mismatches, max box error 0.0**, and `0016` has **12,062 rows with the same exact agreement**. A fresh live-detector `0020` run through the normal `scripts/track.py` path also exactly preserves the incumbent metrics (**34.1009 HOTA / 31.8031 AssA / 40.4251 IDF1 / 239 IDSW / 373 Frag**) at about **20.5 pipeline FPS** on the development laptop. Preserve this equivalence while simplifying the remaining primitives.

Do not spend more development time on detached public-ID remapping, generic crop backbones, gap-only metric heads, or always-on secondary ReID unless a new dataset or stronger supervision changes the evidence.

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

# Previous dev candidate
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --head runs/identity-head-v2/head.pt --new-track-threshold 0.45 --output runs/identity-v2-new045
.\.venv\Scripts\python.exe scripts/evaluate.py runs/identity-v2-new045 dancetrack0016 dancetrack0020

# Current guarded dev candidate
.\.venv\Scripts\python.exe scripts/train_owner.py --output runs/owner-head-v1/head.pt
.\.venv\Scripts\python.exe scripts/track.py dancetrack0016 dancetrack0020 --head runs/identity-head-v2/head.pt --owner-head runs/owner-head-v1/head.pt --new-track-threshold 0.45 --output runs/identity-v2-guarded
.\.venv\Scripts\python.exe scripts/evaluate.py runs/identity-v2-guarded dancetrack0016 dancetrack0020

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
