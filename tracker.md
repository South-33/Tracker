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

Fine-tune the whole network on video sequences so its features can become temporal. Keep the first model deliberately small:

```text
YOLO26n
  -> person box + confidence
  -> 64D tracking embedding
  -> simple motion output (start with dx/dy)
```

Runtime memory per track starts as:

```text
3-5 diverse high-quality embeddings
last box
recent motion
last-seen time
```

Sequence identity labels train front/side/back/partial views of the same person to stay compatible while different people separate. Do not save raw old frames unless experiments show embeddings are insufficient.

Only add recurrent/transformer memory, longer context, learned association matrices, or more state after the simple version proves it needs them.

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

## Current experiment

The first learned-memory probe is now measured. It deliberately **froze YOLO26n** and trained only a small 64D embedding, association head and absence head so we could isolate whether shared detector features plus simple causal memory were enough before fine-tuning the detector itself.

Source commit: `1c00228`. Checkpoint SHA-256: `ced100f58ff1ef5a96d18d23c11c76e7e3eed1db729df1ac27da1c3bdd76eaea`.

Training used 500 AdamW steps at `1e-3`, alternating PersonPath22 starter triples with complete DanceTrack train sequences `0001/0002/0006/0008/0015`. DanceTrack `0012` was held out for validation. Choice accuracy improved from **11.96% to 59.94%** and validation loss fell from **2.511 to 1.303**, so the head learned the supervised choice task. Runtime used person-only detection at `conf=0.1`, `iou=0.7`, 640 px, up to 5 prototypes per track, a 5-second maximum age and a `0.25` birth threshold.

End-to-end tracking did not transfer well enough:

```text
dev: dancetrack0016 + dancetrack0020

                        baseline    learned probe
HOTA                       32.20            27.84
AssA                       19.39            14.35
DetA                       53.94            54.85
IDF1                       30.32            26.09
ID switches                  397             1326
Fragments                    499             1197
Recall                    66.67%           75.67%
Precision                 96.03%           82.78%
```

The failure is concentrated on hard crossings in `0020`: the probe records 1,129 ID switches and 1,051 fragments there, versus 287 and 381 for the baseline. It gets more detections through, but loses track identity and admits too many false positives. Runtime is not the immediate blocker on the RTX 4060: `0016` runs at 36.4 compute FPS / 26.1 pipeline FPS and `0020` at 29.9 / 22.6.

**Decision:** keep the frozen BoT-SORT baseline as the incumbent. Do not promote this learned matcher or simply train it longer. The next useful work is to explain the gap between per-triple validation accuracy and stable online tracking. First measure true-match, best-impostor and absence score calibration by gap/crowding, then replay with oracle track lifecycle to separate representation/association errors from birth, death and global assignment errors. Only change the model or objective after that split is clear.

Local exploratory runs suggest the learned 64D feature may help BoT-SORT in ambiguous crowded frames, but those probes were one-off scratch experiments. Treat that as a hypothesis to reproduce, not a result.

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
