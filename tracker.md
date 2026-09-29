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

## Baseline to beat

Before training the new model, run **YOLO26n + BoT-SORT** on the same frozen development videos and record exact settings here.

Development videos:

```text
dancetrack0016
dancetrack0020
```

Record HOTA, AssA, DetA, IDF1, ID switches, detection precision/recall and FPS.

**Baseline status: not run yet.**

## First experiment

1. Reproduce YOLO26n + BoT-SORT.
2. Add the smallest temporal/tracking heads to YOLO26n.
3. Train on short labeled clips while mixing detection-only image batches.
4. Compare on the exact same development videos.
5. Ablate memory/context. If removing it barely changes tracking, the network did not learn useful temporal reasoning.

Success means the one-network model approaches or beats the baseline while keeping useful detection quality and a credible path to >=15 FPS on Nano.

## Working rules

- Keep this repo small. Git history is the archive.
- Delete dead experiments instead of maintaining legacy compatibility.
- Prefer one obvious path over frameworks and abstraction.
- Keep tests only where a silent bug would invalidate training or evaluation.
- Raw datasets live in `data/`, generated runs in `runs/`, and weights in `weights/`; all stay untracked.
- Never claim a result without exact model settings, tracker settings, evaluated sequences and metrics.
