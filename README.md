# Temporal YOLO Tracker

## Goal

Post-train YOLO26n as one causal video tracker.

    current frame + bounded neural memory
        -> person boxes + confidence + anonymous track slots + updated memory

The model is trained on ordered video clips with person track IDs so it can
learn motion, crossings, occlusion, pose changes, disappearance and recovery.

## Hard rule

The active tracker must not use an external identity tracker at inference.

No BoT-SORT, ByteTrack, Kalman filter, Hungarian online association, external
ReID model, owner MLP, or hand-written ID lifecycle may decide which person
keeps an ID. Ordinary tensor post-processing such as confidence filtering is
fine.

If identity continuity comes from code outside the neural model, it is a
baseline or diagnostic, not the project result.

## Why this experiment

Official YOLO26 pretraining is image detection: Objects365 pretraining followed
by COCO fine-tuning. Standard detection training shuffles image samples and does
not keep temporal state between neighboring video frames.

YOLO26n already knows what people look like. Our bet is that post-training it on
ordered human video clips can teach the missing concept:

    frame t and frame t+1 are physically connected
    the same person moves, turns, overlaps, disappears and returns

## Architecture v0

    YOLO26n pretrained visual backbone/neck
                |
                v
          current visual features
                |
                +---- bounded recurrent track slots from previous frame
                |
                v
          temporal track-slot head
                |
                +--> slot alive/confidence
                +--> slot person box
                +--> updated slot memory

Each persistent slot is one anonymous identity. The slot index is the track ID.
A slot can stay in memory while temporarily invisible and reactivate when the
person returns.

Start small:

- 640 px input.
- 64 track slots.
- 128D memory per slot.
- 8-frame training clips initially.
- Memory is carried frame-to-frame and detached only at clip boundaries or the
  chosen truncated-BPTT boundary.
- Person class only.

Do not add a separate appearance model or classical tracker unless it is only a
comparison baseline.

## Training data

We care about humans in video, from easy to difficult:

| Dataset | Role | What it gives us |
|---|---|---|
| PersonPath22 | train | varied real-world people, scale, pose, camera/viewpoint, occlusion |
| DanceTrack | train + held-out benchmark | crossings, similar-looking people, deformation, crowd association |
| SportsMOT | train + held-out benchmark | running, rapid direction changes, overlap, fast camera motion |
| MOT17 | benchmark only | classic pedestrian generalization across different cameras and densities |
| MOT20 | optional benchmark only | extreme crowd stress |

Do not train on benchmark-only data.

The first experiment intentionally avoids generic static-image fine-tuning.
YOLO26n already has strong image detection pretraining; we want to isolate the
value of sequence training.

## Training sample

A sample is a contiguous ordered clip, never randomly scattered frames:

    [t, t+1, t+2, ... t+7]

Each frame contains person boxes and the dataset track ID for each box.

Augmentations must be temporally consistent. If a clip is horizontally flipped,
every frame receives the same flip.

## Losses

Keep the first version simple:

1. person presence/confidence loss
2. box regression loss
3. track-slot continuity loss across the clip
4. absence/reappearance supervision for temporarily missing identities

The key metric is not a frame-wise identity classifier. The same slot must keep
the same person while the clip runs causally.

## Evaluation

Always compare against:

1. stock YOLO26n detection
2. stock YOLO26n + BoT-SORT tracking
3. temporal YOLO with memory reset every frame
4. temporal YOLO with normal running memory

The memory-reset ablation is mandatory. If resetting memory barely changes
tracking, the temporal model did not learn the intended behavior.

Report HOTA, DetA, AssA, IDF1, ID switches, recall, precision and FPS.

## Current status

Stage-0 smoke test passed on one real 8-frame DanceTrack clip:

- 64 recurrent slots, 128D memory each
- YOLO26n visual weights frozen
- sequence loss: 2.39 -> 0.20 in 100 steps
- resetting memory every frame makes the fitted loss about 14% worse

That is only an overfit proof, not a tracking result. The next gate is
generalization: train across many training clips and require running memory to
beat reset memory on unseen calibration/development clips.

Active scripts are intentionally few:

- scripts/train.py: temporal clip training
- scripts/baseline.py: external YOLO26n + BoT-SORT comparison only
- scripts/evaluate.py: MOT metrics

## Legacy tracker

The previous YOLO26n + learned appearance + project-owned association system is
archived in Git tag legacy-botsort-v1.

It is a useful baseline, not the active architecture. Do not copy its
association/lifecycle machinery back into the temporal model.

## Git workflow

Work only on main. Use small checkpoint commits and push them. Do not create
normal research worktrees or feature branches.

tracker.md is the short active research notebook. Git history and the legacy
tag are the archive.
