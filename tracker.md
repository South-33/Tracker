# Temporal YOLO Tracker - Active Notebook

## Goal

Post-train pretrained YOLO26n on ordered human video clips so one causal network
learns both person boxes and anonymous tracking continuity.

    current frame + bounded neural memory
        -> person boxes + confidence + stable track slots + updated memory

No external tracker may decide identity at inference.

## Core hypothesis

YOLO26n already has strong object/person perception, but its official detection
training is image-based rather than sequence-based. It was not trained with a
running temporal context that says neighboring frames obey motion and identity
continuity.

The bet:

Give YOLO26n short ordered clips with per-person IDs and a bounded recurrent
memory. Post-train it hard enough, and it may learn motion, pose change,
crossings, temporary disappearance and recovery directly in the network.

This is the experiment. Do not drift back to engineering a better classical
tracker around YOLO.

## Architecture v0

Use YOLO26n's pretrained backbone/neck as visual perception.

Add a small shared temporal track-slot head:

- 64 persistent slots
- 128D state per slot
- each slot predicts alive/confidence + normalized person box
- recurrent slot state is updated from current YOLO features
- slot index is the anonymous track ID
- invisible slots remain in bounded memory for recovery

Inference has no external ID association. A forward step is conceptually:

    tracks, memory = model.step(frame, memory)

The first version should use simple recurrent updates and attention to current
YOLO features, not a large video transformer.

## Sequence training

Initial clip length: 8 contiguous frames.

Later increase the effective context only if the 8-frame experiment shows real
temporal signal.

Training clips must preserve order. Randomize which clip is sampled, not the
frame order inside the clip.

Use truncated backpropagation through the whole short clip initially.

Important training cases:

- normal walking
- people approaching/leaving camera
- scale changes
- turning front/side/back
- crossings
- partial/full occlusion
- temporary disappearance/reappearance
- running and rapid direction changes
- moving cameras
- sparse and crowded scenes

## Data plan

Train:

- PersonPath22 for broad real-world human-video diversity
- DanceTrack for difficult association and crossings
- SportsMOT for fast running and camera motion

Benchmark only:

- held-out DanceTrack
- held-out SportsMOT
- MOT17
- MOT20 later if needed

Do not use benchmark-only videos for training or architecture selection.

## First training stages

Stage 0 - smoke test
- load YOLO26n pretrained weights
- freeze most YOLO weights
- train only the temporal slot head on short clips
- prove loss falls and slot continuity beats chance

Stage 1 - temporal post-train
- unfreeze YOLO neck + temporal head
- keep early visual backbone mostly frozen
- train mixed PersonPath22/DanceTrack/SportsMOT clips
- preserve person detection while learning continuity

Stage 2 - only if justified
- selectively unfreeze more YOLO layers
- longer context or harder occlusion curriculum

Do not jump directly to a huge end-to-end retrain.

## Required ablations

Every serious checkpoint must be tested in two modes:

1. normal running memory
2. memory reset to empty every frame

If normal memory does not materially beat reset memory on AssA/IDF1/HOTA, the
model has not learned the thing we care about.

Also compare with stock YOLO26n + BoT-SORT as an external baseline only.

## Success criteria

The temporal model should:

- preserve strong person detection
- materially improve identity continuity using its own memory
- recover after short occlusion/disappearance
- work causally on arbitrary-length video with bounded state
- remain small enough to optimize toward >=15 FPS on Orin Nano Super later

## Archived previous direction

The previous tracker chained YOLO26n to learned appearance, motion/GMC,
association rules and a small owner scorer. It produced a useful benchmark, but
it is not the requested architecture.

Frozen archive:

    git tag: legacy-botsort-v1
    commit: a9bed6c

Do not revive its association/lifecycle implementation in the active model.

## Next concrete step

Build the contiguous-clip dataset + minimal recurrent slot model and prove a
single training batch can overfit. Only after that works should we spend compute
on a full mixed-dataset run.

## Stage-0 result - 2026-10-01

The first temporal-slot scaffold is live:

- pretrained YOLO26n visual features
- 64 recurrent track slots
- 128D bounded state per slot
- 8 contiguous 640px frames
- temporal head only; YOLO visual weights frozen
- no external tracker in the model path

One real DanceTrack training clip contains 56 visible person targets across the
8 frames. Repeating only that clip for 100 optimization steps reduced sequence
loss:

    step 1:   2.3941
    step 50:  0.4509
    step 100: 0.2011

On the trained clip:

    running memory loss:     0.2034
    reset every frame loss:  0.2326
    reset / running:         1.143

So the minimal model can overfit a real ordered clip, and removing temporal
state makes that fit about 14% worse. This is only a smoke test, not evidence of
generalization, but it passes the first architecture gate.

Next: train across many DanceTrack training clips and measure whether running
memory beats reset memory on unseen CALIBRATION/DEV clips before adding
PersonPath22 or SportsMOT.
