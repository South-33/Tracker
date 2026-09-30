This is the project's AGENTS.md.

# Non-negotiable project goal

Build one small causal neural person tracker by post-training YOLO26n on ordered
human video sequences.

    frame + bounded neural memory
        -> boxes + confidence + anonymous track slots + updated memory

The active model itself must own identity continuity.

## Anti-drift rules

- No BoT-SORT, ByteTrack, Kalman filter, Hungarian online association, external
  ReID model, owner scorer, or hand-written ID lifecycle in the active inference
  path.
- Those systems may exist only as frozen comparison baselines.
- If a proposed solution says "YOLO detections -> tracker", stop. That is the
  legacy direction, not the goal.
- The slot/memory state must be bounded and causal. No future frames at
  inference.
- Train on contiguous ordered clips with track IDs. Do not turn video training
  into shuffled independent frames.
- Temporal augmentations must be consistent across each clip.
- A memory-reset ablation is mandatory for every serious candidate.
- Prefer one clear temporal architecture over stacks of auxiliary models.

## Dataset policy

Active training datasets:

- PersonPath22: varied real-world human video.
- DanceTrack: difficult crossings/crowd association.
- SportsMOT: running and fast motion.

Benchmark-only:

- held-out DanceTrack/SportsMOT splits
- MOT17
- optionally MOT20 for extreme crowds

Do not train on benchmark-only videos.

## Research workflow

1. Start from official pretrained YOLO26n.
2. Preserve its useful person-detection knowledge.
3. Add the smallest bounded neural temporal memory that can plausibly maintain
   anonymous person slots.
4. Train on ordered clips with boxes + track IDs.
5. Compare normal memory vs memory reset.
6. Compare against stock YOLO26n + BoT-SORT.
7. Only add complexity when a focused experiment proves a missing capability.

The original Jetson Nano Super >=15 FPS target still matters, but architecture
research happens on the RTX 4060 first. Prefer operations that can later map to
FP16/TensorRT, but do not weaken the core temporal-learning experiment just to
optimize deployment early.

## Repository rules

- Work only on main.
- Use Git commits as checkpoints. Push meaningful checkpoints.
- No normal research worktrees or backup branches.
- Keep the active repo small. Git history is the archive.
- The old tracker is archived at tag legacy-botsort-v1.
- Do not reintroduce legacy tracker code merely because it already works.
- Generated data, runs and weights remain untracked.
- README.md defines the active architecture and dataset plan.
- tracker.md records only current temporal-model decisions and experiments.
