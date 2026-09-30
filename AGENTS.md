This is the project's AGENTS.md

# Project rules

- Work only in the main repository worktree on branch `main`. Do not create
  extra Git worktrees or research branches. Use small checkpoint commits on
  `main`, push them, then continue.
- Before any risky reset, destructive cleanup, or large refactor, make and push
  a checkpoint commit first. Recover with Git history instead of backup
  branches/worktrees.
- `README.md` is the short operating guide. `tracker.md` is the detailed
  research notebook.
- Read `tracker.md` first.
- Optimize for the next real experiment, not framework completeness.
- Treat the current tracker bet as the incumbent. Do not change it merely to make progress: first measure it, identify a concrete weakness, and only replace or complicate it when a focused experiment gives evidence that the change is meaningfully better. "No change" is a valid result.
- Work with broad research freedom inside that constraint: inspect data and failure cases, research papers/implementations, question the current formulation, rewrite or remove parts of the approach, and run ambitious experiments when they have a clear hypothesis. Be conservative about what gets promoted, not about what gets investigated.
- Keep the active repo small. Delete stale code and use Git history as the archive.
- No legacy compatibility unless the current experiment needs it.
- Prefer simple direct code and few files.
- Heavy/generated data, weights and runs stay untracked.
- A result is only real when its settings, data and metrics are reproducible.
- Housekeeping is part of the research loop: keep the tree tidy, make meaningful
  Git checkpoint commits, and push useful checkpoints/results to `main`
  instead of letting validated work live only locally.
- For learned appearance experiments, compare against both the official frozen baseline and the identical manual detector + BoT-SORT path with ReID disabled. Do not attribute detector-preprocessing differences to the embedding.
- Before attributing a gain to appearance training, also compare against raw ROI-pooled YOLO features and the seeded untrained 64D projection exposed by `scripts/track.py --feature-mode`.
- Dataset roles are defined only in `src/tracker/splits.py`. Never duplicate
  split lists inside scripts.
- TRAIN may update weights. CALIBRATION may tune thresholds. DEV may guide the
  online hill-climb.
- CONSUMED_HOLDOUT is historical evidence only and must never guide a new
  change.
- RESERVED_HOLDOUT is sealed until a candidate is frozen. Local 0082/0083 are
  partial 120-frame slices, so do not score them until the complete official
  sequences are installed.
