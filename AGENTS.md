# Agent Rules

Read `tracker.md` before doing research work.

## Goal

Build one small causal neural person tracker by post-training YOLO26n on ordered
human video:

    frame + bounded neural memory
        -> person boxes + confidence + anonymous track slots + updated memory

The neural model itself must own identity continuity.

## Hard boundaries

- No BoT-SORT, ByteTrack, Kalman filter, online Hungarian association,
  external ReID model, owner scorer, or hand-written ID lifecycle in the active
  inference path.
- Classical trackers are comparison baselines only.
- Training uses contiguous ordered clips with track IDs, never shuffled
  independent frames pretending to be video.
- Inference is causal and memory is bounded.
- Every serious candidate gets a running-memory vs reset-memory ablation.
- Do not train on benchmark-only data.

## How to work

- Work only on `main`.
- Before experimenting, read existing code, `tracker.md`, recent commits and
  relevant existing runs so completed work is not repeated.
- Use judgment. You are allowed to spend substantial time researching papers,
  implementations, datasets or related ideas before touching code when that can
  change the quality of the approach.
- Take real shots. A good experiment may be a tiny probe, a new architecture, a
  long training run, a larger dataset pass or a risky intuition-driven idea.
  Choose the scale that matches the expected information or upside.
- Do not confuse caution with rigor. Bold changes are welcome when they attack
  the project goal directly and can teach us something important.
- Do not confuse activity with progress either. Avoid endless micro-tuning,
  cosmetic refactors, broad sweeps without a reason, or building infrastructure
  that does not answer a research question.
- After an experiment, make a decision: promote, reject, or change direction.
- Record only durable evidence/decisions in `tracker.md`; do not log speculative
  hypotheses or per-pass plans there.
- Delete rejected one-off code and stale generated clutter. Keep only reusable
  code that serves the active goal.
- Git history is the archive. The previous tracker is tagged
  `legacy-botsort-v1`.
- Use commits as checkpoints. No normal research worktrees or backup branches.
- Before committing, run `python scripts/check_repo.py`.

If the active path becomes "YOLO detections -> external tracker", stop: that is
the archived direction, not this project.
