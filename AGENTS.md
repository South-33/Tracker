# Agent Rules

Read `tracker.md` when you are new to the project or when the current state is
unclear.

## Goal

Post-train YOLO26n on ordered human video so the model itself can:

- detect people with bounding boxes
- give each person an anonymous tracking ID
- keep the same ID for the same person across frames
- recover the same ID after short occlusions or disappearances when possible

The model may carry a small fixed-size learned state from previous frames. The
exact form of that state is not part of the goal and may change.

## Hard boundaries

- Do not use BoT-SORT, ByteTrack, Kalman tracking, online Hungarian matching,
  external ReID, or hand-written tracking rules to assign IDs when the active
  model runs.
- Classical trackers are comparison baselines only.
- Train on ordered video clips with person track IDs. Do not treat frames in a
  clip as unrelated images.
- When the model runs, use only the current frame and information carried from
  earlier frames. Never use future frames.
- Keep carried state fixed-size so memory use does not grow with video length.
- For serious temporal models, compare normal carried state against resetting
  the state every frame.
- Do not train on benchmark-only data.

## How to work

- Work only on `main`.
- Start from the latest meaningful result and identify the biggest thing the
  model is failing to do.
- Before spending a lot of time building a fix, test whether fixing that problem
  would actually improve tracking. It is fine to use ground truth or a temporary
  shortcut for this diagnostic.
- Take real research shots. You may read papers, inspect other implementations,
  change the architecture, train longer, use more data, or follow a strong
  technical intuition when it directly serves the goal.
- Avoid work that only creates activity: repeated tiny parameter changes,
  cosmetic refactors, large sweeps without a reason, or infrastructure that does
  not answer a research question.
- After an experiment, decide whether to keep the idea, stop pursuing it, or
  change direction.
- Add only durable results and conclusions to `tracker.md`. Do not put future
  plans, brainstorming, or per-pass notes there.
- Delete rejected one-off code and stale generated files. Keep the active repo
  focused on the current approach.
- Git history is the archive. The previous external-tracker system is preserved
  at tag `legacy-botsort-v1`.
- Use commits as checkpoints. Do not create normal research worktrees or backup
  branches.
- Before committing, run `python scripts/check_repo.py`.

If the active solution becomes "YOLO detections -> another tracker that assigns
the IDs," stop. That is the archived direction, not this project.
