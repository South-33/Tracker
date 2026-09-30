# Temporal YOLO Tracker

This file is the stable research operating system for the project. It should
contain durable context, rules and evidence that a fresh agent will still need
later. Do not use it as a scratchpad, task list or place to write hypotheses
before they are tested.

## 1. Project goal

Post-train pretrained YOLO26n on ordered human video so one causal neural model
learns both person detection and anonymous identity continuity:

    current frame + bounded neural memory
        -> person boxes + confidence + stable track slots + updated memory

The core bet is that YOLO26n already has strong visual knowledge of people but
standard image detection training does not teach persistent temporal state.
Sequence post-training with boxes + person IDs may teach motion, pose changes,
crossings, occlusion, disappearance and recovery directly in the network.

The model, not external tracking code, must decide identity continuity.

## 2. What counts as the project result

Allowed in the active model:

- pretrained YOLO26n visual features
- bounded learned temporal memory
- learned neural track slots / recurrent state
- ordinary tensor post-processing such as confidence filtering

Not allowed to own identity at inference:

- BoT-SORT or ByteTrack
- Kalman tracking
- online Hungarian association
- external ReID
- hand-written birth/lost/recovery identity rules
- a second tracker hidden behind YOLO outputs

Those methods may exist only as frozen comparison baselines.

Training-only target assignment is allowed. For example, matching a ground-truth
identity to a neural slot when it first appears in a training clip is supervision,
not inference-time tracking.

## 3. Data policy

Train on human video that covers complementary motion regimes:

- **PersonPath22**: broad real-world people, pose, scale, viewpoint and occlusion.
- **DanceTrack**: similar-looking people, crossings, deformation and crowds.
- **SportsMOT**: running, acceleration, rapid direction changes, overlap and
  camera motion.

Benchmark-only:

- held-out DanceTrack
- held-out SportsMOT
- MOT17
- MOT20 only when an extreme-crowd stress test is useful

Do not train on benchmark-only videos.

Training samples are contiguous ordered clips. Randomize which clip is sampled,
not the order of frames inside it. Spatial/color augmentation that changes
geometry or appearance across a clip must be temporally consistent unless the
experiment explicitly studies otherwise.

## 4. Research loop

This is the default loop for every agent.

### Reassess

Read `AGENTS.md`, this file, the active source, recent commits and relevant
existing run artifacts. Work out what is already known and what uncertainty is
actually blocking progress.

Do not write the hypothesis or plan into this file before running it.

### Search before building

Check Git history, active code and existing runs before creating a new
experiment. If essentially the same idea has already been tested, reuse the
evidence instead of repeating it.

### Choose a meaningful shot

Use judgment about scale. The best next move is the one with the highest
expected value for understanding or advancing the model, not automatically the
smallest or safest one.

It is valid to:

- spend a long stretch reading papers, source code, benchmark protocols or
  dataset documentation before implementing anything
- follow a strong technical intuition even when it is risky
- replace an architecture instead of patching it
- run a long training job or use substantially more data when a short probe
  would not answer the real question
- build a focused prototype that looks very different from the current model

Small probes are valuable when they can cheaply answer the real uncertainty.
They are not a requirement. Do not break a serious idea into so many tiny safe
steps that the project never actually tests the idea.

The discipline is that a shot should have a reason: it should test a capability,
challenge an assumption, explore a promising direction, or meaningfully raise
the ceiling. Avoid motion that only makes the repository busier.

For temporal candidates, the running-memory vs reset-memory comparison is a
required control. A model that performs similarly with memory reset has not
demonstrated the capability this project is trying to learn.

### Decide

Every experiment ends with one of three decisions:

- **Promote**: evidence is strong enough to become part of the active design.
- **Reject**: the idea did not help enough; do not keep tuning it by default.
- **Redirect**: the result exposed a different bottleneck, so change the
  approach rather than patching the failed idea.

"Interesting" without a decision is not a completed experiment.

### Record only durable evidence

After the result is known, add a compact entry to **Established evidence** only
if it will prevent future agents from repeating work or materially changes what
the project believes.

A durable entry should include:

- what was actually tested
- the minimum configuration needed to reproduce the conclusion
- the important numbers
- the decision

Do not copy raw logs, brainstorming, future plans, or every failed parameter
value into this file. Generated run folders hold detail; Git history holds old
implementations.

### Housekeep immediately

After deciding:

- delete rejected one-off code
- keep reusable code only when it serves the active path
- remove stale caches and obsolete scripts
- keep generated data/weights/runs untracked
- update tests when a promoted invariant changes
- run `python scripts/check_repo.py`
- make a meaningful checkpoint commit on `main`

The active tree should describe the current project, not every project it has
ever been.

## 5. Evidence standards

Use the weakest claim supported by the evidence.

- One-clip overfit proves trainability, not generalization.
- Lower training loss does not prove tracking.
- Memory is useful only if running memory beats a reset-memory control on unseen
  sequences.
- A development improvement is not a final result until checked on a sealed
  holdout.
- Holdout data becomes consumed after evaluation and must not guide later
  tuning.
- Speed measured on the RTX 4060 is development evidence, not Jetson
  verification.

Prefer HOTA, DetA, AssA, IDF1, ID switches, recall, precision and FPS for
tracking evaluation. Also inspect behavior by motion/crowd regime when a single
aggregate would hide the failure mode.

## 6. Freedom without drift

The project should remain understandable by one agent after reading these two
Markdown files and the small active source tree.

Agents have broad research freedom. New heads, losses, memory structures,
training curricula, architectures, datasets and substantial rewrites are all
allowed when they are plausible ways to attack the goal.

The constraint is not "stay simple at all costs." The constraint is "earn the
complexity." A complicated approach is fine if it creates a capability or gives
us a real test that a simpler one cannot.

Likewise, do not give up on a promising direction merely because the first small
probe fails. If the agent has a concrete reason to believe scale, data, training
time or a structural change could unlock it, take the larger shot. What should
stop is blind repetition: many variants that fail for the same understood reason
without introducing new information.

Prefer deleting failed machinery over leaving dormant alternatives in the
active tree.

## 7. Current neural foundation

The first implementation is intentionally small:

- official pretrained YOLO26n visual backbone/neck
- 640 px person-only video
- 64 persistent neural track slots
- 128D state per slot
- 8 contiguous frames during the initial training experiments
- temporal head attends to current YOLO feature maps and recurrently updates
  each slot
- each slot predicts alive/confidence + normalized box
- slot index is the anonymous identity
- no external identity association at inference

This architecture is a starting point, not a sacred design. Change it when
evidence shows a simpler or better structure is needed.

## 8. Established evidence

### 2026-10-01 - Stage-0 temporal smoke test: PROMOTE FOUNDATION

One real 8-frame DanceTrack training clip contained 56 visible person targets.
YOLO26n visual weights were frozen and only the 64-slot / 128D temporal head was
trained for 100 steps.

    sequence loss:              2.3941 -> 0.2011
    fitted running-memory loss: 0.2034
    reset-every-frame loss:     0.2326
    reset / running:            1.143

Conclusion: the minimal recurrent-slot model can overfit a real ordered clip,
and its fitted result uses temporal state enough that resetting memory makes the
loss about 14% worse. This establishes trainability only. It does not establish
generalization or competitive tracking quality.

### Legacy external-tracker direction: ARCHIVED

The previous YOLO26n + learned appearance + project-owned classical association
system is preserved at Git tag `legacy-botsort-v1` (commit `a9bed6c`).

It was a useful baseline, but it is not the requested architecture. Do not
reintroduce its association/lifecycle machinery into the active model simply
because it already works.

## 9. Deployment constraint

The eventual target remains >=15 FPS on Jetson Orin Nano Super 8GB.

Do architecture research on the RTX 4060 first. Prefer operations that can later
map cleanly to FP16/TensorRT, but do not distort the temporal-learning experiment
to optimize Jetson deployment before the model has demonstrated useful tracking.

Final Jetson acceptance must be measured on the actual target hardware.
