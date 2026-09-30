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

Use this simple loop indefinitely:

### 1. Find the weakness

Start from the latest meaningful result. Ask what is actually failing or holding
the model back: detection, identity continuity, occlusion recovery, motion,
memory usage, speed, data coverage, or something else.

Use diagnostics when needed. Do not start by asking "what can I tweak?"

### 2. Paint it red

Before spending a lot of effort building a solution, test whether the suspected
capability would matter if we could simply give it to the system.

Use an oracle, forced input, exaggerated intervention, privileged signal or
temporary cheat when possible. The point is not to build something deployable.
The point is to answer:

> If this weakness were fixed, would the tracker actually get meaningfully better?

Examples:

- give perfect previous-frame position to test whether motion is the bottleneck
- preserve the correct identity through an occlusion to measure recovery
  headroom
- give much longer context to test whether context length is limiting
- substitute ground-truth detections to separate detection from association
- force memory on/off to measure whether the model is using temporal state

If the red-painted/oracle version barely helps, the suspected weakness is
probably not worth learning properly. Find the next weakness.

If it helps a lot, there is real headroom. Now we know what capability is worth
building.

### 3. Take the real shot

Build or train a realistic way for the model to learn that capability.

Use whatever scale makes sense. It can be a quick probe, literature exploration,
a risky architecture change, substantially more data, or a long training run.
Good intuition is a valid reason to try something. Do not make an idea artificially
small just to be cautious.

The requirement is not "small experiment." The requirement is "meaningful
experiment."

For temporal candidates, running-memory vs reset-memory remains a required
control. If memory can be reset without meaningful loss, the model has not
demonstrated the temporal capability we care about.

### 4. Decide, record, clean

End with a decision:

- **Promote**: keep it because evidence says it advances the goal.
- **Reject**: stop pursuing it by default.
- **Redirect**: the experiment revealed a different bottleneck; attack that
  instead.

Only after the result is known, write durable evidence into **Established
evidence** when it will matter to a future agent. Record the tested idea,
important numbers and conclusion, not brainstorming or future plans.

Then delete rejected one-off code and stale clutter, keep only reusable active
code, update tests when needed, run `python scripts/check_repo.py`, checkpoint
`main`, and continue the loop.

Reading all repository context is a bootstrap action for a fresh or confused
agent, not a mandatory step on every pass.

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
