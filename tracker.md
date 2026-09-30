# Temporal YOLO Tracker

This file contains the parts of the project that should stay useful over many
research passes: the goal, data rules, research loop, evidence standards,
current model, and results that future agents should know.

Do not use this file for brainstorming, todo lists, or ideas that have not been
tested yet.

## 1. Project goal

Start from pretrained YOLO26n and post-train it on ordered human video with
person track IDs.

We want YOLO26n to keep doing person detection, but also learn tracking itself:

- output a bounding box for each visible person
- output an anonymous ID for each person
- keep the same ID for the same person from frame to frame
- keep that ID through motion, turning, crossings, and partial occlusion
- recover the same ID after a short disappearance when the video gives enough
  information to do so

The model may carry a small fixed-size learned state from previous frames so it
has some context about what it has already seen. The exact form of that state is
an implementation choice, not part of the project goal.

The main bet is simple:

YOLO26n already knows a lot about what people look like. Standard object
detection training mostly teaches it from individual images. It is not normally
trained to understand that frame 2 follows frame 1, that a person moved from one
place to another, turned around, crossed someone else, disappeared, and came
back.

If we post-train YOLO26n on ordered video clips with person IDs, while letting it
carry a small amount of information from previous frames, maybe the model itself
can learn tracking without needing BoT-SORT or another tracker after it.

That is the project.

## 2. What counts as the project result

The active model may use:

- pretrained YOLO26n features
- learned state carried from previous frames
- learned layers that use current features together with previous-frame state
- normal tensor post-processing such as confidence filtering

The following may be used for comparison or diagnostics, but they must not
assign IDs when the active model is running:

- BoT-SORT
- ByteTrack
- Kalman tracking
- online Hungarian matching
- an external ReID model
- hand-written rules for when an ID starts, disappears, returns, or changes
- another tracker placed after YOLO detections

During training, it is fine to use matching code to tell the model which target
identity it should learn. That is supervision, not inference-time tracking.

## 3. Data policy

Training data should cover different kinds of human motion instead of relying on
one style of video.

- **PersonPath22**: everyday people, different scales, poses, camera views, and
  occlusions.
- **DanceTrack**: people crossing, similar appearances, deformation, and crowds.
- **SportsMOT**: running, acceleration, fast direction changes, overlap, and
  moving cameras.

Benchmark-only data:

- held-out DanceTrack
- held-out SportsMOT
- MOT17
- MOT20 when an extreme-crowd test is useful

Do not train on benchmark-only videos.

A training sample is an ordered contiguous clip. Keep the frame order inside the
clip.

If an augmentation changes the scene geometry or appearance, apply it
consistently across the clip unless an experiment is specifically testing
something else.

## 4. Research loop

Use this loop repeatedly.

### 1. Find the biggest weakness in the latest result

Start from the latest meaningful result and ask what the model is actually
failing to do.

Examples:

- it misses people
- IDs switch when people cross
- it forgets people after an occlusion
- it performs the same even when previous-frame state is removed
- it handles walking but not running
- it needs more past context
- it is too slow
- the training data does not contain enough of the failure case

Use diagnostics when the failure is unclear.

Do not begin with "what parameter can I tweak?" Begin with "what is the model
failing to do?"

### 2. Test whether fixing that weakness would matter

Before spending a lot of time teaching the model a new capability, temporarily
give the system that capability in the easiest possible way and measure the
result.

This is only a diagnostic. It does not need to be a valid final solution. It may
use ground truth or information the final model would not be allowed to use.

The purpose is to answer one question:

> If this problem were fixed, would the tracker improve enough to make it worth
> working on?

Examples:

- use ground-truth person boxes to measure how much missed detections are
  limiting tracking
- give the correct previous position to measure how much better motion knowledge
  could help
- keep the correct ID through an occlusion to measure how much better recovery
  could help
- give much more previous context to test whether the current context is too
  short
- compare normal previous-frame state with resetting it every frame to test
  whether the model is actually using temporal information

If this temporary fix barely improves the result, do not spend a large amount of
time teaching the model that capability. Look for a more important weakness.

If it improves the result a lot, then we have evidence that the capability is
worth learning properly.

### 3. Try to make the model learn it for real

Now build or train a realistic solution that gives the model that capability
without the temporary shortcut.

Use the scale the idea deserves. This may mean:

- a small diagnostic experiment
- spending time reading papers or source code first
- changing the architecture
- training for much longer
- using more or better-targeted data
- increasing the amount of previous context
- trying a risky idea because there is a good technical reason for it

The experiment does not need to be small or safe. It needs to have a clear
reason and a real chance of teaching us something useful.

For models that carry state between frames, compare normal carried state against
resetting that state every frame. If resetting it does not meaningfully hurt
tracking, the model has not shown that it learned useful temporal tracking.

### 4. Decide, record, and clean up

After the result, make a clear decision:

- **Keep it** if it meaningfully advances the project.
- **Stop pursuing it** if the evidence says it is not helping enough.
- **Change direction** if the result reveals that a different problem matters
  more.

Only after the experiment is finished, add a short result to **Established
evidence** if it will help a future agent avoid repeating work or understand why
the project changed direction.

Record what was tested, the important numbers, and the conclusion. Do not record
brainstorming or future plans.

Delete rejected one-off code and stale files, keep only reusable active code,
update tests when needed, run `python scripts/check_repo.py`, checkpoint `main`,
and continue.

Reading all repository context is useful when an agent is new or confused. It is
not a required step before every experiment.

## 5. Evidence standards

Say only what the evidence supports.

- Overfitting one clip proves the model can learn that clip. It does not prove
  generalization.
- Lower training loss does not prove better tracking.
- Previous-frame state is useful only if the model performs meaningfully worse
  when that state is reset on unseen video.
- A development improvement is not a final result until it is checked on data
  that was not used to choose the model.
- Once a holdout result has been seen, do not use that holdout to tune the next
  model.
- Speed on the RTX 4060 does not prove speed on Jetson hardware.

For tracking, report HOTA, DetA, AssA, IDF1, ID switches, recall, precision, and
FPS when they are relevant. Also look at the specific failure type when an
average score hides what is going wrong.

## 6. Freedom without going in circles

Agents have broad research freedom. They may change the model, losses, previous-
frame state, training method, datasets, context length, or large parts of the
implementation when there is a good reason.

Do not stay simple just for the sake of being simple. A more complicated idea is
fine if it is needed to create a useful capability.

Also do not keep trying many small versions of the same failed idea when they
keep failing for the same understood reason. Either make a meaningful change to
the idea or move to a different weakness.

Keep the active repository easy to understand. Delete failed machinery instead
of leaving many unused alternatives in the active tree.

## 7. Current prototype

The current implementation is only the first way we are trying the idea. It is
not the definition of the project.

Right now it uses:

- pretrained YOLO26n visual features
- 640 px input
- 8 ordered frames during the initial training experiments
- 64 learned person slots
- a 128-value learned state for each slot
- a learned module that looks at current YOLO features and updates those slots
- each slot predicts whether a person is present and where their box is
- the slot number acts as the anonymous ID

There is no external tracker assigning IDs when the model runs.

If another learned state design works better, we should replace this one.

## 8. Established evidence

### 2026-10-01 - First temporal training smoke test: KEEP AS FOUNDATION

The current 64-slot / 128-state prototype was trained on one real 8-frame
DanceTrack clip containing 56 visible person targets. YOLO26n visual weights were
frozen and only the temporal part was trained for 100 steps.

    sequence loss:              2.3941 -> 0.2011
    normal carried-state loss:  0.2034
    state-reset-every-frame:    0.2326
    reset / normal:             1.143

Conclusion: the prototype can learn one ordered clip, and resetting its carried
state makes the fitted result about 14% worse. This only proves that the current
prototype can learn and use some previous-frame state on that clip. It does not
prove that tracking generalizes to unseen video.

### Previous external-tracker direction: ARCHIVED

The old YOLO26n + learned appearance + classical association system is preserved
at Git tag `legacy-botsort-v1` (commit `a9bed6c`).

It remains useful as a comparison baseline, but it is not the requested
architecture. Do not bring its ID-assignment logic back into the active model
just because it already works.

## 9. Deployment requirement

The eventual target is at least 15 FPS on Jetson Orin Nano Super 8GB.

Do architecture research on the RTX 4060 first. Keep future deployment in mind,
but do not weaken the tracking experiment before the model has shown useful
tracking behavior.

Final Jetson speed must be measured on the actual Jetson hardware.
