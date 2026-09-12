# Goal: reach useful tracking first

Build one causal, stateful neural person tracker that keeps the same anonymous identity through people walking together, crossing, turning, overlapping, disappearing, and returning seconds later. Train on this laptop's RTX 4060; target one camera at **at least 15 FPS on Orin Nano Super 8 GB**.

The user's priority is a substantial move into the useful **80% range**, then incremental optimization. Do not spend the night making a weak system 1% better. Take large, reasoned bets. Failed experiments are welcome when they change what we know and improve the next bet. Novelty is optional; reaching the behavior is mandatory.

Read `AGENTS.md`, this file, `tracker.md`, then run `.\lab.ps1 status`. Create an active goal from this document when asked to run the goal. Work autonomously within it; preserve enough usage/context to checkpoint and hand off.

## Behavior and implementation contract

```text
step(rgb_t, dt_seconds, previous_state)
    -> current visible boxes, person scores, anonymous public IDs,
       continuity confidence, new_state
```

- New people can enter at any time. An initial two-second roster is a diagnostic shortcut, not an acceptable final tracker.
- Keep identity through overlap and short absence. Measure return gaps in seconds, initially 0.5-10 seconds. Never initialize identity from GT during a claimed system evaluation.
- Emit current observations only. Remember an absent person internally; do not draw an old box as if observed now. Abstention is allowed but counts against coverage.
- One deployed neural model plus bounded state and bookkeeping. External matching trackers, multiple hypotheses, larger vision models, and offline optimizers may be references or training teachers. Their runtime costs and any non-neural identity solver must not be hidden inside the final "one model" claim.
- Public IDs do not recycle during a run. Internal slots may be reused after documented expiration while allocating a new public ID. Report capacity misses and expiration failures. Immutable canonical embeddings, frozen encoders, a particular slot count, and the present enrollment scheme are hypotheses, not permanent architecture rules.
- An LLM is optional. Use it, a stronger tracker, motion, pose, or geometry supervision only if it supplies useful training information. A teacher's failure in one sampling/readout setup does not disprove that entire source of information.
- Keep this project standalone. Do not integrate or deploy to SmartCampus as part of research. Nano performance must be measured on the Nano; keep its operational rules and restore services after any permitted benchmark.

## What "80%" means here

These are explicit project targets, not published benchmark thresholds. Freeze them before promotion; do not lower them after looking at results.

Use version 2 of `TrackingAccumulator` alongside official TrackEval. A **covered identity segment** is a contiguous annotated visible segment with the same public identity, no switch or cross-person ID reuse, and detections on at least 80% of its visible frames at IoU >=0.5. Every GT segment is in the denominator, including people never tracked. Public ID reuse invalidates segments using that ID. Also report the fraction of visible GT person-frames belonging to successful segments, so many easy short segments cannot hide long failures.

The useful-system milestone requires all of:

| Check | Target on frozen full, uncut development sequences |
|---|---|
| Covered identity segment rate | >=80% |
| Covered identity frame rate | >=80% |
| Overall detection recall / precision | >=80% / >=98% |
| Standard tracking quality | HOTA, AssA, DetA, IDF1, IDSW, fragmentation, FP and misses beside the same frozen reference; investigate regressions |
| New arrivals | Enabled throughout the run; report arrival delay and missed new people |
| Reviewed re-entry | >=80% unconditional correct recovery within 0.5 seconds, followed for two seconds; <=1% wrong-person links among accepted links |

Re-entry needs at least 20 reviewed events across at least three sequences, separated into physical exit/return and occlusion, with denominators and gap durations. Until that evidence exists, report "unverified". If available footage does not contain enough events, acquire appropriate permitted/public data or ask for the missing evidence; do not relabel annotation gaps as physical exits.

`strict_visible_segment_survival` is a **legacy switch-free diagnostic**. It can score 100% while tracking nobody. It is never a promotion metric. Historical values also used a different thresholded-matching implementation and must not be compared numerically to version 2 without replay. This custom metric is not HOTA and does not replace it. [Official TrackEval](https://github.com/JonathonLuiten/TrackEval) provides detection and association metrics separately.

After the development milestone: freeze source, weights, settings and preprocessing; evaluate the architecture gate; then the four final holdouts once; export an uncut heldout demo; verify recurrent ONNX and TensorRT FP16 parity; measure real Nano >=15 FPS including preprocessing, state updates and transfers, memory and thermals on sustained video. A research milestone does not complete the deployment goal.

## Research rhythm: a few probes, then a substantial bet

1. State the largest measured failure and why resolving it could move the whole system toward 80%. Generate a few competing explanations, including a simple removal/inversion and a change in state or learning. Use primary literature to expand and challenge the choices. Do not manufacture novelty for its own sake.
2. Choose one active hypothesis. Write the expected system benefit, changed assumption, decisive measurement, source, inputs, compute budget, and continue/scale/pivot criteria in `research-cycle.json` before running it.
3. **At most three exploratory experiments per cycle, across train-only diagnostics and formal dev probes combined.** Use fewer if evidence is enough. A renamed script, a "diagnostic," or clearing the ledger does not earn more attempts on the same question. Planned evaluations/checkpoints inside a substantial run are part of that run, not a route to unrelated sweeps.
4. Then commit to a substantial experiment or explicitly change direction. A substantial experiment tests the deployable mechanism in complete causal rollouts with its own accumulated mistakes and meaningful training/data/context. It can also be a decisive full-system intervention without learning. The bar is information and scope, not an impressive step count.
5. Finish the declared compute unless there is an execution defect, a demonstrated impossible input condition, or a predeclared stop criterion. Watch learning curves, distinct examples, rollout seconds, and performance outside fitted examples. If the mechanism is learning late, scale it. Do not call 10,000 repeated tiny batches universally "fair scale."
6. Interpret once: what changed, what did not, which assumption is now weaker, and what specific next bet follows. Preserve executable evidence, update compact context, and commit. Keep the best runnable system while developing its replacement.

Small checks are appropriate to catch broken gradients, invalid labels, data leakage, or OOM before a substantial run. Small scientific gains are worth following only if they validate a credible path to a large system gain. Do not optimize thresholds, widths or one loss coefficient merely because an improvement is easy to obtain. Equally, do not abandon a sound mechanism because its first working version has not already reached 80%.

When a cycle fails, distinguish **implementation defect**, **negative at tested scale**, **information missing under this protocol**, and **deployment conflict**. A GT-assisted score is a conditional diagnostic, not generally an upper bound on all algorithms. A weak teacher readout is not proof that motion/appearance/geometry contains no useful information. Broader rejection requires broader evidence.

## Data and comparisons

- `data/manifest-train-full-dense-no0096.json` is the local clean training/dev manifest. Its `available_frames` lists, not the legacy top-level frame-limit label, define coverage. Some dense training sequences remain short prefixes.
- `dancetrack0012` has been inspected repeatedly and is **development data for architecture selection**, even when omitted from gradient updates. It is not fresh generalization evidence. `0016` and `0020` have also been repeatedly used for dev. Record exposure honestly.
- Never train on `0096`, the architecture gate. Keep final `0004`, `0005`, `0007`, `0010` untouched until the frozen final evaluation. Verify manifests and source ancestry instead of trusting filenames alone.
- Compare identical full sequences, annotations, detector settings, timing, and score definitions. Store public predictions and evaluate every frame, including misses, without resetting around difficult events.
- Keep oracle initialization, candidate filtering, future information, perfect enrollment and selected continuously detected episodes visibly labeled. Remove those advantages for system claims. Do not compare a 66% selected-event result with an 80% whole-video target.
- DanceTrack stresses similar appearance and varied motion. Add representative footage for ordinary arrivals/re-entry before claiming the user's broader behavior. Its benchmark purpose is documented by the [dataset authors](https://github.com/DanceTrack/DanceTrack).

## Useful research starting points, not a prescribed architecture

| Direction | What to inspect | What this repository has not established |
|---|---|---|
| Learned ID prediction | [MOTIP official code](https://github.com/MCG-NJU/MOTIP): trajectory context, ID training, handling new identities, full inference rules | A small frozen-feature classifier's failure does not reproduce or refute the published system; audit any runtime assignment against our contract |
| Temporal query learning | [MeMOTR paper](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html): memory injection and tracking-specific processing | A frozen detector feature head does not test temporal perception trained with the complete objective |
| Distillation / uncertain state | Stronger offline or larger teachers, bounded learned uncertainty, delayed identity decisions | Teacher quality and student deployment cost must be measured independently; GT-selected bundles remain component experiments |

Search newer primary work when it can change the decision. Read its actual training/data/inference code, not just benchmark tables. Record the precise borrowed mechanism and the evidence that would falsify the adaptation. Avoid a growing literature dump in active context.

## First work after this context reset

1. Restore one complete reference replay path and standard scoring on the already-exposed full dev sequences. The previous runnable implementation is in Git at `92da962`; inspect it rather than resetting the current tree. The clean local detector checkpoint is listed in `tracker.md`. An external matching tracker is permitted as a **reference only**. Freeze its exact source/weights/settings and keep it runnable.
2. This is one bounded infrastructure repair, not a new research project: aim to have the reference path running within the first hour. Report an actual missing dependency/data issue promptly. Do not build a dashboard or generic experiment platform.
3. From that comparison and the existing failure evidence, choose the first substantial bet. At most three experiments may resolve decisive unknowns first. The paused joint-bundle idea is an option, not an obligation. Temporal perception, a faithfully adapted published learner, distillation, or another justified reformulation remain open.
4. The next handoff should contain a reproducible reference and either a completed substantial causal experiment or a resumable one with elapsed compute, learning evidence and exact continuation commands. More small diagnostic files alone do not satisfy this work interval.

## Handoff and context maintenance

`tracker.md` owns current scientific facts and the next decision. `research-cycle.json` owns active machine state. This goal owns intent, acceptance and research behavior. `README.md` owns entry commands. Avoid duplicating mutable status across them.

After a result, remove superseded instructions and unsupported family bans from active context. Preserve the observation, limitation and useful inference in a short evidence row. Archived failures should be fetched only when relevant, not reread at every start. Delete scratch only after its exact generating source, configuration and result are archived and committed. A JSON number alone is not reproducibility.

Keep laptop work behind `lab.ps1`; use one GPU-heavy job, deterministic frozen caches only where valid, and retain the activity-aware CPU guard. Improve measured workflow delays, but stop harness work once the next scientific experiment is reliable and easy to run.

Commit coherent decisions and push to `South-33/Tracker` without force-pushing. Keep data, checkpoints and caches ignored. Before usage/context ends, save atomic resumable checkpoints and update the next action with exact commands and remaining acceptance gaps. Never mark the goal complete merely because a run finished or a budget ended.
