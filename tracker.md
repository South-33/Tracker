# Tracker

<!-- cycle-status: decision_required -->

One document for the goal, current evidence, next decision and commands. `AGENTS.md` holds lean agent rules; `research-cycle.json` holds execution state. Do not create separate goal, readme, results or experiment Markdown files.

## Goal and priority

Build one causal, stateful neural person tracker that preserves anonymous identity through walking together, crossing, turning, overlap, occlusion, and leaving the frame and returning seconds later. Train on this laptop's RTX 4060; target one camera at **>=15 FPS on Orin Nano Super 8 GB**.

**Reach the useful 80% regime first.** Take substantial, reasoned bets. Do not spend the night making a weak tracker 1% better. Failures are useful when they narrow uncertainty and improve the next bet. Novelty is optional. A small improvement matters when it validates a credible route to a large system gain, not merely because it is easy to obtain.

```text
step(rgb_t, dt_seconds, previous_state)
    -> current visible boxes, person scores, anonymous IDs,
       continuity confidence, new_state
```

New people must be admitted throughout the run. Remember absent people internally; never publish stale boxes as current observations. Initially evaluate absence/re-entry over 0.5-10 seconds. Abstention is allowed and counts against coverage. No GT identity initialization or future frames in claimed system evaluation.

Deploy one neural model plus bounded state/bookkeeping. External matching trackers, offline optimizers and heavyweight teachers may serve as references or training supervision; do not hide them in the deployed one-model claim. Public IDs never recycle within a run. Internal slots may expire and be reused with new public IDs; report capacity/expiry misses. Frozen encoders, immutable canonical embeddings and the existing two-second enrollment are experimental choices, not requirements. An LLM is optional.

Keep the project standalone. No SmartCampus integration or deployment during this research. Measure Nano performance on the Nano, follow its operational rules and restore services after permitted benchmarks. Laptop results do not establish Nano performance.

## Start here: current decision

Updated 2026-09-12. **No tracker has passed version-2 acceptance; there is no validated incumbent.** The active tree has perception/state infrastructure but no complete trainer/replay entry point. Old component scores are not current system progress.

**Provisional diagnosis:** historical runs suggest accumulating identity errors and guards sacrificing coverage. Size has not been isolated. Pair RT-DETR already trained the whole detector. Recheck the diagnosis against the runnable reference.

1. Read this document and run `.\lab.ps1 status`. Create an active goal from it when asked to run the goal. Preserve enough usage/context to checkpoint and hand off.
2. Restore one runnable reference and score the already-exposed full dev sequences. Git `92da962` contains the earlier complete implementation; inspect and adapt relevant code without resetting this tree. `e6dae0b` preserves the accumulated later research. An external matching tracker is allowed as a reference only. Freeze source/weights/settings and keep the reference runnable while developing its replacement.
3. Bound that repair to getting a real comparison running, aiming for the first hour. Report concrete missing data/dependencies promptly. Do not turn it into a dashboard or generic experiment platform.
4. Choose the substantial bet from the comparison and the evidence below. The paused victim-only anonymous-bundle idea is an option, not an obligation. Temporal perception, a faithfully adapted published learner, teacher-guided training or a justified state reformulation remain open.
5. The next handoff must contain a reproducible reference and a completed or resumable substantial causal experiment with measured learning and exact continuation commands. More small diagnostic files alone do not satisfy this interval.

## Decision checklist and research loop

Before choosing a bet, record brief conclusions, evidence paths and uncertainty in the existing ledger declaration. Reuse measurements; probe only unknowns that could change the decision. These questions do not require six experiments or another report.

| Check | Question the agent must resolve or explicitly leave uncertain |
|---|---|
| Locate the failure | What costs most coverage on full causal output: missing boxes, wrong ID choice, corrupted memory, forgetting, or missed arrivals? Inspect the first error in representative windows and check labels. Separate causes from later symptoms; candidate availability is not detector recall. |
| Match training to use | Does training include plausible self-generated mistakes, misses, absence and recovery? Which histories/candidates come from GT and are unavailable at runtime? Clean-history accuracy cannot establish recovery from corrupted state. |
| Diagnose learning before blaming size | What is trainable/frozen? Count distinct sequences/examples and history seconds. Compare fitting and generalization curves. Low training loss with poor generalization implicates data/objective; high loss also warrants checking labels, gradients and optimization. A size claim needs a controlled comparison. |
| Check prior attempts and literature | Find the closest result in the decision table, legacy index and attempts using mechanism/synonyms. State the changed assumption; missing source limits conclusions. Compare published supervision, data, temporal context and inference, not just architecture names. |
| Choose for system impact | Consider competing explanations, a simple removal/inversion and a learning/state change. What could remove a large measured failure class? Predict the effect and contradictory evidence. Judge coverage, identity and precision together; silence cannot count as progress. |
| Give the bet a fair test | Declare one hypothesis, changed assumption, system gain, source/inputs, decisive evaluation, distinct data, temporal span, compute budget and continue/scale/pivot criteria. Compound recipes are allowed; identify essential changes. Freeze a runnable reference and source/input hashes. |

**At most three exploratory experiments per cycle, counting train-only diagnostics and formal dev probes together.** Use fewer when ready, then launch a substantial run or justify a different hypothesis. Renaming/resetting does not buy more variants. Small checks resolve implementation defects or decision-changing unknowns.

A substantial bet tests the deployable mechanism on complete causal rollouts with its own errors and meaningful data/context/compute. A decisive full-system intervention without training also qualifies. Finish declared compute unless a defect, impossible input condition or predeclared stop criterion applies. Planned checkpoints belong to that run. Weak early scores do not justify abandoning a learning mechanism; repeating a few examples is not adequate scale.

**Re-evaluate at the first meaningful planned checkpoint, endpoint and handoff.** Is the predicted failure shrinking, displaced, or hidden by lower coverage? Is fitting improving while generalization stalls? What contradicts the diagnosis? Use existing relevant signals, not fresh sweeps. Record continue/scale/repair/pivot, supporting evidence and the next command. Repeated contradictions or stalled progress require revisiting the cause before another local tweak; review need not interrupt healthy training.

Classify outcomes as **implementation defect**, **negative at tested scale**, **information missing under this protocol**, or **deployment conflict**. Oracle results and weak teacher readouts do not establish universal ceilings. Preserve evidence and update decision memory below. Amend this checklist only for recurring, costly decision errors.

## Acceptance: define 80% before seeing the result

These are project targets, not published benchmark thresholds. Freeze them before promotion and do not lower them after inspecting scores.

Version 2 of `TrackingAccumulator` defines a **covered identity segment** as a contiguous annotated segment using the same public ID, with no switch/cross-person reuse and detections on >=80% of annotated frames at IoU >=0.5. Every GT segment counts, including people never tracked. Reusing a public ID invalidates segments using it. The frame rate weights successful segments by their annotated person-frame counts, preventing many easy short segments from hiding long failures.

Here "visible" means GT annotation present, not independently verified human visibility. Review occlusion cases before making behavior claims. `strict_visible_segment_survival` is a legacy switch-free diagnostic that can score 100% while tracking nobody. **Never promote from it.** Version 2 also changes thresholded matching, so historical numbers need replay before comparison.

| Full uncut dev check | Target |
|---|---|
| Covered identity segment rate / frame rate | >=80% / >=80% |
| Overall detection recall / precision | >=80% / >=98% |
| Standard tracking scores | HOTA, AssA, DetA, IDF1, IDSW, fragmentation, FP and misses beside the same frozen reference; investigate regressions |
| New arrivals | Enabled throughout; report arrival delay and missed newcomers |
| Reviewed re-entry | >=80% unconditional correct recovery within 0.5 s, followed for 2 s; <=1% wrong-person links among accepted links |

Re-entry requires >=20 reviewed events across >=3 sequences, separated into physical exit/return and occlusion, with denominators and gap durations. Until available, report unverified; acquire suitable permitted/public data if needed. Annotation gaps alone do not prove physical exits.

Compare identical full sequences, labels, detector settings, timing and metric versions. Save predictions for every frame without resets around difficult events. Label oracle initialization, GT candidate filtering, perfect enrollment, future context and selected continuously detected episodes. Remove those advantages for system claims. A 66% selected-event score is not comparable with the whole-video 80% target.

After the dev milestone: freeze source, weights and preprocessing; evaluate the architecture gate; then final holdouts once; produce an uncut heldout demo; verify recurrent ONNX and TensorRT FP16 parity; measure sustained Nano FPS including preprocessing, transfers and state updates, plus memory/thermals. Completing a training run or research milestone does not complete this goal. Custom coverage scores supplement [official TrackEval](https://github.com/JonathonLuiten/TrackEval), which measures detection and association separately.

## Decision memory: avoid repeats without banning ideas

The [legacy index](evidence/legacy/index.json) retains 63 historical JSON artifacts; most generating scripts were already missing when archived. These are observations with limited reproducibility, not current acceptance results. No historical result has been replayed under version 2. Consult detailed files only for the mechanism being considered. Prior scope text calling `0012` "heldout" means excluded from that fit, not fresh architecture-validation evidence.

Search result filenames first with `rg -l -i 'mechanism|synonym' evidence/legacy -g '*.json'`. Find newer attempts with `rg --no-ignore -l -i 'mechanism|synonym' evidence/attempts -g record.json`, then read only relevant fields. Ordinary searches exclude archived source/context to avoid pulling superseded instructions into the current task.

| Tested mechanism | Recorded outcome and scope | What a worthwhile revisit must change |
|---|---|---|
| Hard freeze / partial suspect-box updates | [Anchor diagnostic](evidence/legacy/confidence-weighted-anchor-train-diagnostic.json): hard freeze had 98.35% legacy survival at 0.46% recall; partial updates restored recall but corrupted identity. | State semantics or trustworthy information, not just gain/momentum. |
| Local safety plus canonical recovery | [3k-step full-dev result](evidence/legacy/local-recovery-dev.json): 97.46% precision, 36.23% recall, 925 switches. SAFE/local-quarantine variants also sacrificed coverage. | Causal learning/recovery that survives its own errors; prove useful coverage, not clean-link accuracy alone. |
| Fresh hazard guards and retry | [Clean-prefix guard](evidence/legacy/clean-prefix-guard-train-diagnostic.json), [retry guard](evidence/legacy/retry-guard-train-diagnostic.json): clean-state signal did not yield robust causal tracking. | The distribution after false vetoes/misses; another clean-state threshold is not that change. |
| Opening roster / more canonical exemplars | [Census](evidence/legacy/opening-census-train-diagnostic.json), [exemplars](evidence/legacy/opening-exemplar-train-diagnostic.json): limited improvement in tested opening/recovery cases. | Continuous arrivals and a stronger identity-learning objective; opening-only success does not solve the product. |
| Handcrafted geometry / flow / templates | [Sparse flow](evidence/legacy/sparse-flow-train-diagnostic.json), [spatial template](evidence/legacy/spatial-template-train-diagnostic.json), [structural conflict](evidence/legacy/structural-conflict-train-diagnostic.json): local cues had weak discrimination or suppressed many safe links. | Learned/integrated information or a different causal state; another cutoff alone lacks a large-gain case. |
| Identity adaptation / frozen history attention | [Decoder adaptation](evidence/legacy/identity-decoder-train-diagnostic.json), [10k temporal ranker](evidence/legacy/temporal-transformer-train-diagnostic.json), [backward linker](evidence/legacy/backward-temporal-linker-train-diagnostic.json): local learning did not establish reliable system identity. | Representation, training distribution or complete temporal perception; don't equate a frozen-feature head with a published end-to-end model. |
| Protected queries / separate pointer / NULL | [Perception](evidence/legacy/protected-track-query-perception-train-diagnostic.json), [NULL](evidence/legacy/protected-query-null-train-diagnostic.json), [pointer](evidence/legacy/tracking-pointer-train-diagnostic.json): the 1.33M pointer after 10k steps reported 18.95% hard recovery and 76.54% absent false claims. | Correct owner/absence semantics, training exposure or information path; assess full causal output. |
| Raw/pair-conditioned perception / interaction motion | [Pair RT-DETR](evidence/legacy/pair-conditioned-rtdetr-train-diagnostic.json) trained all detector parameters for 10k steps on RGB pairs; [raw motion](evidence/legacy/raw-pair-motion-train-diagnostic.json) and [interaction motion](evidence/legacy/interaction-motion-train-diagnostic.json) also recorded tested-scale negatives. | A specific objective, data or temporal-context change. Unfreezing alone repeats prior work; these runs do not disprove learning motion from pixels. |
| Mask / pose / point teachers | [Masks](evidence/legacy/mask-teacher-train-diagnostic.json), [pose](evidence/legacy/pose-teacher-train-diagnostic.json), [points](evidence/legacy/point-tracking-teacher-train-diagnostic.json): the sampled cue/readout tests did not meet their recovery criteria. | Verify teacher cues/readout and the student supervision before dismissing the whole information source. |
| Joint permutation bundle | [Oracle-selected pairs](evidence/legacy/joint-bundle-oracle-train-diagnostic.json): 93/140 selected continuous episodes recovered; [observable grouping](evidence/legacy/expanding-bundle-train-diagnostic.json): 24/140 with 26.87% safe collateral, still clean GT initialization. | Membership formation and self-owned state without oracle help. Victim-only anonymous entry is untested, not already a win. |

Also search the index for candidate-rank, signatures, shadow/fork, voting, token-match and reanchor diagnostics before proposing those variants. The 99.27% candidate-availability and old 73-74% baseline claims lack surviving supporting artifacts; reproduce before using them as premises. Candidate availability is not deployable detector recall.

**After each cycle, replace or amend its row:** hypothesis; result and scope; evidence path; continue/scale/pivot decision; changed assumption required to reopen. Merge redundant rows. Keep failed-run source/results in Git and `evidence/attempts/`, including negative outcomes. Delete stale prescriptions, not the evidence needed to avoid repetition. Never clear counters or erase the previous decision just to make an old idea look new.

## Assets and data exposure

Input hashes live once in `research-cycle.json.inputs`. Clean detector ancestry is `runs/expanded-0002/last.pt`, loaded by `ancestry.py`. The clean training/dev manifest is `data/manifest-train-full-dense-no0096.json`; actual `available_frames` lists define coverage, not its legacy top-level frame-limit label.

| Sequences | Role and availability |
|---|---|
| `0001/0002/0006/0008/0012/0015` | Full training sequences. `0012` repeatedly used for architecture selection, even when omitted from gradients. |
| `0082/0083` | 120/603 frames each; training prefixes. |
| `0016/0020` | Full dev, 2,163 and 583 frames; repeatedly exposed. |
| `0096` | Architecture gate, never training; historically inspected in earlier diagnostics. |
| `0004/0005/0007/0010` | Reserved final holdouts according to prior records; keep untouched until frozen final evaluation. |

Verify manifests and lineage. Do not call the local subset a full DanceTrack benchmark. [DanceTrack](https://github.com/DanceTrack/DanceTrack) targets similar appearance and varied motion; representative arrivals/re-entry need suitable footage beyond a convenient dance subset.

Two provisional research leads; neither has established our 80% target:

- **Joint ID learning with damaged history.** [MOTIP](https://arxiv.org/html/2403.16848v2) uses trajectory omissions, ID swaps and interaction among current objects. Test broader video training with corrupted history and causal evaluation. Prior guards, clean-history rankers and RGB-pair training do not establish that this combination was tried; verify archived source. MOTIP samples 30 DanceTrack frames but backpropagates through the detector on four: adapt and measure this approach on 8 GB.
- **Supervise memory during occlusion.** [Object Permanence](https://arxiv.org/abs/2103.14258) uses recurrent memory and synthetic/real training with invisible-object supervision. Test visible identity recovery while separating internal hidden-person state from visible output. Establish useful supervision before building a simulator.

Use newer primary literature when it could change the decision; inspect code and deployment assumptions. Future context is allowed for training teachers only, and cannot make an unobservable identity certain for a causal student. Replace these leads when contradicted; preserve tested outcomes.

## Run the work

```powershell
Set-Location D:\Project\tracker
.\lab.ps1 status
.\lab.ps1 doctor
.\lab.ps1 test
```

Use `.venv\Scripts\python.exe`. Working dependencies/pins and upstream patches live in `pyproject.toml`, `requirements-laptop.txt`, `scripts/bootstrap.py`; do not reinstall working GPU packages routinely.

| Work | Command |
|---|---|
| Declared train-only experiment | `.\lab.ps1 diagnose <name>` |
| Declared formal dev experiment | `.\lab.ps1 probe --kind mechanism` (also `reality`, `break-it`) |
| Declared substantial run/resume | `.\lab.ps1 run` |
| Declared post-training evaluation | `.\lab.ps1 eval` |
| Score saved MOT predictions | `.\lab.ps1 score --run runs/<replay> --manifest data/<manifest>.json` |
| Verify/time annotation indexing | `.\lab.ps1 benchmark-data` (CPU, training annotations only) |

Before `diagnose`, set cycle status to `probing` and declare `development.active_experiment`. Source belongs in `experiments/<name>/probe.py`, accepts `--device`, and writes its declared JSON output. Old `runs/_diag_<name>/probe.py` is accepted for migration.

```json
{
  "name": "your-experiment",
  "script": "experiments/your-experiment/probe.py",
  "output": "runs/your-experiment/result.json",
  "status": "planned",
  "question": "Which uncertainty changes the substantial bet?",
  "prior_evidence": ["Relevant evidence path and exact changed assumption"],
  "expected_system_gain": "Why this could move the tracker toward 80%",
  "compute_budget": "Examples, rollout seconds, steps, expected laptop time",
  "decision_rule": "Evidence that justifies scale, promotion or pivot"
}
```

The lab records source hashes and counts completed exploratory results. At three it refuses another small experiment. Declare a substantial `big_run` with a frozen post-run evaluation, or document why a different hypothesis deserves a new cycle. Unit tests and scoring already-saved predictions do not consume scientific probes; they cannot be used to hide additional model runs.

`score` expects `<sequence>.txt` in MOT format and `run.json` containing `sequence`, `frames`, exact `manifest_sha256` and source/checkpoint provenance. It reports official HOTA/IDF1 plus version-2 coverage. All frames, including empty predictions, are evaluated. Prefix results remain labeled as prefixes.

`src/tracker/` owns reusable model/state/data/evaluation/lab code; `experiments/` owns current recipes; `scripts/` owns setup/data/evaluation entry points; `tests/` owns regressions. `evidence/attempts/` stores executable snapshots, context inside JSON and outcomes. `evidence/legacy/` holds historical observations. Data, weights, caches, third-party dependencies and large run artifacts remain ignored/local.

## Laptop efficiency and handoff

Keep heavy work behind `lab.ps1` and one GPU-heavy process at a time. Its activity-aware CPU guard leaves interactive headroom. Cached passing tests are reused only for matching code/dependencies; dynamic input/ledger checks still run each launch. Explicit `lab test` always reruns tests. `status` reads a bounded log tail rather than scanning the feature cache. `doctor` verifies declared mechanics, not scientific validity.

Grouped annotation indexing preserves labels and measured 14x faster preparation on 6,637 training frames; that is not a training-step speed claim. Once a complete trainer exists, spend one bounded engineering check on steady steps/second, data wait and peak VRAM. Keep mixed precision, strict checkpoint loading, positional buffers and resumable checkpoints. Choose batch/unroll by meaningful temporal context and measured memory; use accumulation where appropriate. Log effective batch, updates, distinct frames, rollout seconds and precision mode.

Frozen caches require deterministic preprocessing and frozen eval-mode perception. Disable/re-key them when those assumptions change; do not force every scientific idea into a frozen representation for convenience. Measure cache reuse and transfers before adding workers, pinned tensors or compilation. The [PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) explains why these choices require measurement. Stop harness work when the next meaningful experiment is reliable and easy to run.

Before a handoff: save atomic resumable checkpoints; record the exact next command, evidence, elapsed compute and remaining acceptance gaps; update the current-decision section and relevant decision row. Keep this document compact by rewriting stale instructions while retaining linked results. The lab preserves Markdown context as data inside evidence records, so it creates no duplicate documents. Preserve source/config/results before removing a recipe. Commit coherent decisions and push to `South-33/Tracker` without force-pushing. Never mark the goal complete because a run or budget ended.
