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

1. Read this document and run `.\lab.ps1 status`. Create an active goal from it when asked to run the goal. Preserve enough usage/context to checkpoint and hand off.
2. Restore one runnable reference and score the already-exposed full dev sequences. Git `92da962` contains the earlier complete implementation; inspect and adapt relevant code without resetting this tree. `e6dae0b` preserves the accumulated later research. An external matching tracker is allowed as a reference only. Freeze source/weights/settings and keep the reference runnable while developing its replacement.
3. Bound that repair to getting a real comparison running, aiming for the first hour. Report concrete missing data/dependencies promptly. Do not turn it into a dashboard or generic experiment platform.
4. Choose the substantial bet from the comparison and the evidence below. The paused victim-only anonymous-bundle idea is an option, not an obligation. Temporal perception, a faithfully adapted published learner, teacher-guided training or a justified state reformulation remain open.
5. The next handoff must contain a reproducible reference and a completed or resumable substantial causal experiment with measured learning and exact continuation commands. More small diagnostic files alone do not satisfy this interval.

## Research loop: a few probes, then commit

1. Name the largest measured failure and explain how resolving it could move the whole system toward 80%. Generate a few competing mechanisms, including a simple removal/inversion and a change in state or learning. Use primary literature to expand and challenge the choices.
2. **Check prior evidence before implementing.** Read the relevant row below, search `evidence/legacy/index.json` and saved attempt records by mechanism/synonyms, then inspect the matching result. Record its path, tested conditions and the assumption your proposal changes. If source is missing, reproduce before treating the old conclusion as established.
3. Declare one hypothesis in the ledger: expected system gain, changed assumption, source/inputs, decisive measurement, compute budget and continue/scale/pivot criteria. Preserve exact executable source and input hashes.
4. **At most three exploratory experiments per cycle, including train-only diagnostics and formal dev probes combined.** Use fewer when enough is known. Calling a run a diagnostic, renaming a script or clearing a counter does not create more attempts on the same question.
5. Then launch a substantial experiment or document a different hypothesis. A substantial bet tests the deployable mechanism on complete causal rollouts with its own accumulated errors and meaningful data/context/compute. A decisive full-system intervention without learning also qualifies. Planned training checkpoints are part of that run, not unrelated parameter sweeps.
6. Finish declared compute unless there is a real execution defect, demonstrated impossible input condition or predeclared stop criterion. Watch distinct examples, rollout seconds, learning curves and performance outside fitted examples. Scale a mechanism that is still learning. Ten thousand repetitions of a few examples are not automatically a fair test.
7. Interpret the result, preserve it, update one decision row and commit. State what changed, what failed, what uncertainty remains and why the next action deserves compute. Keep the last runnable reference.

Use small checks to catch broken gradients, invalid labels, leakage or OOM. Do not chase thresholds, width or loss-weight tweaks merely to obtain a positive number. Equally, do not abandon a sound mechanism because its first working version has not already reached 80%.

Label outcomes precisely: **implementation defect**, **negative at tested scale**, **information missing under this protocol**, or **deployment conflict**. A GT-assisted experiment is conditional evidence, not a universal upper bound. A poor teacher readout does not disprove motion, geometry or appearance. Broader rejection needs broader evidence. Revisiting a family is welcome when the changed assumption is explicit; repeating the same failed setup is not.

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
| Raw/pair-conditioned perception / interaction motion | [Pair RT-DETR](evidence/legacy/pair-conditioned-rtdetr-train-diagnostic.json), [raw motion](evidence/legacy/raw-pair-motion-train-diagnostic.json), [interaction motion](evidence/legacy/interaction-motion-train-diagnostic.json): recorded tested-scale negatives. | A specific objective, data or temporal-context change. These runs do not disprove learning motion from pixels. |
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

Use newer primary literature when it can change a decision. Inspect actual training/data/inference code, not just benchmark tables. [MOTIP](https://github.com/MCG-NJU/MOTIP) is a starting point for ID prediction; [MeMOTR](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html) for temporal query memory. Audit runtime association against our contract. Faithful adaptation, larger offline teachers and distillation remain available; avoid turning this document into a literature dump.

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
