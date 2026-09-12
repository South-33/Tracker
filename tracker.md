# Tracker

<!-- cycle-status: probing -->

Updated 2026-09-12. Workspace: `D:\Project\tracker`.

This file is compact scientific state. Git is the archive. Delete stale claims, dead detail, duplicated guidance, and old next-actions when they stop improving the next decision.

## Goal and hard contract

Build a standalone causal multi-person DanceTrack tracker for Jetson Orin Nano Super 8 GB:

```text
step(rgb_t, dt_seconds, previous_state)
    -> visible_boxes, person_scores, anonymous_ids,
       continuity_confidence, new_state
```

Target the **80-90% strict switch-free visible-segment regime with useful recall**, then freeze and verify full uncut dev HOTA/IDF1/FP/miss/IDSW, reviewed re-entry, four untouched final holdouts, one uncut heldout demo, ONNX parity, TensorRT FP16 parity, and real Nano FPS/memory/thermals at >=15 FPS.

Runtime stays one neural model plus bounded bookkeeping. No deployed Hungarian, ByteTrack, Kalman, full-history optimization, or second heavyweight network. Public IDs normally never recycle. Suspect spatial evidence must not overwrite protected identity authority. Publish only current observations, never stale coast boxes.

`dancetrack0096` is an architecture gate, never training data. Four final holdouts remain untouched. Clean RT-DETR ancestry is `runs/expanded-0002/last.pt`; the clean long manifest is `data/manifest-train-full-dense-no0096.json`.

## Research operating system

Treat the repo as part of the model context. **Context poison is a bug.** `AGENTS.md`, this file, and `research-cycle.json` are the active research-guidance surfaces. If prose and the machine ledger disagree, repair that before research.

Each architecture cycle gets one jump claim and at most three optional **formal probes**: mechanism, reality, break-it. Probe slots protect scarce non-gate dev evidence from repeated peeking; they are not a tax on learning. Train-only no0096 `diagnose` work may train real mechanisms freely, but keep one declared active development question at a time and stop a line after a fair decisive negative instead of sweeping variants. A big run is earned, never mandatory. No Probe 4.

Do not hard-close learned ideas from under-trained runs. The K-worldline family learned late but still failed its fair full-system evaluation, so distinguish "not learned yet" from "learned mechanism is wrong."

Small threshold/width/loss/head-count changes are not scaling research. Think from the measured failure first. Generate a simple/weird mechanism, a removal/inversion, and a state/learning reformulation before using literature. Literature is for falsification and implementation help, not for choosing the search space.

## Bird's-eye state

Three facts dominate everything else:

1. **Do not use the old 73-74% detector-first number as an active baseline.** It has no surviving result artifact in the current repo, and a direct replay of the full `expanded-0002` resident-query checkpoint on the current non-gate dev protocol produced only **37.18% strict / 62.69% precision / 59.82% recall**, with 2,695 switches and 12,595 public-ID reuses. Treat the older number as stale context until independently reproduced.
2. **Detection availability is excellent.** In the completed uncut K-worldline audit, the true owner was present in the current top-96 clean RT-DETR proposals on **32,365/32,603 = 99.27%** of visible rows. The bottleneck sits between available evidence and identity-state commitment.
3. **Rare bad writes compound catastrophically, but safe local tracking has enough ceiling.** The earlier 0096 optimistic oracle reached **89.80% strict / 99.91% precision / 63.72% recall**. The newer non-gate local-quarantine ceiling reaches **100.00% strict / 100.00% precision / 68.80% recall** with zero switches/reuses while taking ordinary reciprocal local links by default. The remaining learned problem is hazard detection, not identity association.

### What failed at system scale

**K=3 causal worldlines:** 800-step training genuinely learned late, but uncut non-gate dev with oracle identity establishment was disastrous: owner candidate availability **99.27%**, K-set owner retention **32.59%**, top-1/recall **15.32%**, precision **20.07%**, strict survival **16.24%**. Recurrent identity transport destroyed information already present in the current frame. It is superseded; do not tune K/GRU/rollout length.

**Protected set-ID prediction:** 5,000 steps also showed late learning, but the fully causal uncut dev tracker became ultra-conservative. Combined result: **99.51% precision, 15.56% recall, 60.26% strict survival, 77 ID switches, 4,275 public-ID reuses, only 4 births across 46 GT people**. Replacing ordinary tracking wholesale with global identity classification was the wrong systems move. Do not rescue it with NEW thresholds or head-size tweaks.

Pure abstention already failed for the same structural reason: an older conservative authority reached **75.51% strict / 99.43% precision** at only **2.22% recall**.

## Prior whole-system negatives

Several mechanisms had good oracle ceilings but failed once their own state errors fed back. The protected binder learned exact association at the expense of survival; SAFE/scout probation reached **81.20% strict / 97.87% precision / 6.17% recall**; learned local quarantine reached **91.45% / 97.00% / 8.77%** and made false vetoes absorbing; one private shadow and a two-branch hedge had hard recall ceilings of **12.19%** and **24.40%**. Do not revive these with thresholds, larger K, or retry rules.

Immutable opening-canonical recovery remains the strongest useful system ceiling: **97.86% strict / 99.95% precision / 59.24% recall**. Only **183** rare oracle-safe reanchors unlocked **19,249** ordinary local publications. The learned 3,000-step version still failed at **53.85% strict / 97.46% precision / 36.23% recall**, showing that the first wrong write, not recovery capacity, is the dominant systems problem.

Recent trusted appearance helped clean hazard ranking but not enough for system safety: its fixed reciprocal rule accepted at only **97.0%** precision and its whole-system ceiling was **97.86% strict / 99.91% precision / 54.39% recall**. Constant velocity, sparse LK flow, private shadows/forks, and aged retry/re-entry are not the jump.

## Other hard negatives

A permanent first-hazard fuse had a decisive ceiling of **100% strict / 100% precision / 4.19% recall**; transient hazards cannot mean permanent retirement. Adjacent-frame signatures and center/shape geometry did not beat the local IoU selector safely enough. Sparse LK flow adds only 26 net safe choices over last-box geometry across **59,290** clean transitions. These branches are closed to tuning. The surviving systems lesson is the **state-distribution problem**: after the first wrong write, otherwise useful local safety features become nearly useless.

## Closed state-side shortcuts

Frozen trusted boxes look deceptively strong on random clean gaps, but the actual aged tail is much harder. The tested fresh-reciprocal / aged-track-perspective rule reached only **98.29% strict / 99.89% precision / 32.34% recall**; 3,753 aged fallback candidates contained only 116 oracle-safe recoveries. Do not revive stale-track nomination, retry thresholds, private shadows, or motion-state variants. The useful surviving ceiling is still immutable opening-canonical recovery at **97.86% strict / 99.95% precision / 59.24% recall**, where only 183 rare safe reanchors unlock 19,249 ordinary publications.

Opening-side fixes are not the jump. Exact-two-second enrollment is still imperfect on crowded `0020`, but train-only oracle dedupe of all mature private tracklets improves represented opening owners only **93/101 -> 94/101**; a coexistence-aware rule rescues that one owner while duplicate slots jump **5 -> 40**. Three immutable opening exemplars and even the full opening feature bank move recovery top-1 only **12.75% -> 13.43%** across train and **9.22% -> 9.75%** on crowded `0012`. Evidence: `runs/opening-census-train-diagnostic.json` and `runs/opening-exemplar-train-diagnostic.json`.

The first-write diagnosis is now sharper. On oracle-clean `0012`, only **312/5,280** unsafe local links are fresh; after a veto/miss there are **4,968** aged hazards. At the fresh boundary the owner is still visible **93.59%** of the time. With deployable score filtering, the owner proposal is available in **258/309** fresh hazards; when available it lies in the track's top-3 IoU candidates **77.52%**, top-5 **93.41%**, and top-8 **99.61%**. Simple tie-breakers fail: top-3 detector score resolves **40.70%**, canonical appearance **39.53%**, and recent appearance **21.32%**. Evidence: `runs/hazard-onset-train-diagnostic.json` and `runs/fresh-candidate-rank-train-diagnostic.json`.

Short private state does not make that candidate cloud easy. A one-frame top-8 beam contains an owner-preserving path for **79.94%** of fresh hazards, but fixed geometric path rules resolve only **24.92-38.51%**. Spatial encoder RoI templates are also flat with IoU: template selection is **94.77%** correct over fresh links versus **94.96%** for the existing local winner; an agreement rule vetoes only **20.83%** of hazards while dropping about **6.4%** of safe links. Pairwise swap repair is not the answer either: only **3.2%** of `0012` hazard links form true two-cycles. Evidence: `runs/fresh-short-beam-train-diagnostic.json`, `runs/spatial-template-train-diagnostic.json`, and `runs/pairwise-crossover-train-diagnostic.json`.

Frozen track-query shortcuts also fail hard. Reusing the current RT-DETR decoder with immutable canonical content, recent trusted content, or its pretrained person denoising embedding plus the trusted box gives oracle-safe recall only **9.50% / 3.77% / 1.13%** on `0012`. A tracking-by-query family would therefore require real temporal association learning; it is not a free reuse of the detector decoder. Evidence: `runs/protected-track-query-train-diagnostic.json`.

Survival-conditioned safety learning confirms the state-distribution problem but is still insufficient. A same-size 16-D hazard MLP trained only on oracle-clean prefixes, with `0012` held out and a fixed zero-logit rule, accepts **77.93%** of safe links and vetoes **97.77%** of clean hazards, yet causal replay reaches only **77.69% strict / 99.19% precision / 15.97% recall** and corrupts by frame **47**. A fresh-link-only version is much cleaner on held-out `0012`: **99.40% accepted precision / 53.83% safe acceptance / 93.91% hazard veto**. Adding a learned top-8 proposal/track context set makes generalization worse (**97.50% / 82.92% / 59.94%**). One-frame retry does not repair the state problem: raw reciprocal retry after false veto is only **90.13%** precise; the same guard recovers **7.20%**, and a separately trained clean-stale retry guard is only **88.37%** precise while recovering **11.58%**. Stop retry/age-model variants. Evidence: `runs/clean-prefix-guard-train-diagnostic.json`, `runs/fresh-context-train-diagnostic.json`, `runs/quarantine-retry-train-diagnostic.json`, and `runs/retry-guard-train-diagnostic.json`.

Recent train-only dead ends also include waiting 1-6 frames after a hard event, detector query-index continuity (~54% owner purity), lower opening thresholds, and one-frame unpublished geometric bridges. Decoder-only identity adaptation is also negative at tested scale. Generic adjacent-ID training improved held-out recent-feature selection **83.69% -> 87.94%** but stayed below geometry **89.74%**; training only on **1,210** real top-8 geometry mistakes made it worse at **81.25%**, with hard-alternate recovery only **19.13%**. Stop loss/temperature variants on the same single-frame decoder representation. Evidence: `runs/identity-decoder-train-diagnostic.json` and `runs/identity-hard-train-diagnostic.json`.
A full 10,000-step 30-frame causal self-attention ranker over raw trusted detector features also failed the jump test on held-out `0012`: **33.86%** recovery on IoU-hard cases, **67.85%** accuracy on low-margin safe cases, and **54.13%** overall ambiguous-set accuracy despite near-perfect train batches. Frozen-output temporal attention is negative at fair scale; do not spend on larger post-hoc history models. Evidence: `runs/temporal-transformer-train-diagnostic.json`.

## Killed frozen-temporal cycles

Visible-only protected decoder track queries showed real hard-crossing signal but failed system safety: heldout development hit **53.44%** hard recovery / **92.01%** safe accuracy, while frozen non-gate dev fell to **45.41% / 82.08%**. Forced-veto retries then exposed the semantic failure: by +5 first-accept precision was **80.37%** and absent owners were falsely certified **73.65%** of the time. Inspection of the training source shows why: every protected query was trained only on owners visible in both adjacent frames, and its person logit was always trained positive. That experiment never taught an explicit absent/NULL state.

The inverted 1.13M-parameter backward temporal linker also failed after a full 10,000-step train-only run. On heldout `0012` at gap 1 it reached **97.82% accepted precision / 84.90% safe acceptance / 63.19% hazard veto / 19.68% hard-owner recovery**, with **28.40% absent-owner false claims**. Gaps 2-5 degraded further even though training loss kept improving. This is strong evidence that temporal association on top of the frozen single-frame RT-DETR representation is not the jump. Artifacts: `runs/protected-track-query-perception-*.json` and `runs/backward-temporal-linker-train-diagnostic.json`.

## Killed lightweight raw-motion branch

A full 10k raw pair-motion CNN reached only **20.22%** hard recovery at **97.00%** accepted precision and falsely claimed **51.85%** of absent protected owners. Direct pixel motion is learnable but not identity-safe here. Evidence: `runs/raw-pair-motion-train-diagnostic.json`.

## Other perception negatives

Pair-conditioned predecessor/contrast/NULL models, dense frozen encoder transport, SAM2.1 masks, and YOLO11x pose all failed at useful identity-safe recovery despite strong detector coverage or strong auxiliary perception. Best notable numbers were **46.67%** hard recovery for identity contrast at **98.44%** precision, **14.45%** for pointwise dense transport, **34.38%** for SAM2.1 with poor safe preservation, and **10.42%** for pose. Do not revisit these with head/loss/readout tuning. Evidence: `runs/pair-conditioned-*.json`, `runs/protected-query-null-train-diagnostic.json`, `runs/dense-field-transport-train-diagnostic.json`, `runs/mask-teacher-train-diagnostic.json`, `runs/pose-teacher-train-diagnostic.json`.

## Structural conflict quarantine: negative

The zero-threshold overlap rule contains **593/595 = 99.66%** of hazardous reciprocal writes, but DanceTrack geometry is too crowded for it to be selective: **94.98%** of all reciprocal links and **94.74%** of safe links fall into the conflict set. Only **5.26%** of safe links are structurally certified, at **99.67%** precision. Do not tune an overlap cutoff; the threshold-free fast path does not exist. Evidence: `runs/structural-conflict-train-diagnostic.json`.

## Confidence-weighted private anchor: negative

Without any recovery oracle, hard veto/freeze is extremely safe but dead: **98.35% strict / 100% precision / 0.46% recall**. Letting each vetoed link move only the private spatial anchor by the guard-derived gain `sigmoid(-hazard_logit)` (mean gain **0.208**) restores recall to **43.22%**, but identity corruption returns immediately: **31.40% strict / 96.72% precision**, with 264 switches and 5,097 public-ID reuses. This closes the simple continuum between frozen and partially/full suspect-box assimilation. Do not tune gain or momentum constants. Evidence: `runs/confidence-weighted-anchor-train-diagnostic.json`.

## Interaction-aware motion: negative at fair scale

A 27k-parameter six-frame, eight-neighbor geometry forecaster did learn more than constant velocity, but not a finish-line jump. After 10k train-only steps, heldout `0012` hard recovery was **35.06%** versus **16.67%** for constant velocity, while safe preservation fell to **94.79%** (best early checkpoint: 35.06% hard / 97.67% safe; peak hard 37.36%). More fitting reduced train loss without improving heldout association. Stop small geometry-only history/neighbor/width variants. Evidence: `runs/interaction-motion-train-diagnostic.json`.

## Separate tracking pointer decoder: negative at fair scale

A separate 1.33M tracking decoder still failed owner-or-NULL semantics after 10k steps: heldout `0012` was **18.95% hard recovery / 80.33% safe acceptance / 79.91% non-NULL precision / 76.54% absent false claims**. A zero-parameter mutual-best track/proposal rule improved precision to **91.31%** and absent false claims to **35.80%**, but hard recovery fell to **15.38%**. Stop pointer/layer/NULL variants and do not train a larger association matrix on the same evidence. Evidence: `runs/tracking-pointer-train-diagnostic.json`, `runs/pointer-mutual-train-diagnostic.json`.

## Active cycle: joint permutation bundle

Strong point correspondence is not the missing cue. Official scaled-online CoTracker3 with a proper 16-frame causal window, 4x4 points seeded inside the previous oracle-clean owner box, no future frames, and no backward tracking recovered only **21/96 = 21.88%** hard links while preserving **87/96 = 90.63%** safe controls. It had zero missing-visible-point and zero unsupported-proposal cases, so do not tune or distill this family. Evidence: `runs/point-tracking-teacher-train-diagnostic.json`.

The failure shape instead points to a state reformulation. On heldout-train `0012`, **447/533 = 83.86%** of hard transitions choose another real dancer's proposal; only 86 choose background or an unowned proposal. The dominant failure is therefore an identity permutation during interaction, not detector localization.

An oracle-membership joint bundle produced the first large architecture jump in several cycles. For two-person conflict episodes, suppress publication for 8 frames and keep at most eight coupled one-to-one constant-velocity hypotheses. GT supplies only the correct two-person proposal set, with identities hidden from the bank. After deduplicating repeated hard frames, 140 episodes had both people continuously detected for the full window, and the outgoing permutation was correct on **93/140 = 66.43%**. Evidence: `runs/joint-bundle-oracle-train-diagnostic.json`.

Observable first-frame bundle triggers are not sufficient by themselves. "Victim favorite is closer to another trusted track" has about **0.99%** safe false trigger and about **98%** partner correctness when it fires, but covers only about **35%** of person-swap hazards. Two tracks choosing the same proposal has about **1.52%** safe trigger but only **27%** hard person-swap coverage. Quarantining the lowest bidirectional-IoU-margin reciprocal link captures only **129/595 = 21.68%** hazards while suppressing **1,073** safe links, about **9.28%** of safe writes.

The first causal expanding-component attempt also fails decisively. Connecting protected tracks by shared top-2 current proposals captures **87/140 = 62.14%** oracle-valid hard episodes, but suppresses **26.87%** of otherwise-safe rows. Its anonymous expanding bank recovers only **24/140 = 17.14%** targets overall, **27.59%** conditional on seeding, preserving only **25.81%** of the 66.43% oracle-bundle benefit. Do not tune rank, pool size, horizon, K, or motion weights on this trigger. Evidence: `runs/expanding-bundle-train-diagnostic.json`.

The simpler next question is whether the **existing strong first-hazard signal can be used only as a bundle-entry alarm instead of a freeze decision**. The fresh-link guard already showed **99.40% accepted precision / 53.83% safe acceptance / 93.91% hazard veto** on heldout `0012`; its causal failure came from false vetoes freezing spatial state and becoming absorbing. Diagnose the state semantics first: oracle-seed only the victim ID at a hard event, build an anonymous proposal set around that victim, let other protected IDs join only after they intersect that anonymous set, and ask whether joint motion can (a) recover hard exits and (b) unwind deliberately injected false alarms on safe links. If this is strong, plug in the frozen guard as the real entry alarm. If even oracle victim-only entry fails, kill the bundle family rather than training another association head.


## Decisive supporting evidence

Clean first-corruption diagnostics remain useful: full-dev owner-present cases give protected box **27/33**, protected box/feature cue oracle **28/33**, raw protected box **35/43**. On 0096 historical diagnostics gave protected last box **25/31** and protected cue oracle **28/31** when owner-present. Raw proposal clouds were worse, so candidate-set discipline matters.

PRESENT/ABSENT observability is weakest at first disappearance: visible-vs-all-absent best-IoU AUROC **0.923**, but visible-vs-first-absent only **0.769**; retaining 99% of visible rows rejected only 8.5% of first absences. A good gate therefore needs switcher/context evidence, not just a visibility threshold.

A train-only zero-parameter sanity check tried to remove learning entirely and commit from scout agreement/reciprocity. It failed the system tradeoff: unanimous voting reached **80.51% strict but only 5.46% recall**; majority voting reached **54.79% recall but only 32.05% strict**; a reciprocal-dual rule reached **85.38% strict but only 16.87% recall**. This was hypothesis generation on cached training proposals, not a dev probe. Raw voting is not the replacement; the new cycle preserves ordinary continuity and quarantines only rare ambiguity.

Evidence labels: **hard-closed** only for a measured information/deployment contradiction; otherwise say **negative at tested scale**, **superseded**, or **scaling-open**. Do not carry a graveyard of old architectures in active context.

## Active repo map

- `src/tracker/ancestry.py`, `detector.py`, `features.py`, `data.py`, `geometry.py`: clean detector/data infrastructure.
- `src/tracker/state_split.py`: allocate-once protected public identity authority.
- `src/tracker/enrollment.py`: exact bounded private two-second opening enrollment; births close after roster freeze.
- `src/tracker/local_continuity.py`: reciprocal local proposal generation, trusted state, and current-frame safety features.
- `src/tracker/detections.py`, `proposals.py`: deterministic frozen proposal cache/stream for long rollouts.
- `src/tracker/lab.py`, `research_cycle.py`, `research-cycle.json`: research state machine and guardrails.
- `tracking_metrics.py`: causal diagnostic metrics; any Hungarian inside it is evaluation-only.

Failed SAFE/binder/local-recovery implementation is archive-only and should not remain in the active code surface.

## Acceptance path after a promotion

Freeze model and operating rule; run full uncut dev HOTA/IDF1/FP/miss/IDSW and reviewed re-entry; only then use final holdouts; produce one uncut heldout demo; verify ONNX and TensorRT FP16 parity; finally measure real Orin Nano Super FPS, memory, and thermals. The project is not complete before all are measured.

## Commands

```powershell
Set-Location D:\Project\tracker
.\lab.ps1 status
.\lab.ps1 doctor
.\lab.ps1 test
.\lab.ps1 diagnose <name>
.\lab.ps1 run
```

Use `.venv\Scripts\python.exe`, one GPU-heavy process at a time. Do not bypass the frozen lab workflow or silently introduce new flags.




