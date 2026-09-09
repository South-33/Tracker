# Remember people across frames

## Goal

Build and evaluate a standalone, causal person detector-tracker. Each call consumes one RGB image, elapsed time, and bounded memory. It returns bounding boxes, anonymous persistent IDs, uncertainty, and updated memory.

The model should keep the correct identity when people walk beside one another, overlap, cross in opposite directions, turn around, become fully hidden, or leave the image and return seconds later. Optimize those behaviors, not the use of a particular architecture or teacher.

Target one camera stream at **at least 15 FPS end-to-end on the Jetson Orin Nano Super 8 GB**. Develop and train on the RTX 4060 laptop. SmartCampus integration and cross-camera identity are outside this goal.

Use one deployed neural network with shared image features and learned temporal association. No full generative LLM, text generation, separate ReID network, ByteTrack, Hungarian association, or Kalman tracker in the candidate runtime. Selected pretrained multimodal layers may remain as a feature encoder if they earn their cost in measured results. External trackers are allowed as baselines. Hungarian assignment is also allowed in supervised training and evaluation. Ordinary image preprocessing, ID allocation, expiry, and output formatting do not need to be neural.

LLM involvement is optional. Useful visual, temporal, spatial, and identity representations may come from a video model, an image model, a multimodal model, or direct supervised learning. Choose from measured tracking performance and deployment cost.

## Instructions for GPT-5.6 Sol

Work in `D:\Project\tracker`. Pursue this as an active goal when this brief is launched. Do not stop at a plan, scaffold, successful export, or attractive demo. Implement, train, inspect failures, and measure the strongest candidate feasible in the available time. Do not set a token budget unless one is supplied.

Use an eight-hour work window as a planning assumption if no duration is supplied. It is not evidence that training will converge in eight hours. Establish a start time and reserve the final 45 minutes for evaluation, recovery, and a useful handoff. A completed overnight feasibility study is different from a completed robust tracker; report those statuses separately.

Use the ordered phases below. Their time allowances guide prioritization, not a reason to abandon a promising experiment. Finish early if the stated deliverables are complete. If time runs out, preserve resumable work and identify exactly what remains unproved. Never mark the full tracking objective achieved because the work window ended.

Make routine implementation decisions autonomously. Continue independent work if a dataset, weight download, export, or device connection is unavailable. Record an actual blocker and its failed command instead of inventing results. Do not buy compute, use paid inference APIs, deploy to SmartCampus, or publish results as part of this run.

## What the research supports

Sources were checked on September 10, 2026. Published results below describe the authors' hardware and protocols. None establishes this project's Nano performance.

| Evidence | Decision it supports | Limit |
| --- | --- | --- |
| MOTR carries track queries across frames and trains them with tracklet-aware assignment. [1][motr] | Learned association within one model is an established starting point. | It is not an off-the-shelf lightweight Nano tracker. The reference training used eight GPUs. |
| MeMOTR keeps longer-term identity memory and injects it into current track features. [2][memotr] | Preserve reliable appearance separately from the latest observation. | Its published memory-optimized training example still uses about 10 GB per GPU. Borrow the principle, not its complete training stack. |
| RT-DETRv4-S reports roughly 10M parameters, 25 GFLOPs and 3.66 ms on T4. Its pretrained detector receives DINOv3 supervision without retaining the teacher at inference. [3][rtpaper] | Start with a compact pretrained detector and add memory. | Those numbers exclude the proposed tracking extension and do not transfer to Nano. |
| FixDT reports a speed increase from 19.6 to 28.8 FPS with fixed-size query memory and attention changes. [4][fixpaper] | Fixed tensor shapes are worth preserving. | The released implementation still needs an export and correctness audit. A fixed-size buffer alone is not an exportable recurrent graph. |
| FastTrackTr has an RT-DETR-based joint network and an AGX Orin deployment experiment. It uses Hungarian association and a Kalman filter after neural predictions. [5][fasttrack] | Useful deployment and speed reference. | It does not satisfy the candidate's no-external-association requirement. AGX Orin is not Orin Nano. |
| FDTA explicitly improves instance discrimination through spatial, temporal, and identity learning. [6][fdta] | Train same-person separation, rather than assuming person detection features contain it. | Extra modules and its training requirements should not be copied without measuring their cost. |
| PermaTrack learns persistence using recurrent memory and supervision for invisible objects. [7][permatrack] | Occlusion-aware supervision can teach useful behavior without an LLM. | Its complete architecture and association behavior are not the required deployment design. |
| TAPNext++ improves point re-detection with longer training sequences and exit/re-entry augmentations; synthetic image-warp training alone hurt general tracking. [8][tapnext] | Train memory beyond adjacent frames and mix controlled augmentation with real motion. | Point tracking is not automatic multi-person detection and tracking. Its 1024-frame training uses distributed machinery unsuitable for this laptop. |
| BeyondSight separates observability from actor persistence. [9][beyondsight] | A hidden actor may remain in memory without a visible detection. | It studies autonomous driving with different inputs and supervision, not this camera-only identity task. |

**Recommended starting candidate:** RT-DETRv4-S with fixed-capacity track queries and a small memory update. Use MOTR's training assignment principle and MeMOTR's separation of recent and reliable historical features. This is an engineering recommendation, not a benchmark-proven winner.

If its measured cost is unsuitable, consider the same tracking design on D-FINE-N, for which official code and weights exist. Recheck person recall after changing the detector. Do not spend the night maintaining multiple complete architecture forks. [10][dfine]

MOTIP is a useful alternative formulation: a network predicts IDs from historical trajectories. Inspect it if persistent queries cannot recover after large displacement, but do not silently introduce an external matcher to make the candidate work. [11][motip]

SAM 3/3.1 also demonstrates neural detection and tracking with memory, including shared memory across objects. It is a reference or optional offline comparison, not an assumed fit for this device. It brings a segmentation stack and a different deployment path. Do not replace the compact-model experiment with a port of that entire stack. [12][sam]

## What “reuse the brain” means

There are two legitimate routes:

1. **Distillation:** a frozen teacher processes training inputs; the compact student learns useful object and temporal relationships. Only the student is deployed.
2. **Direct reuse:** retain a pretrained visual encoder, optionally selected multimodal layers, and train direct tracking heads and memory. Those retained layers remain in the deployed model and must meet the latency budget.

Removing the vocabulary head and autoregressive generation does not compress all the remaining visual computation into a YOLO-sized network. Removing layers can also remove the capability being sought. Measure retained feature quality and compute separately.

LED is relevant to direct reuse: it injects hidden representations from early LLM layers into a detector and retains those layers during inference. It is **not** evidence that its LLM can be removed after training. [13][led]

Research on visual representations inside language models finds useful temporal correspondence information, but also reports cases where the original visual encoder contains more useful visual information. This supports testing both visual and multimodal features rather than assuming the latter are better. [14][visualreps]

Finding “the person in a red shirt” is a useful capability but does not establish persistent identity. The difficult case is two people in similar red shirts crossing, turning, and returning in a different order. A teacher must help distinguish physical individuals in those cases.

A plausible physics prior should help estimate possible motion and uncertainty. It cannot reveal an unobserved identity swap when the available observations are indistinguishable. The correct behavior in such a case is uncertainty. Avoid a model that replaces missing evidence with confident identity guesses.

## Candidate contract

```text
step(rgb_t, dt_seconds, previous_state)
    -> visible_boxes, person_scores, anonymous_ids,
       continuity_confidence, new_state
```

Start with a fixed 640x640 input, batch one, and letterboxing that preserves aspect ratio. Record transforms and map boxes back to source coordinates. Reduced resolution is a later measured tradeoff, not a free optimization.

Use these provisional capacity settings for the first implementation:

- 64 resident track slots, shared by visible and dormant tracks.
- Retain dormant tracks for up to 10 seconds unless memory capacity forces expiry.
- Preserve the detector's 300 discovery queries for the initial correctness/export check. Try 100 only after measuring the detection and tracking impact.
- Report results by actual person count. A capacity of 64 is a resource limit, not a promise to correctly track 64 people.

These are experiment settings, not established optimal values. Freeze them before comparing teachers.

Each resident slot should have a recent query, a reliable appearance summary, a box/reference location, elapsed time since useful evidence, and validity/visibility information. Keep state on the GPU, with fixed tensor shapes. One compact historical summary per slot is the starting point; add a second appearance prototype only if turning or view changes expose a measured limitation.

Use a small update gate so a heavily occluded or ambiguous observation does not overwrite a clean appearance memory. During full invisibility, retain useful memory rather than repeatedly writing background features into it. Train the gate using available visibility/quality supervision and identity consistency. Do not pretend an unsupervised score is a calibrated identity probability.

Separate three concepts:

| Concept | Meaning |
| --- | --- |
| Person confidence | Current pixels support a person observation. |
| Visibility | How much of that person is currently observable. |
| Continuity confidence | Evidence supports assigning this observation to this remembered individual. |

Dormant is a memory lifecycle state. It is not a visibility class or proof of current physical presence. Keep hidden hypotheses out of the normal visible-box output. An optional diagnostic overlay may show them differently, clearly marked as predictions.

Slot indexes are storage addresses. Public IDs must be unique within a run and must not repeat when a slot is recycled. Allocate a fresh monotonically increasing ID for each birth; include a run identifier in saved results. Reset all state at a new video or camera discontinuity. Preserve and account for actual elapsed time when frames are skipped.

Discovery queries must learn to explain new people; resident queries explain their assigned tracks. Supervised matching may assign unmatched ground-truth people to discovery queries, then carry that mapping forward. Do not rematch every frame independently during training and call the result persistent identity learning. Do not use one global identity classification label space across unrelated videos.

A tiny deterministic wrapper may allocate free slots and retire expired ones. It must not compare appearance or geometry to reconnect identities. Returning-person association must come from the network. Learn duplicate suppression through the joint set-prediction training; broad overlap suppression would defeat the central overlapping-person task.

**Re-entry needs an explicit test of spatial reach.** A track query anchored near its last box may not discover a returning person far away. First test the existing joint query interaction. If it fails this case, add one bounded learned path from dormant memory to global/coarse current-frame features or discovery features. Keep it inside the same model. Do not solve the problem with an external nearest-neighbor ReID search.

## Code audit before implementation

The following details were checked in the released code and are easy to get wrong:

| Finding | Required action |
| --- | --- |
| RT-DETRv4-S inherits `num_layers: 3`. | Do not budget a six-to-three-layer speed saving. [15][smallconfig] |
| `_get_decoder_input` selects current-image proposal features; the detector does not accept temporal state. | Explicitly add resident-query inputs and return decoder embeddings. Persisting a sorted list of boxes is not this modification. [16][decoder] |
| The stock postprocessor sorts flattened class scores and gathers boxes. | Preserve slot provenance before sorting. Output row 17 is not track 17. [17][postprocessor] |
| The stock ONNX exporter constructs a batch of 32 and declares dynamic batch axes. | Make a B1 export path with explicit state inputs/outputs. The stock script is inappropriate as-is for this 8 GB experiment. [18][exporter] |
| D-FINE has deliberate within-decoder detach operations. | Do not remove all `detach()` calls. Distinguish box-refinement detach from the temporal feature graph and intentional truncated backpropagation. [16][decoder] |
| FixDT's FSQM uses Python loops, `.item()`, object containers, and recycled IDs; the repository's root license is AGPL-3.0. | Treat it as a reference. Audit tensor/device behavior and ID recycling before reuse. Do not assume that copying it preserves the original proposal's permissive-license rationale. [19][fsqm] [20][fixlicense] |

Verified RT-DETRv4 revision: `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`.

Verified MOTR revision: `8690da3392159635ca37c31975126acf40220724`.

Verified FixDT/MO-YOLO revision: `029e23a776ad916d87f27335f804bdb0064d1466`, branch `feature/decoder-tracker-update`.

Pin the revisions actually used and hash downloaded checkpoints. Preserve upstream notices for any reused code. Use an isolated environment and a small local adaptation, rather than duplicating an entire repository inside another copy. Verify checkpoint loading explicitly; a missing checkpoint must not silently turn into a randomly initialized model.

## Training data

No single dataset covers the requested behavior adequately. Use datasets for distinct jobs.

| Data | Best use here | Important limitation | Priority |
| --- | --- | --- | --- |
| DanceTrack | Similar-looking people, side-by-side movement, nonlinear crossings, maintaining separate tracks during overlap. | Its MOT-format labels use constant trailing values; do not interpret the last column as measured visibility. | First training/evaluation subset. [21][dance] |
| PersonPath22 | Fixed-camera pedestrians, natural disappearances and longer identity histories. | Released annotations are sampled at 5 FPS. Unannotated frames are unknown, not empty. Its “3.5% above 10 seconds” statistic is cumulative occlusion per track, not a distribution of individual 10-second gaps. | Next real-world dataset and coarse-duration test. [22][personpath] [23][personpaper] |
| SportsMOT | Fast direction changes, turning, similar uniforms and a moving camera. | Sports motion and broadcast views differ from fixed pedestrian cameras. | Generalization after the first experiment. [24][sports] |
| CrowdHuman | Recover missed or merged people in crowded still images. | Detection data, not temporal identity supervision. | Add only if detection recall is the bottleneck. [25][crowd] |
| MOTSynth | Synthetic pedestrian diversity and controlled augmentation on real rendered trajectories. | A finite released dataset, not an unlimited scene generator. Inspect which labels exist during full occlusion before using them as hidden-position truth. | Later selective pretraining, not a full overnight download. [26][motsynth] |
| Fresh local footage | Final application-domain generalization. | Needs independent person/occlusion/re-entry truth. Prior SmartCampus acceptance clips must not be silently consumed as training data. | Later untouched holdout, not an overnight prerequisite. |

The official DanceTrack source points to [noahcao/dancetrack on Hugging Face][dancefiles]. The listed archives are approximately 3.61 GB (`train1.zip`), 3.30 GB (`train2.zip`), and 4.21 GB (`val.zip`). Do not download the hidden-label test archives for the first run. Start with `train1.zip` and `val.zip`; extract only the sequences needed initially. Inspect archive contents before extraction and estimate expanded storage.

Use whole source sequences for train/development/holdout separation. Freeze the chosen sequences and event windows before inspecting candidate predictions. A reasonable first allocation is four to six training sequences, two separate development sequences, and four official validation sequences; reduce quantity if necessary, but keep the split honest. Label all subset results as subset results. Do not call them official full-benchmark scores.

Create one manifest containing source URL, revision or file hash, sequence, split, frame timestamps, annotation cadence, class mapping, box convention, and any missing media. Do not interpolate missing identity or visibility labels and present them as observations. Keep separate masks for valid box supervision, known visibility, known identity continuity, and ignored regions.

Use native image geometry consistently. Verify the COCO person-class mapping rather than confusing category ID 1 with a model's contiguous index 0. For initial pretrained parity, retaining the original classification head and selecting the person channel may be safer than replacing the head immediately.

Record dataset and weight terms separately from code licenses. PersonPath22's repository states Apache-2.0 for utility code and CC-BY-NC-4.0 for retrieved videos/annotations. This run is research; do not infer deployment or redistribution rights from the detector's code license. [22][personpath]

## Curriculum and losses

First overfit two short labeled clips. One should contain two nearby people or a crossing; the other should contain a controlled disappearance and return. This is a wiring test, not evidence of generalization. Verify that temporal parameters receive gradients and that the model actually uses previous state.

Then use this progression:

1. **Visible continuity:** adjacent and moderately spaced frames, births and departures, separate identities for overlapping people.
2. **Partial occlusion:** other people and varied foreground occluders; update memory only from useful evidence.
3. **Full disappearance and return:** varied durations and displacement, including returns at another image edge.
4. **Negative returns:** another person appears near the disappearance point; a person never returns; several plausible people return together.
5. **Long streams:** repeated occlusions, accumulated state errors, repeated slot reuse, and crowds that exceed the declared capacity.

Initially use the detector's native classification/box losses, sequence-aware assignment, and a compact identity contrastive objective with same-sequence hard negatives. Add visibility supervision only where known. Do not add eight overlapping loss terms before the model can learn a basic crossing. Retain useful pretrained detection behavior and monitor person recall as tracking improves.

For an 8 GB training GPU, start with batch one, mixed precision, frozen backbone/encoder, and short unrolls. Measure peak memory and step time before scheduling training. Gradually unfreeze only when the measurements support it. Gradient accumulation increases effective batch size; it does not reduce the activations of one unroll.

Use temporal subsampling with true `dt` to expose several seconds within short training clips. This helps cover larger gaps but does not equal full training through every intermediate frame. Later use truncated backpropagation with memory carried across chunks and losses at return points. Document where state is detached. A return loss cannot train memory-writing behavior across a detached boundary unless another objective supplies that learning signal.

Synthetic full occlusion must alter image evidence. Dropping a detector result does not hide a person from an image-to-track network. Use varied foreground occluders and known-identity source tracks, or rendered data with suitable labels. Keep ordinary frames and real motion in the mixture. Do not train only on black rectangles or smooth image warps.

For exit/re-entry augmentation, preserve the source person's identity and elapsed time while changing visibility through a controlled crop or scene boundary. Label it as augmentation, not a naturally observed exit. Test real exit/re-entry separately. Do not fabricate precise hidden boxes when labels do not support them.

Start with gaps spanning 0.25, 0.5, 1, 2, 3, and 5 seconds; add 5-10 seconds after the shorter cases work. Include turns and similar clothing. Changing clothes over hours, multi-camera identity, face recognition, and arbitrary camera cuts are outside this first goal.

## Teacher experiment

Do not spend most of the night training several students before finding whether their teachers offer useful instance information.

Run a small frozen-feature association probe on labeled development clips. Extract person-aligned features, then predict which prior person matches a current observation, including a no-match outcome. Use the same small probe, pair sampling, embedding dimension, data split, and optimization budget for each teacher. Include hard negatives from nearby people with similar clothing. Evaluate after disappearance, turning, and displacement rather than only adjacent-frame retrieval.

| Condition | Question |
| --- | --- |
| Student without additional teacher | What does supervised tracking and its existing pretrained detector provide? |
| V-JEPA 2.1 ViT-B/16, 80M | Do video-trained dense features improve temporal instance discrimination? |
| Qwen3.5-0.8B visual encoder features | How useful is the visual tower before language-layer processing? |
| Qwen3.5-0.8B multimodal features | Do the multimodal layers add useful identity or spatial information beyond that visual tower? |
| DINOv3 image features, if access is already available | Does an additional image-feature objective explain the gain? |

V-JEPA 2.1 releases the 80M ViT-B model as a distilled variant focused on dense, temporally consistent representations. That motivates this experiment; it does not prove person re-identification performance. [27][jepa]

Qwen3.5-0.8B is a vision-language model with a separate visual tower and a hybrid language stack. Its released configuration mixes linear-attention and full-attention layers. Do not transfer LED's two-layer Qwen2 result into a claim about two arbitrary Qwen3.5 layers. [28][qwen] [29][qwenconfig]

Extract features with direct model forward calls, not generated descriptions or chain-of-thought. Inspect the installed model's feature outputs and token-to-image mapping. Do not assume a language generation API returns object-aligned embeddings. Dense features pooled over person regions and person crops answer different questions; compare matching extraction protocols and retain scene context when testing interaction claims.

Use a fixed neutral task prefix if one is needed. Do not give the teacher ground-truth identity names, future outcomes, or the correct match in its prompt. Be aware that visual tokens cannot attend to a text instruction placed after them in a causal language stack. Freeze the layer/prompt choice on development data.

For the initial controlled comparison, each teacher sees only the same past/current frames available up to the prediction time. A full-clip, future-aware teacher is permissible in a separate privileged-training experiment, but that additional information must be reported and controlled. It is not a fair unlabelled improvement over a causal teacher.

Cache only the needed detached features. Include source frame IDs, crop/resize transforms, teacher revision, layer, precision and prompt in cache keys. Teacher features must align with the student's augmented input. Do not reuse a cached full-image feature as if it described a differently cropped or occluded image.

If a teacher looks useful, add one object-aligned distillation objective to the student. Start with normalized feature or pairwise-relation alignment plus the existing identity supervision. Match object correspondence with training labels; raw hidden-state vectors from unrelated models are not directly comparable. Remove the teacher and training-only projection from the exported model and verify that the deployed graph has no teacher dependency.

A weak probe is a screening result, not a proof that all forms of distillation fail. Do not reject the general LLM hypothesis because one layer or one 0.8B checkpoint fails. Conversely, a Qwen win over V-JEPA alone does not isolate “world understanding”: pretraining data, model size, resolution and adaptation differ.

**Direct-reuse challenger:** if Qwen's multimodal features substantially beat its visual features, test a truncated visual/multimodal encoder with direct heads and the same memory contract. Profile the retained subgraph before training it. No vocabulary logits, text generation, or growing KV cache at runtime. Keep this a bounded challenger; do not let its TensorRT port consume the main experiment without evidence of a useful advantage.

## Evaluation

Use TrackEval for HOTA, DetA, AssA, IDF1, ID switches, and fragmentation. Preserve the benchmark's official matching and ignore-region rules. Standard MOT scores and the custom episode test are separate outputs. [30][trackeval]

Required baselines:

1. The same pretrained detector plus ByteTrack.
2. The same detector plus C-TWiX, if its released weights/setup can be made usable within a bounded effort. C-TWiX is an online coordinate-only association model, making it a useful low-cost control. Report its detector source and training data. [31][twix]
3. The identical recurrent student without an additional teacher.

Generate the detector baseline once and reuse those observations across external trackers. The recurrent candidate may legitimately alter detection through memory; therefore report both detection and association metrics instead of attributing every change to identity learning. Add one released end-to-end tracker comparison only if it is straightforward to reproduce; it must not block evaluating the primary candidate.

### Episode tests

Create a frozen list of labeled episodes covering:

- Side-by-side walking without overlap.
- Partial and full overlap while moving in the same direction.
- Opposite-direction crossings, including similar clothing.
- Turning from front to side to back.
- Full occlusion followed by return.
- Exit from the image followed by return, including return far from the old box.
- A different person appearing after the original leaves.
- Multiple simultaneous returns, non-return, and memory-capacity exhaustion.

Score complete uninterrupted clips. Do not reset memory immediately before the hard moment. Inspect all selected cases, not only successful videos.

For each true disappearance episode, define the ground-truth person, pre-gap window, disappearance type, gap duration, and recovery deadline before predictions are opened. Use a provisional 0.5-second recovery deadline and inspect an additional two seconds for a delayed identity switch. Report recovery latency as well as success. Adjust this protocol only on development data.

Use disjoint outcomes with wrong-person reuse taking precedence:

| Outcome | Definition |
| --- | --- |
| Wrong-person reuse | The old public ID is attached to a different ground-truth person within the scoring window, even if it also briefly returns to the correct person. |
| Correct recovery | The same public ID returns to the correct person by the deadline and remains consistent through the scored continuation. |
| False new | The returning person is detected but assigned a fresh ID. |
| Abstention | A person is observed, but continuity remains explicitly unknown. |
| Lost observation | The returning person is not detected by the deadline. |

Keep episodes where the model never established a pre-gap identity in the overall denominator as pre-gap failures. Also report conditional recovery on episodes with a valid pre-gap ID. Otherwise a detector that misses difficult people can look artificially good at recovery.

Use bins `0-0.25`, `0.25-0.5`, `0.5-1`, `1-2`, `2-3`, `3-5`, and `5-10` seconds only when annotation timing supports them. Report occlusion and out-of-frame re-entry separately. With sparse labels, give duration intervals/coarser bins and omit unresolved events from precise-bin claims while reporting their count. A large box overlap is not proof of full occlusion.

Report raw event counts, unconditional and conditional recovery, wrong-link rate among accepted continuity decisions, coverage, false visible boxes during known invisibility, and per-scenario breakdowns. Compare recovery at a matched low wrong-link rate, with operating thresholds selected on development data. Provide uncertainty intervals; bootstrap at source-sequence level where there are enough independent sequences. Zero errors in a small sample does not establish a zero error rate.

Do not invent a universal accuracy threshold before observing baseline difficulty. A useful improvement must preserve detection recall, improve recovery at comparable wrong-link risk, and recur across held-out sequences. Mark small or mixed differences inconclusive. One training seed is a pilot result; run more seeds for a promising result if compute remains.

### Required correctness checks

- Clear/reset state changes behavior on memory-dependent clips; state is actually consumed.
- Repeated slot reuse never recycles a public ID or carries a previous person's appearance into a new birth.
- Empty frames, long gaps, scene resets, and an all-dormant/full memory do not produce invalid tensors or unbounded state.
- Births work while other people are already tracked; overlapping people are not globally suppressed.
- CPU/PyTorch, ONNX where supported, and TensorRT preserve slot mapping and sequence behavior within declared tolerances.
- A multi-frame rollout has no growing autograd graph or accumulated frame/KV buffer during inference.
- Student export/inference works with teacher packages and weights absent.
- Synthetic scoring fixtures verify that a deliberate ID swap is counted as wrong-person reuse rather than successful persistence.

Use focused tests for these failure modes. Do not spend time on tests that merely repeat implementation details.

## Hardware and benchmarking

Read-only inventory during preparation found:

- Laptop: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM; about 196 GiB free on D: at inspection time.
- Jetson: Linux aarch64/tegra, PyTorch 2.8.0, TensorRT 10.3.0, NumPy 1.26.4.

Recheck available memory, free disk, installed versions, and device reachability when starting. These are inventory facts, not benchmark results. Create an isolated laptop environment; do not alter SmartCampus dependencies or upgrade JetPack/CUDA/TensorRT for this prototype.

For Jetson operations use the existing front door:

```powershell
& D:\Project\SmartCampus2\dev.ps1 jetson doctor
& D:\Project\SmartCampus2\dev.ps1 jetson health
```

Before hardware work, read [Jetson operations](D:/Project/SmartCampus2/jetson/jetson.md) and the applicable [agent rules](D:/Project/SmartCampus2/AGENTS.md). Use `dev.ps1 jetson exec` or `script` for isolated research commands. Keep research artifacts in a separate remote directory. Do not run a normal production sync/deploy to transfer this project.

Temporary pipeline stops for compute are already authorized, provided the pipeline is left running afterward. Prepare downloads and laptop exports before stopping services. Use bounded benchmark sessions, device-local timeouts, and cleanup that restores the affected services even if the host connection fails. Record prior service states and verify engine, both cameras, and recorder health after restoration. Do not leave the engine stopped while training on the laptop or waiting for a download. Do not run competing model/camera owners alongside it for a supposedly isolated benchmark.

Follow the existing qualified power/clock policy from the Jetson operations document. The branded “Super” mode is not a measured runtime configuration. Record power mode, clocks, thermals, throttling, and memory with every performance report.

Build TensorRT engines on the Jetson for its installed stack. Do not copy a Windows/4060 engine and assume portability. Serialized TensorRT engines have platform/device/version constraints. [32][trtsupport]

Measure three boundaries:

1. **Neural call:** fixed B1 image/state tensors to network outputs, with proper GPU synchronization.
2. **Frame processing:** an available source RGB frame through resize/normalize, transfer, neural call, memory update, ID allocation, and CPU-visible boxes.
3. **Source loop:** actual video decoding or capture through the same output, including any input wait/queue behavior. Record source resolution and cadence.

The standalone target is sustained at least 15 processed frames/s, with frame-processing p95 at or below 66.7 ms. Report p50/p95/p99 and source-loop throughput separately. Do not silently exclude decoding and call a replay benchmark end-to-end. Keep live input latest-only and bounded; no multi-frame batching or waiting to fill batches. A fresh image and previous state is the recurrent step.

Warm up, then run at least five minutes on representative person footage. Include empty/light scenes and harder person-count/overlap scenes. Carry actual returned state into the next call. A `trtexec` loop feeding the same dummy state is only a compute screen, not a recurrent tracking benchmark. NVIDIA's own benchmarking guidance distinguishes GPU compute, data transfer, and host wall time. [33][trtbench]

Report process and device memory after warm-up, growth over the run, and peak memory. Keep a margin for the OS and memory spikes; report the actual margin rather than declaring that fitting once means it is safe. Synthetic maximum-slot stress measures resource capacity, not tracking accuracy.

Optimize in this order: eliminate unnecessary copies/synchronization, fixed state shapes and B1 buffers, FP16, measured query reduction, then a smaller detector if needed. Re-score accuracy after each change. INT8 is later work and requires representative calibration and sequence-level validation. Do not spend an overnight run debugging quantization before establishing learned identity continuity.

## Execution order

| Phase | Work | Evidence required before moving on |
| --- | --- | --- |
| 1. Inventory and data, roughly 0-1 h | Verify hardware; pin source and checkpoint; create environment; start limited data downloads; freeze sequence splits and evaluation contract. | Exact inputs/versions and a loaded pretrained person detector. No silent random initialization. |
| 2. Deployment screen, roughly 1-2 h | Run B1 detector and a representative fixed-state extension through ONNX/TensorRT. Use a bounded Jetson session, then restore production. | Actual Nano time/memory, or exact export failure and a viable next fix. Untrained recurrent timing is labelled untrained. |
| 3. Learned continuity, roughly 2-5 h | Implement sequence assignment and bounded memory; overfit the two debugging clips; then train on the small split. Run external baselines. | A resumable checkpoint, finite losses, temporal gradients, actual ID continuity, and failure clips. |
| 4. Teacher and hard cases, roughly 5-7 h | Screen V-JEPA and Qwen visual/multimodal features while the student trains if resources permit. Add one useful teacher objective; expand gap and negative-return tests. | Comparable probe results and a matched student experiment, or a precise reason it was not feasible. |
| 5. Final evidence, last 45-60 min | Freeze checkpoint; run held-out scoring and recurrent Jetson benchmark; inspect failure cases; restore services; write results and continuation commands. | Reproducible outputs and a candid statement of what passed, failed, and remains unknown. |

Adapt these allocations to measured training speed. Keep only one GPU-heavy laptop process active at a time unless measured memory allows more. Cached teacher extraction and student training should not compete blindly for 8 GB VRAM.

If the initial detector/export work is slow, still finish the data and evaluator. If the student cannot overfit a two-person crossing, fix assignment/state/gradient flow before adding a teacher. If detection misses one of two overlapping people, address that before blaming memory. If memory is reliable nearby but fails distant re-entry, investigate spatial reach. If the teacher offers no benefit, keep the supervised tracker and report that result.

A new feature, dataset, or loss must answer a specific observed failure. Keep one concise decision table in the result file rather than writing research diaries. Cap retries for unchanged external failures; continue useful independent work.

## Deliverables

Keep the project small. A suitable eventual layout is:

```text
tracker/
  GOAL.md
  README.md          # exact setup, training, evaluation, demo and benchmark commands
  pyproject.toml
  src/tracker/       # model, state, data, training, evaluation and runtime ownership
  tests/
  configs/           # frozen experiment settings, not copies of SmartCampus policy
  third_party/       # only if needed; pinned upstream code and notices
  data/              # ignored downloads/manifests as appropriate
  runs/              # ignored checkpoints, metrics, predictions, logs and videos
  RESULTS.md         # one current evidence report and next commands
```

Do not create all these folders as empty scaffolding. Add them when they own working content. Keep large downloads/checkpoints out of Git. Do not scan unrelated projects or copy private footage into this project merely to avoid public data preparation.

By the end, provide:

- A working `step` API and a video runner, or the closest executable implementation with the missing boundary stated precisely.
- A checkpoint with training settings, source revisions, dataset split hashes, elapsed training time and resume command.
- Standard MOT metrics and custom episode results, including denominators and failure cases.
- At least one uncut held-out demo with visible boxes/IDs and clearly separate dormant-memory diagnostics.
- Raw Jetson timing/memory/thermal results for the actual recurrent runtime, or an explicit “Nano target not yet verified.”
- Teacher probe/distillation results when completed; otherwise the extraction/training command and the concrete remaining dependency.
- Verified restoration of SmartCampus after any hardware stop.
- `RESULTS.md` opening with the actual outcome, followed by a compact comparison table, important failures, decisions, and exact next commands. No invented percentages, “production-ready” claims, or an implication that a synthetic demo proves natural re-entry tracking.

Useful result columns are: candidate/checkpoint, source split, person recall, HOTA, AssA, IDF1, recovery numerator/denominator, wrong-person events, false-new events, abstentions, latency, Nano FPS, peak memory, and evidence path. Use `not measured` rather than filling absent results with estimates.

## Primary sources

1. Zeng et al., **MOTR: End-to-End Multiple-Object Tracking with Transformer**, ECCV 2022. [Official code and training notes][motr].
2. Gao and Wang, **MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking**, ICCV 2023. [Paper][memotrpaper]; [official code and memory requirements][memotr].
3. **RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models**, 2025 preprint / ECCV 2026. [Paper][rtpaper]; [official source][rtrepo]; [official S checkpoint link][rtweights].
4. Liao et al., **DecoderTracker: Decoder-only end-to-end method for multiple-object tracking**, Pattern Recognition 177, 2026. [Publisher paper][fixpaper].
5. Liao et al., **FastTrackTr: Real-time Multi-Object Tracking with Transformers for Real World**, July 2025 paper revision / IEEE TII 2026. [Paper, especially association and deployment sections][fasttrack].
6. Shao et al., **From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking**, CVPR 2026. [Paper][fdtapaper]; [official code][fdta].
7. Tokmakov et al., **Learning to Track with Object Permanence**, ICCV 2021. [Official PermaTrack repository][permatrack].
8. Jung et al., **TAPNext++: What's Next for Tracking Any Point?**, April 2026 preprint. [Paper][tapnext].
9. Papais et al., **BeyondSight: Object Permanence for End-to-End Autonomous Driving**, July 2026 preprint. [Paper][beyondsight].
10. **D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement**, ICLR 2025. [Official code and N/S weights][dfine].
11. Gao et al., **Multiple Object Tracking as ID Prediction**, CVPR 2025. [Official MOTIP repository][motip].
12. Meta, **SAM 3 / SAM 3.1 Object Multiplex**, official repository and March 2026 release note. [Code and model links][sam].
13. Zhou et al., **LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation**, CVPR Findings 2026. [Paper, including retained LLM inference cost][led].
14. Liu et al., **Visual Representations inside the Language Model**, October 2025 preprint. [Paper][visualreps].
15. RT-DETRv4 authors, **Small-model inherited configuration**, pinned source. [Configuration][smallconfig].
16. RT-DETRv4 authors, **D-FINE decoder**, pinned source. [Decoder inputs, outputs and detach behavior][decoder].
17. RT-DETRv4 authors, **PostProcessor**, pinned source. [Sorting and box gathering][postprocessor].
18. RT-DETRv4 authors, **ONNX export script**, pinned source. [Exporter][exporter].
19. DecoderTracker authors, **FSQM implementation**, pinned source. [Memory lifecycle code][fsqm].
20. DecoderTracker authors, **MO-YOLO root license**, pinned source. [License][fixlicense].
21. Sun et al., **DanceTrack**, CVPR 2022. [Official dataset repository][dance]; [author-linked archive distribution][dancefiles].
22. Amazon Science, **PersonPath22 release and annotation documentation**. [Repository, including cadence and terms][personpath]; [dataset page][personpage].
23. Shuai et al., **Large Scale Real-world Multi-Person Tracking**, ECCV 2022. [Paper, page 9 occlusion definition][personpaper].
24. Cui et al., **SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes**, ICCV 2023. [Official repository][sports].
25. Shao et al., **CrowdHuman: A Benchmark for Detecting Human in a Crowd**, 2018. [Official dataset][crowd].
26. Fabbri et al., **MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?**, ICCV 2021. [Official repository][motsynth]; [download/preparation instructions][synthdata].
27. Mur-Labadia et al., **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**, 2026. [Paper][jepapaper]; [official checkpoints][jepa].
28. Qwen Team, **Qwen3.5-0.8B**, 2026. [Official model card][qwen].
29. Qwen Team, **Qwen3.5-0.8B architecture configuration**. [Configuration][qwenconfig].
30. Luiten et al., **TrackEval / HOTA**, official implementation. [Repository][trackeval].
31. Miah et al., **Learning Data Association for Multi-Object Tracking using Only Coordinates**, Pattern Recognition 2025. [Official C-TWiX code and weights instructions][twix].
32. NVIDIA, **TensorRT 10.3 Support Matrix**. [Platform and engine compatibility][trtsupport].
33. NVIDIA, **TensorRT Performance Benchmarking**. [Measurement definitions][trtbench].

[motr]: https://github.com/megvii-research/MOTR/tree/8690da3392159635ca37c31975126acf40220724
[memotr]: https://github.com/MCG-NJU/MeMOTR
[memotrpaper]: https://arxiv.org/html/2307.15700v3
[rtpaper]: https://arxiv.org/html/2510.25257v1
[rtrepo]: https://github.com/RT-DETRs/RT-DETRv4/tree/55fefaaed7efe2a5f72d0a18fd4e05965e35c292
[rtweights]: https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z
[fixpaper]: https://www.sciencedirect.com/science/article/abs/pii/S0031320326002074
[fasttrack]: https://arxiv.org/html/2411.15811v4
[fdta]: https://github.com/Spongebobbbbbbbb/FDTA
[fdtapaper]: https://arxiv.org/abs/2512.02392
[permatrack]: https://github.com/TRI-ML/permatrack
[tapnext]: https://arxiv.org/html/2604.10582v1
[beyondsight]: https://arxiv.org/html/2607.09138v1
[dfine]: https://github.com/Peterande/D-FINE
[motip]: https://github.com/MCG-NJU/MOTIP
[sam]: https://github.com/facebookresearch/sam3
[led]: https://arxiv.org/html/2503.13794v2
[visualreps]: https://arxiv.org/html/2510.04819v1
[smallconfig]: https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/configs/dfine/dfine_hgnetv2_s_coco.yml
[decoder]: https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/dfine_decoder.py
[postprocessor]: https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/postprocessor.py
[exporter]: https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/tools/deployment/export_onnx.py
[fsqm]: https://github.com/liaopan-lp/MO-YOLO/blob/029e23a776ad916d87f27335f804bdb0064d1466/MOTR/models/fsqm.py
[fixlicense]: https://github.com/liaopan-lp/MO-YOLO/blob/029e23a776ad916d87f27335f804bdb0064d1466/LICENSE
[dance]: https://github.com/DanceTrack/DanceTrack
[dancefiles]: https://huggingface.co/datasets/noahcao/dancetrack/tree/main
[personpath]: https://github.com/amazon-science/tracking-dataset
[personpage]: https://amazon-science.github.io/tracking-dataset/personpath22.html
[personpaper]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680493.pdf
[sports]: https://github.com/MCG-NJU/SportsMOT
[crowd]: https://www.crowdhuman.org/
[motsynth]: https://github.com/dvl-tum/motsynth-baselines
[synthdata]: https://github.com/dvl-tum/motsynth-baselines/blob/main/docs/DATA_PREPARATION.md
[jepa]: https://github.com/facebookresearch/vjepa2
[jepapaper]: https://arxiv.org/html/2603.14482v2
[qwen]: https://huggingface.co/Qwen/Qwen3.5-0.8B
[qwenconfig]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
[trackeval]: https://github.com/JonathonLuiten/TrackEval
[twix]: https://github.com/Guepardow/TWiX
[trtsupport]: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Support-Matrix-Guide.pdf
[trtbench]: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html
