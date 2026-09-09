# Continue from here

Updated September 10, 2026. Repository: https://github.com/South-33/Tracker. Local workspace: `D:\Project\tracker`.

**The project is runnable, trained briefly, evaluated, and ready for a substantial next experiment. Robust identity tracking and the Nano target are not solved.** Read GOAL.md for the full research and the user's instructions to reassess, consult primary literature, take meaningful architectural/training steps, and avoid endless small probes.

## What is ready

- Isolated CUDA environment, installed package and pinned upstream code. Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128. `pip check` passes. No SmartCampus dependencies, code or services were modified.
- One recurrent model with 64 resident slots and 300 discovery queries, learned appearance memory and an ID lifecycle without external runtime matching. 10,651,351 parameters.
- Strict pretrained checkpoint loading, a real-image detector preview, verified memory influence and temporal gradients, and 15 passing tests.
- Candidate checkpoint `runs/unroll12/last.pt`: 400 total optimizer steps, first 100 with four-frame windows and next 300 with twelve-frame windows, on the first 120 frames of two training sequences. This is a pilot, not a converged model.
- Uncut replay videos, MOT predictions and official TrackEval metrics. ByteTrack comparison uses the same detector weights with resident queries omitted; original pretrained-detector ByteTrack is also recorded on the first debug clip. NumPy compatibility patches are reproducible through bootstrap.
- Fixed B1 ONNX export, `runs/unroll12/candidate.onnx`. Independent FP32 CPU PyTorch and ONNX Runtime feedback rollouts preserved public IDs over 30 real frames. Maximum visible-box difference was 0.00251 pixels. This does not establish TensorRT/FP16 equivalence.
- All twelve selected DanceTrack prefixes are available: 1,440 images plus annotations, frozen by sequence. Six train, two development and four untouched holdout sequences. No training, download or benchmark job is intentionally left running at handoff.
- A tested disappearance-episode scoring core counts wrong-person reuse before recovery, penalizes delayed switches and retains pre-gap failures. The reviewed real event manifest and association-to-scorer integration remain unfinished.

Small machine-readable evidence is committed under `evidence/`. Models, images, caches, full logs and videos remain local in ignored folders. A fresh Git clone does not contain those large artifacts; bootstrap/data commands reproduce setup, and a same-laptop Sol task can use the existing checkpoint directly.

## Tracking results

Every row below is a 120-frame prefix, not a full-benchmark score. Thresholds were fixed at person 0.4, birth 0.6 and continuity 0.5 for the candidate; ByteTrack uses high 0.5, new 0.6, matching 0.8 and a one-second buffer. The same fine-tuned detector without resident queries is a different operating mode, so detector and association effects are not isolated by this comparison.

| Split / sequence | Model | HOTA | IDF1 | ID switches | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train 0001 | Initial 100-step pilot | 70.18 | 74.33 | 3 | 69.66 | 97.02 |
| Train 0001 | 400-step candidate | 90.22 | 99.40 | 0 | 99.40 | 99.40 |
| Train 0001 | ByteTrack, same fine-tuned detector | 69.99 | 91.46 | 5 | 98.91 | 96.79 |
| Train 0001 | ByteTrack, original pretrained detector | 69.89 | 91.84 | 1 | 94.67 | 93.10 |
| Train 0002 | 400-step candidate | 75.48 | 85.59 | 3 | 88.32 | 99.58 |
| Dev 0016 | 400-step candidate | 60.82 | 62.43 | 5 | 48.86 | 88.43 |
| Dev 0016 | ByteTrack, same fine-tuned detector | 76.46 | 84.04 | 4 | 83.29 | 89.71 |
| Dev 0020 | 400-step candidate | 38.04 | 48.26 | 21 | 78.74 | 45.17 |
| Dev 0020 | ByteTrack, same fine-tuned detector | 34.92 | 47.72 | 39 | 85.57 | 46.88 |

The initial first-clip failure was duplicate resident/newborn tracks on dancers, not merely audience detection. Additional training with longer windows fixed that clip. The second clip still has identity failures. Development 0016 is substantially worse than ByteTrack because of false observations; development 0020 exposes low crowded-scene recall in both methods. There is no overall superiority claim and no held-out result yet. Keep all four holdout sequences untouched until the next candidate and settings are frozen.

Review `runs/unroll12-replay/replay.mp4`, `runs/unroll12-replay-0002/replay.mp4`, `runs/dev-0016/replay.mp4` and `runs/dev-0020/replay.mp4`. Predictions and metrics sit beside each video.

## Laptop speed and memory

This laptop has roughly 15.2 GiB system RAM and an 8 GB RTX 4060. Avoid large RAM-resident datasets, multiple heavy GPU jobs, or an unnecessary GPU cache.

Frozen image features now cache automatically on disk. The cache fingerprints backbone/encoder weights, precision, Torch version, input geometry and preprocessing, with per-frame file metadata and annotation hashes. It is valid only while those modules remain frozen and preprocessing deterministic. Do not use it unchanged with image augmentation or an unfrozen encoder.

A paired 40-step training comparison used the same initial checkpoint, seed and twelve-frame windows. Every logged loss was identical:

| Mode | Total measured time | Steady seconds / optimizer step | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Uncached | 26.719 s | 0.6474 s | 1,281 MiB |
| First cache build, optional 512 MiB GPU working set | 25.031 s | 0.5672 s | 1,847 MiB |
| Warm disk cache, no extra GPU working set (default) | 17.703 s | 0.4318 s | 1,276 MiB |

The default warm path was about **1.50x faster**. This is one short paired pilot, not a broad performance guarantee. The first cache pass costs time and disk space; later passes benefit. A 512 MiB GPU working set had zero hits in this sampling pattern, so it is opt-in rather than the default. `--no-feature-cache` provides the uncached reference. `--cache-mib 512` is available if a different workload demonstrates reuse. Full evidence: `evidence/training-speed.json` and `runs/training-speed.json`.

CPU/PyTorch and ONNX parity, and laptop replay times, are development evidence only. **No Nano benchmark has been run.** The Jetson and production pipeline remain untouched.

## Start the next substantial experiment

Use the existing environment. Do not re-download CUDA, weights or the selected dataset.

```powershell
Set-Location D:\Project\tracker
.\.venv\Scripts\python.exe -m tracker.cli train --manifest data/manifest-120.json --initialize runs/unroll12/last.pt --unroll 16 --steps 2000 --save-every 100 --output runs/expanded
```

This starts a new optimizer on the six-sequence training split while retaining learned weights. The 16-frame setting permits the current random strides on a 120-frame prefix. `--resume runs/expanded/last.pt` continues that same experiment; `--steps` is a total target. Changing dataset, seed, learning rate or debug mode requires a new experiment using `--initialize`, not a misleading resume.

That command is a practical next run, not a reason to spend the entire night training an inadequate objective. Use the results above to choose a substantial improvement: broader temporal/identity supervision and learned newborn separation are the first candidates. Study MOTR/MeMOTR/MOTIP and relevant newer work when the hypothesis calls for it. Extend to full sequences and meaningful disappearance/re-entry events before claiming long memory. Consider a larger architectural change or useful frozen teacher when the evidence warrants it; do not endlessly adjust thresholds.

Evaluate complete development prefixes with the existing replay/evaluation/baseline commands in README. Do not train on development footage. Freeze one candidate, then use the untouched holdout. The pure episode scorer still needs reviewed events and the evaluation association bridge. Export/runtime accuracy and a bounded Nano TensorRT benchmark remain separate required gates.

Full-sequence preparation later:

```powershell
python scripts/fetch_data.py train1.zip val.zip
python scripts/prepare_data.py
```

The partial train archive remains resumable. Do not run a second duplicate download. Existing prefix manifests remain unchanged when full frames are added.

## Artifact identities and important fixes

- Candidate SHA256: `756e8c0fdb5e3fca5b553c0e0fb8f2a7f799977a69f2007f7d9ab6d3eefe977f`. Size 78,185,117 bytes. Its optimizer/RNG/settings are included. Keep it as a reference; use a new output directory for new work.
- Original RT-DETRv4-S SHA256: `238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee`.
- Twelve-prefix manifest SHA256: `519fb90b1fc4b7d824aa19aefdf5479e3ab571418594e8181ff76e7252a14558`.
- Bootstrap owns the upstream decoder feature-output patch, lazy unused-profiler import, TrackEval NumPy fixes and minimal ByteTrack source download. Upstream notices remain with source.
- Register upstream cached position embeddings as nonpersistent buffers after strict loading; otherwise they remain on CPU. Disable denoising after strict loading so the checkpoint's embedding parameter is still accepted.
- Initial AMP loss scale 128 avoided step-zero overflow and passed the training runs. Explicit finite-loss/gradient checks remain active.
- The pilot still omits native FDR auxiliary/local losses, explicit identity contrastive supervision, controlled missing-observation training, learned global re-entry and teacher objectives. Continuity scores are not calibrated identity probabilities.
- `runs/speed-*` and `runs/expanded-smoke` are validation runs, not promoted tracking candidates.

No full tracking goal has been marked achieved. Preserve evidence, make a serious next experiment, and leave enough time and usage for a useful handoff.
