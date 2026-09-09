# Handoff to GPT-5.6 Sol

September 10, 2026. Work in `D:\Project\tracker`. Read this first, then README.md and GOAL.md. Continue from the existing checkpoint. This is an executable pilot, not a validated tracker.

## Verified

- Separate CUDA environment works: Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, RTX 4060. Editable package installed; pip check passes. Installed versions are in requirements-laptop.txt.
- Official detector checkpoint loads strictly. weights/rtv4_s.pth is 169,278,400 bytes; SHA256 238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee.
- Candidate: 10,651,351 parameters, 64 resident slots, 300 discovery queries. Empty memory exactly preserves pretrained discovery boxes and person logits. Previous-frame feature gradient L1 was 3.1012. Evidence: runs/verify.json.
- Eight tests pass in the new environment, covering identity assignment through crossings and returns, memory clearing, expiry, geometry, false observations, public-ID reuse and capacity.
- GPU training completed 100 optimizer steps on four-frame windows from two training sequences. Resume from step 2 worked. Loss was 3.3626 at step 1 and 0.7933 at step 100 on different windows. This does not establish convergence or generalization.
- Checkpoint: runs/smoke/last.pt. Settings and losses: runs/smoke/config.json and loss.jsonl.
- Uninterrupted 120-frame training-prefix replay completed: runs/smoke-replay/replay.mp4, MOT rows, predictions.jsonl and run.json. It allocated 14 IDs. Observations increased from 9 at frame 30 to 14 at frame 120; inspect duplicate births and identity consistency.
- Replay measured about 26 source-loop FPS and 27.4 ms frame-processing p95 on the laptop. These are short laptop development numbers, not Nano results.

## Next actions

1. Fix TrackEval's removed NumPy aliases through a reproducible bootstrap patch. Current failure: AttributeError for np.float at third_party/TrackEval/trackeval/datasets/mot_challenge_2d_box.py:228. Check np.int too. Then run:

   `.\.venv\Scripts\python.exe scripts/evaluate.py --manifest data/manifest-debug.json --run runs/smoke-replay`

   No HOTA/IDF1 result exists yet. Debug-training metrics are not held-out evidence.

2. Inspect the complete replay before expanding training. Resume:

   `.\.venv\Scripts\python.exe -m tracker.cli train --debug --manifest data/manifest-debug.json --steps 500 --unroll 4 --save-every 10 --output runs/smoke --resume runs/smoke/last.pt`

3. Complete the frozen data split. Only dancetrack0001 and dancetrack0002 have guaranteed complete 120-frame prefixes in data/manifest-debug.json; a third sequence is partially extracted. Downloads were stopped to prioritize environment setup. No training or download process is intentionally left running. Resume one download path at a time:

   `python scripts/prepare_data.py --remote --frames 120`

   `python scripts/fetch_data.py train1.zip val.zip`

   Preserve the approximately 1.4 GB partial train archive. No complete train/dev/holdout manifest exists yet. Keep the predetermined sequence split.

4. Add a same-detector ByteTrack baseline and the disappearance-episode scorer, including wrong-person reuse and pre-gap failures.
5. Run ONNX export and sequence parity. Export code exists but has not been exercised. Then follow GOAL.md's bounded isolated Nano procedure. SmartCampus has not been modified or stopped. Nano performance remains unverified.
6. Longer-gap training, explicit identity separation, calibrated abstention, distant re-entry tests, teacher probes and distillation remain undone. Four-frame training does not establish ten-second memory.

## Findings to preserve

- Upstream cached position embeddings stayed on CPU after model.to(cuda). The loader now registers them as nonpersistent buffers after strict loading.
- Disable denoising after strict checkpoint loading, otherwise an expected embedding parameter disappears.
- Default AMP loss scale caused nonfinite gradients at step zero. Initial scale 128 passed the subsequent 100 steps.
- Bootstrap owns two small upstream patches: optional final decoder features and lazy import of the unused FLOP profiler. No calflops, Transformers or teacher weights are required by current inference.
- The detector preview includes audience members as well as dancers. Inspect annotation scope before treating every unmatched visible person as background. DanceTrack alone cannot establish all-person behavior.
- CUDA and major dependency wheels are cached under data/wheels. Do not download them again.
- Runtime has one joint neural decoder and no external appearance/geometry matching. Pilot training omits native FDR auxiliary/local losses and teacher objectives. Newborn suppression and continuity remain early learned behavior.

GOAL.md remains the complete research and acceptance brief. No full tracking goal has been marked achieved.
