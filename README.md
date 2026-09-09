# Remember tracker

Standalone research into a compact person detector that carries identity memory between frames. Read [GOAL.md](GOAL.md) for the target and research, and [RESULTS.md](RESULTS.md) for what actually works and the next action.

The current candidate adds 64 resident queries to the pretrained RT-DETRv4-S detector's 300 discovery queries. Both use the same decoder. Its wrapper allocates anonymous IDs and expires memory; it does not perform appearance matching, Hungarian association, or Kalman tracking. Hungarian assignment is used only in training and evaluation.

This is an early experiment. A runnable model is not evidence of reliable occlusion or re-entry handling.

## Setup on the laptop

Use PowerShell in `D:\Project\tracker`. The environment is separate from SmartCampus.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/bootstrap.py --evaluation
.\.venv\Scripts\python.exe -m pytest -q
```

Install CUDA PyTorch before the editable package so dependency resolution does not choose a CPU wheel. Do not upgrade Jetson's PyTorch, TensorRT, or JetPack for this project. The pinned source retains its upstream license notices. Bootstrap makes only two small upstream changes: optional final decoder feature output and lazy loading of the unused FLOP profiler. No teacher weights or Transformers package are required at runtime.

If the CUDA wheel download stalls, the range downloader resumes completed parts:

```powershell
python scripts/fetch_file.py 'https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl' 'data/wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl'
.\.venv\Scripts\python.exe -m pip install 'data/wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl'
.\.venv\Scripts\python.exe -m pip install torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
```

The official checkpoint can also be fetched with:

```powershell
python scripts/fetch_file.py 'https://drive.usercontent.google.com/download?id=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z&export=download&confirm=t' weights/rtv4_s.pth
.\.venv\Scripts\python.exe scripts/bootstrap.py --evaluation
```

## Data

```powershell
python scripts/fetch_data.py train1.zip val.zip
python scripts/prepare_data.py
```

To start earlier, `python scripts/prepare_data.py --remote --frames 120` fetches selected ZIP members through HTTP ranges. It freezes six training, two development and four holdout sequences by source name, then uses their first 120 frames. That produces `data/manifest-120.json`. Full preparation produces `data/manifest-full.json`. Downloading a full archive and selected members concurrently duplicates some transfer; do not restart a second copy of either process.

`data/manifest-debug.json`, when present, contains only the first two training sequences. It is for diagnosing implementation failures. It contains no development or holdout set. Whole sequences keep the same split when expanded. The source annotation's constant last column is not a visibility measurement.

## Run

```powershell
.\.venv\Scripts\python.exe -m tracker.cli inspect --image data/dancetrack/dancetrack0001/img1/00000001.jpg
.\.venv\Scripts\python.exe -m tracker.cli verify
.\.venv\Scripts\python.exe -m tracker.cli train --debug --manifest data/manifest-debug.json --steps 100 --unroll 4 --save-every 10 --output runs/debug
.\.venv\Scripts\python.exe -m tracker.cli train --debug --manifest data/manifest-debug.json --steps 500 --unroll 4 --save-every 10 --output runs/debug --resume runs/debug/last.pt
```

Use `--initialize runs/unroll12/last.pt` to carry learned weights into a new dataset experiment with a fresh optimizer. Use `--resume` only to continue the same dataset, seed, learning rate and debug mode. Both options are mutually exclusive. Current evidence and the best continuation command are in RESULTS.md.

### Laptop training

Frozen image features are cached automatically under `data/features`, keyed by the actual backbone/encoder weights, Torch version, precision, input size and preprocessing. The first pass builds the cache; later passes avoid image decoding and frozen encoder work. Changed image file metadata or annotation hashes invalidate that frame's entry. The cache is valid only for deterministic preprocessing and frozen eval-mode features; disable it before introducing image augmentation or unfreezing those modules. It is not a cache of labels inferred by a teacher.

Use `--no-feature-cache` for an uncached reference. `--cache-mib 512` optionally keeps a bounded working set on GPU; the default is disk caching with no extra GPU working set so longer unrolls have more VRAM. The measured debug sampling pattern did not reuse the 512 MiB working set, so it is not enabled by default. Twelve-frame FP16 unrolls fit comfortably in this laptop's 8 GB GPU. Keep one GPU training job active at a time.

The paired 40-step test with identical initialization/seed produced identical losses with and without caching. Warm disk caching was faster; exact time, peak memory, cold-cache cost and limits are in RESULTS.md and `runs/training-speed.json`. Those measurements do not establish speed for a different architecture or longer sequence length.

`--steps` is the total completed optimizer-step target, including resumed steps. The checkpoint includes optimizer, scaler, RNG states, source revision, dataset-manifest hash, settings and training progress. Checkpoints resume at sequence boundaries; memory is intentionally initialized for each sampled training clip. Debug sampling cycles through short windows from two sequences. Longer-gap learning is not established by this four-frame pilot.

The initial loss uses person varifocal classification, L1/GIoU boxes, resident continuity, and an IoU quality gate. It does not yet include upstream FDR local/auxiliary losses, explicit identity contrastive supervision, or teacher distillation. Backbone and encoder are frozen; decoder and memory heads train. Keep this limitation visible when judging results.

Replay one complete available sequence prefix with the real returned state:

```powershell
.\.venv\Scripts\python.exe -m tracker.cli replay --checkpoint runs/debug/last.pt --manifest data/manifest-debug.json --split train --sequence dancetrack0001 --video --output runs/debug-replay
.\.venv\Scripts\python.exe scripts/evaluate.py --manifest data/manifest-debug.json --run runs/debug-replay
```

Run the same commands on a sequence from the frozen development/holdout manifest after debug training succeeds. Choose thresholds on development data. No checkpoint means the added tracking parameters are untrained. Replay writes MOT rows, per-frame observations, timing metadata, and optionally an uncut MP4. Timings from this laptop are development measurements, not Nano performance claims.

Bootstrap pins TrackEval's source. Standard MOT metrics do not replace the separate disappearance-episode protocol in GOAL.md; a reviewed real event manifest remains required.

The evaluator's NumPy compatibility patch is owned by `bootstrap.py --evaluation`. A synthetic perfect-box identity swap test verifies that official IDF1 falls and ID switches increase. `src/tracker/episodes.py` scores reviewed episode associations, with wrong-person reuse taking precedence and pre-gap failures retained in the denominator; its tests cover delayed switches and abstention. The real event manifest and the bridge from reviewed ground truth to evaluation associations still need implementation.

The optional baseline uses the official ByteTrack association code with the same letterboxed detector inputs:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[baseline]'
.\.venv\Scripts\python.exe scripts/bootstrap.py --baseline --evaluation --skip-weights
.\.venv\Scripts\python.exe scripts/baseline.py --sequence dancetrack0001 --checkpoint runs/unroll12/last.pt --output runs/byte-unroll12
.\.venv\Scripts\python.exe scripts/evaluate.py --manifest data/manifest-debug.json --run runs/byte-unroll12
```

Omit `--checkpoint` to use the original pretrained detector. The fine-tuned model's discovery predictions without resident queries are a distinct detector operating mode; report baseline detection accuracy as well as association accuracy. Baseline packages and code are not imported by the candidate runtime.

## Export

```powershell
.\.venv\Scripts\python.exe -m tracker.cli export --checkpoint runs/debug/last.pt --output runs/candidate.onnx
```

Check independent FP32 feedback rollouts after export:

```powershell
.\.venv\Scripts\python.exe scripts/verify_onnx.py --checkpoint runs/unroll12/last.pt --onnx runs/unroll12/candidate.onnx --output runs/unroll12/onnx-parity.json
```

This checks CPU PyTorch against ONNX Runtime CPU with actual frames and independently updated memory. It does not establish TensorRT or FP16 parity.

The export has a fixed B1 image and explicit recent/reliable appearance, boxes, age, validity and elapsed-time inputs. It outputs fixed query boxes, person logits, features, continuity logits and quality logits. The small memory-update/ID wrapper remains outside the ONNX graph and must be ported and validated with the graph. This export alone does not satisfy the recurrent TensorRT deployment gate.

No SmartCampus code, services, model ownership or production configuration belongs in this project. Follow GOAL.md's separate Jetson benchmark procedure when laptop correctness is established.
