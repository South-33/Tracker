"""Benchmark the packaged tracker on one sequence and check the FPS target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import cv2
import torch

from tracker.system import PersonTracker

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence")
    parser.add_argument("--tracker", default="runs/person-tracker.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-memory-gib", type=float, default=8.0)
    parser.add_argument("--output", default="runs/target-benchmark.json")
    args = parser.parse_args()
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.max_memory_gib <= 0:
        raise ValueError("--max-memory-gib must be positive")

    tracker = PersonTracker(
        ROOT / args.tracker,
        device=args.device,
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    paths = sorted(
        (ROOT / "data" / "dancetrack" / args.sequence / "img1").glob("*.jpg")
    )
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    if not paths:
        raise FileNotFoundError(args.sequence)

    warmup = min(max(args.warmup, 0), len(paths))
    measured = 0
    elapsed = 0.0
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        if index == warmup:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
        tracker.step(frame)
        if index >= warmup:
            measured += 1

    if measured:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    fps = measured / max(elapsed, 1e-9)
    device_name = (
        torch.cuda.get_device_name(0)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else platform.processor()
    )
    report = {
        "tracker": args.tracker,
        "sequence": args.sequence,
        "device": args.device,
        "device_name": device_name,
        "torch": torch.__version__,
        "frames_total": len(paths),
        "warmup_frames": warmup,
        "measured_frames": measured,
        "seconds": elapsed,
        "fps": fps,
        "target_fps": args.target_fps,
        "passes_target": fps >= args.target_fps,
        "max_memory_gib": args.max_memory_gib,
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated())
        report["peak_cuda_memory_bytes"] = peak_bytes
        report["peak_cuda_memory_gib"] = peak_bytes / (1024**3)
        report["passes_memory_target"] = (
            report["peak_cuda_memory_gib"] <= args.max_memory_gib
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passes_target"] else 2)


if __name__ == "__main__":
    main()
