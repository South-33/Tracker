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


def hardware_metadata(device: torch.device) -> dict:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        result["nv_tegra_release"] = tegra.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence")
    parser.add_argument("--tracker", default="runs/person-tracker.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--target-fps",
        "--min-fps",
        dest="target_fps",
        type=float,
        default=15.0,
    )
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
    step_seconds = 0.0
    read_seconds = 0.0
    for index, path in enumerate(paths):
        read_started = time.perf_counter()
        frame = cv2.imread(str(path))
        read_elapsed = time.perf_counter() - read_started
        if frame is None:
            raise FileNotFoundError(path)
        if index < warmup:
            tracker.step(frame)
            continue
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        tracker.step(frame)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        step_seconds += time.perf_counter() - started
        read_seconds += read_elapsed
        measured += 1

    tracker_fps = measured / max(step_seconds, 1e-9)
    pipeline_fps = measured / max(step_seconds + read_seconds, 1e-9)
    report = {
        "tracker": args.tracker,
        "sequence": args.sequence,
        "frames_total": len(paths),
        "warmup_frames": warmup,
        "measured_frames": measured,
        "tracker_step_seconds": step_seconds,
        "tracker_step_fps": tracker_fps,
        "jpeg_read_ms_per_frame": read_seconds * 1000.0 / max(measured, 1),
        "conservative_jpeg_pipeline_fps": pipeline_fps,
        "target_fps": args.target_fps,
        "passes_tracker_target": tracker_fps >= args.target_fps,
        "passes_conservative_pipeline_target": pipeline_fps >= args.target_fps,
        "max_memory_gib": args.max_memory_gib,
        "hardware": hardware_metadata(torch.device(args.device)),
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        report["peak_cuda_allocated_bytes"] = peak_allocated
        report["peak_cuda_allocated_gib"] = peak_allocated / (1024**3)
        report["peak_cuda_reserved_bytes"] = peak_reserved
        report["peak_cuda_reserved_gib"] = peak_reserved / (1024**3)
        report["passes_memory_target"] = (
            report["peak_cuda_reserved_gib"] <= args.max_memory_gib
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    passes = report["passes_tracker_target"] and report.get(
        "passes_memory_target",
        True,
    )
    raise SystemExit(0 if passes else 2)


if __name__ == "__main__":
    main()
