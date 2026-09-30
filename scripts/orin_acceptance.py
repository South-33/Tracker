"""Run the frozen tracker acceptance benchmark on an NVIDIA Jetson target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", nargs="?", default="dancetrack0020")
    parser.add_argument("--tracker", default="runs/person-tracker.pt")
    parser.add_argument("--min-fps", type=float, default=15.0)
    parser.add_argument("--max-memory-gib", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", default="runs/orin-benchmark.json")
    args = parser.parse_args()

    tegra = Path("/etc/nv_tegra_release")
    if not tegra.exists():
        raise SystemExit(
            "This acceptance command must be run on an NVIDIA Jetson target "
            "(/etc/nv_tegra_release is missing)."
        )

    model_path = Path("/proc/device-tree/model")
    hardware = {
        "nv_tegra_release": tegra.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip(),
        "machine": command_output("uname", "-m"),
        "kernel": command_output("uname", "-a"),
        "model": (
            model_path.read_bytes()
            .rstrip(b"\x00")
            .decode("utf-8", errors="replace")
            if model_path.exists()
            else None
        ),
        "nvpmodel": command_output("nvpmodel", "-q"),
        "jetson_clocks": command_output("jetson_clocks", "--show"),
        "tensorrt": command_output(
            "dpkg-query",
            "-W",
            "-f=$"+"{Version}",
            "libnvinfer10",
        ),
    }

    benchmark_command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_tracker.py"),
        args.sequence,
        "--tracker",
        args.tracker,
        "--warmup",
        str(args.warmup),
        "--min-fps",
        str(args.min_fps),
        "--max-memory-gib",
        str(args.max_memory_gib),
        "--output",
        args.output,
    ]
    completed = subprocess.run(
        benchmark_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    benchmark_path = ROOT / args.output
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.exists()
        else None
    )
    report = {
        "hardware": hardware,
        "benchmark": benchmark,
        "benchmark_exit_code": completed.returncode,
    }
    if completed.stderr:
        report["benchmark_stderr"] = completed.stderr.strip()

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

    passed = bool(
        benchmark
        and benchmark.get("passes_tracker_target")
        and benchmark.get("passes_memory_target", True)
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
