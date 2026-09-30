"""Balanced DanceTrack report with split and density breakdowns.

This script scores existing MOT prediction files. It does not run or tune the
tracker. Development is the default. Final holdout reporting requires an
explicit acknowledgement flag to make accidental leakage harder.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

from tracker.splits import (
    CALIBRATION,
    CONSUMED_HOLDOUT,
    DEV,
    RESEARCH_EVAL,
    RESERVED_HOLDOUT,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "Recall", "Precision")


def density_stats(sequence: str) -> dict:
    sequence_dir = ROOT / "data" / "dancetrack" / sequence
    frames = len(list((sequence_dir / "img1").glob("*.jpg")))
    counts = [0] * (frames + 1)
    gt = sequence_dir / "gt" / "gt.txt"
    with gt.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            if 1 <= frame <= frames:
                counts[frame] += 1
    average = sum(counts) / max(frames, 1)
    density = "sparse" if average < 7 else "medium" if average < 15 else "dense"
    return {
        "frames": frames,
        "avg_people_per_frame": average,
        "max_people_per_frame": max(counts, default=0),
        "density": density,
    }


def evaluate(run: Path, sequences: list[str], mode: str) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        str(run),
        *sequences,
    ]
    if mode == "historical":
        command.append("--historical-holdout")
    elif mode == "final":
        command.append("--final-holdout")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    output = completed.stdout.strip()
    start = output.rfind("\n{")
    if start >= 0:
        output = output[start + 1 :]
    elif not output.startswith("{"):
        start = output.find("{")
        if start < 0:
            raise RuntimeError("evaluate.py produced no JSON")
        output = output[start:]
    return json.loads(output)


def macro(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {
        metric: statistics.fmean(row[metric] for row in rows)
        for metric in METRICS
    }
    frames = max(sum(row["frames"] for row in rows), 1)
    result["IDSW_per_1k_frames"] = (
        sum(row["IDSW"] for row in rows) / frames * 1000
    )
    result["Frag_per_1k_frames"] = (
        sum(row["Frag"] for row in rows) / frames * 1000
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Folder containing <sequence>.txt predictions")
    parser.add_argument(
        "--split",
        choices=("calibration", "dev", "research", "historical", "final"),
        default="research",
        help="Data role to summarize. Default is calibration + development.",
    )
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required with --split final. Reserved holdout must not guide tuning.",
    )
    parser.add_argument(
        "--confirm-historical",
        action="store_true",
        help="Required with --split historical. Reproduction only, never tuning.",
    )
    parser.add_argument("--output", default=None, help="Optional report JSON path")
    args = parser.parse_args()

    split_map = {
        "calibration": list(CALIBRATION),
        "dev": list(DEV),
        "research": list(RESEARCH_EVAL),
        "historical": list(CONSUMED_HOLDOUT),
        "final": list(RESERVED_HOLDOUT),
    }
    if args.split == "final" and not args.confirm_final:
        raise SystemExit(
            "FINAL_HOLDOUT is evaluation-only. Re-run with --confirm-final "
            "only for a frozen model or an already-consumed final result."
        )
    if args.split == "historical" and not args.confirm_historical:
        raise SystemExit(
            "CONSUMED_HOLDOUT is historical evidence only. Re-run with "
            "--confirm-historical only to reproduce an old report."
        )

    run = (ROOT / args.run).resolve()
    sequences = [
        sequence
        for sequence in split_map[args.split]
        if (run / f"{sequence}.txt").exists()
    ]
    missing = [
        sequence
        for sequence in split_map[args.split]
        if sequence not in sequences
    ]
    if missing:
        raise FileNotFoundError(
            f"{run} is missing predictions for: {', '.join(missing)}"
        )

    rows = []
    for sequence in sequences:
        row = {"sequence": sequence, **density_stats(sequence)}
        metrics = evaluate(run, [sequence], args.split)
        metrics.pop("sequences", None)
        row.update(metrics)
        rows.append(row)

    by_density = {}
    for density in ("sparse", "medium", "dense"):
        density_rows = [row for row in rows if row["density"] == density]
        if density_rows:
            by_density[density] = macro(density_rows)

    micro = evaluate(run, sequences, args.split)
    report = {
        "run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "split": args.split,
        "holdout": args.split in {"historical", "final"},
        "headline": "macro_per_sequence",
        "macro_per_sequence": macro(rows),
        "micro_combined": {
            key: value for key, value in micro.items() if key != "sequences"
        },
        "by_density": by_density,
        "sequences": rows,
    }

    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
