"""Run the packaged causal person tracker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from tracker.system import PersonTracker
from tracker.splits import CONSUMED_HOLDOUT, RESERVED_HOLDOUT

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--output", default="runs/causal-system")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--final-holdout", action="store_true")
    parser.add_argument("--historical-holdout", action="store_true")
    args = parser.parse_args()
    if args.final_holdout and args.historical_holdout:
        raise ValueError("choose only one holdout mode")
    for sequence in args.sequences:
        if sequence in RESERVED_HOLDOUT and not args.final_holdout:
            raise ValueError(
                f"{sequence} is RESERVED_HOLDOUT; use --final-holdout only after freeze"
            )
        if sequence in CONSUMED_HOLDOUT and not args.historical_holdout:
            raise ValueError(
                f"{sequence} is CONSUMED_HOLDOUT; use --historical-holdout only for reproduction"
            )

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for sequence in args.sequences:
        tracker = PersonTracker(
            ROOT / args.tracker,
            device=args.device,
        )
        image_paths = sorted(
            (ROOT / "data" / "dancetrack" / sequence / "img1").glob("*.jpg")
        )
        if not image_paths:
            raise FileNotFoundError(f"No frames found for {sequence}")

        rows = []
        started = time.perf_counter()
        for frame_number, path in enumerate(image_paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            tracks = tracker.step(image)
            for track in tracks:
                x1, y1, x2, y2, identity, score, _cls, _index = track.tolist()
                rows.append(
                    [
                        frame_number,
                        int(identity),
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        score,
                        -1,
                        -1,
                        -1,
                    ]
                )

        with (output / f"{sequence}.txt").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    ",".join(
                        f"{value:.6f}" if isinstance(value, float) else str(value)
                        for value in row
                    )
                    + "\n"
                )
        stats = tracker.stats()
        stats.update(
            {
                "sequence": sequence,
                "rows": len(rows),
                "pipeline_fps": len(image_paths)
                / max(time.perf_counter() - started, 1e-9),
            }
        )
        all_stats.append(stats)

    metadata = {
        "system": "packaged causal person tracker",
        "tracker": args.tracker,
        "config": tracker.config.metadata(),
        "sequences": all_stats,
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
