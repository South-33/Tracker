"""Paired laptop annotation-index benchmark; no images, model or heldout data."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from tracker.data import index_annotations


def old_index(rows, frames):
    truth = {}
    for frame in frames:
        selected = rows[rows[:, 0] == frame]
        truth[frame] = (selected[:, 1].astype(np.int64), selected[:, 2:6].astype(np.float32))
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args()
    manifest = Path("data/manifest-train-full-dense-no0096.json")
    sources = []
    for record in json.loads(manifest.read_text())["records"]:
        if record["split"] == "train":
            rows = np.loadtxt(Path("data/dancetrack") / record["sequence"] / "gt/gt.txt", delimiter=",", ndmin=2)
            sources.append((record, rows))
    for record, rows in sources:
        old = old_index(rows, record["available_frames"])
        new = index_annotations(rows, record["available_frames"])
        for frame in old:
            for before, after in zip(old[frame], new[frame]):
                np.testing.assert_array_equal(before, after)
    timings = {"previous": [], "grouped": []}
    for repeat in range(6):
        methods = [("previous", old_index), ("grouped", index_annotations)]
        for name, function in methods if repeat % 2 == 0 else reversed(methods):
            started = time.perf_counter()
            for record, rows in sources:
                function(rows, record["available_frames"])
            timings[name].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    result = {
        "scope": "CPU annotation indexing only, loaded arrays; excludes CSV/image decoding and model training. No training-step speed claim.",
        "sequences": [r["sequence"] for r, _ in sources],
        "frames": sum(len(r["available_frames"]) for r, _ in sources),
        "exact_label_parity": True, "seconds": timings, "median_seconds": medians,
        "speedup": medians["previous"] / medians["grouped"],
    }
    output = Path("runs/laptop-data-path/result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
