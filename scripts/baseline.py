"""Run the frozen YOLO26n + BoT-SORT baseline on DanceTrack sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def detector_weights() -> Path:
    target = ROOT / "weights" / "yolo26n.pt"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    YOLO("yolo26n.pt")  # Let Ultralytics fetch the official asset once.
    downloaded = ROOT / "yolo26n.pt"
    if not downloaded.exists():
        raise FileNotFoundError("Ultralytics did not download yolo26n.pt")
    shutil.move(downloaded, target)
    return target


def run_sequence(sequence: str, output: Path, model_path: Path, tracker_path: Path) -> dict:
    sequence_dir = ROOT / "data" / "dancetrack" / sequence
    images = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No frames found for {sequence}")

    model = YOLO(model_path)
    first = cv2.imread(str(images[0]))
    model.predict(first, classes=[0], conf=0.1, iou=0.7, imgsz=640, device=0, verbose=False)

    rows = []
    inference_seconds = 0.0
    total_started = time.perf_counter()
    for frame_number, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)
        started = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker=str(tracker_path),
            classes=[0],
            conf=0.1,
            iou=0.7,
            imgsz=640,
            device=0,
            verbose=False,
        )[0]
        inference_seconds += time.perf_counter() - started
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        scores = boxes.conf.detach().cpu().numpy()
        for box, track_id, score in zip(xyxy, ids, scores):
            x1, y1, x2, y2 = map(float, box)
            rows.append([
                frame_number, track_id, x1, y1, x2 - x1, y2 - y1,
                float(score), -1, -1, -1,
            ])
    total_seconds = time.perf_counter() - total_started

    output.mkdir(parents=True, exist_ok=True)
    prediction = output / f"{sequence}.txt"
    with prediction.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(",".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + "\n")
    return {
        "sequence": sequence,
        "frames": len(images),
        "tracks_rows": len(rows),
        "inference_tracker_fps": len(images) / inference_seconds,
        "pipeline_fps_including_jpeg_io": len(images) / total_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="+", help="DanceTrack sequence names")
    parser.add_argument("--output", default="runs/yolo26n-botsort")
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable BoT-SORT ReID using native YOLO detector features.",
    )
    args = parser.parse_args()

    output = ROOT / args.output
    weight = detector_weights()
    tracker = Path(ULTRALYTICS_ROOT) / "cfg" / "trackers" / "botsort.yaml"
    tracker_config = tracker.read_text(encoding="utf-8")
    tracker_path = tracker
    if args.reid:
        tracker_config = tracker_config.replace("with_reid: False", "with_reid: True")
        if "with_reid: True" not in tracker_config:
            raise RuntimeError("could not enable ReID in the installed botsort.yaml")
        output.mkdir(parents=True, exist_ok=True)
        tracker_path = output / "botsort-reid.yaml"
        tracker_path.write_text(tracker_config, encoding="utf-8")

    stats = [
        run_sequence(sequence, output, weight, tracker_path)
        for sequence in args.sequences
    ]

    metadata = {
        "system": (
            "YOLO26n + Ultralytics BoT-SORT native ReID"
            if args.reid
            else "YOLO26n + Ultralytics BoT-SORT"
        ),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "repo_commit": git_commit(),
        "model": str(weight.relative_to(ROOT)),
        "model_sha256": sha256(weight),
        "tracker": tracker_path.name,
        "tracker_config": tracker_config,
        "reid": args.reid,
        "settings": {"classes": [0], "conf": 0.1, "iou": 0.7, "imgsz": 640, "device": 0},
        "sequences": stats,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
