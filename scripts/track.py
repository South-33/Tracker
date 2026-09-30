"""Track people with YOLO26n, a learned 64D feature, and BoT-SORT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.nms import non_max_suppression

from tracker.data import letterbox, restore_boxes
from tracker.model import TrackingYOLO

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tracker_config(
    device,
    appearance_threshold,
    with_reid=True,
    new_track_threshold=0.45,
):
    config = YAML.load(check_yaml("botsort.yaml"))
    config["with_reid"] = with_reid
    config["model"] = "auto"
    config["device"] = str(device)
    config["appearance_thresh"] = appearance_threshold
    config["new_track_thresh"] = new_track_threshold
    return config


@torch.inference_mode()
def run_sequence(
    model,
    sequence,
    output,
    *,
    device,
    with_reid,
    feature_mode,
    min_detections_for_reid,
    appearance_threshold,
    new_track_threshold,
):
    sequence_dir = ROOT / "data" / "dancetrack" / sequence
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No frames found for {sequence}")

    config = tracker_config(
        device,
        appearance_threshold,
        with_reid=with_reid,
        new_track_threshold=new_track_threshold,
    )
    tracker = BOTSORT(IterableSimpleNamespace(**config))
    rows = []
    reid_frames = 0
    reid_detections = 0
    compute_seconds = 0.0
    total_started = time.perf_counter()
    for frame_number, path in enumerate(image_paths, 1):
        original = cv2.imread(str(path))
        if original is None:
            raise FileNotFoundError(path)
        image, _ = letterbox(original, [], 640)
        image = image.unsqueeze(0).to(device)

        started = time.perf_counter()
        raw, pyramid = model.extract(image)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(
            prediction,
            conf_thres=0.1,
            iou_thres=0.7,
            classes=[0],
            max_det=300,
        )[0]
        if len(detections):
            boxes = detections[:, :4]
            use_reid = with_reid and len(detections) >= min_detections_for_reid
            if use_reid and feature_mode == "raw":
                features = torch.nn.functional.normalize(
                    model.pool_boxes(pyramid, [boxes]),
                    dim=1,
                )
            else:
                features = model.embed_boxes(pyramid, [boxes]) if use_reid else None
            if features is not None:
                reid_frames += 1
                reid_detections += len(features)
            original_boxes = restore_boxes(boxes, original.shape[:2])
            results = Boxes(
                torch.cat((original_boxes, detections[:, 4:6]), 1).cpu(),
                original.shape[:2],
            ).numpy()
            tracks = tracker.update(
                results,
                original,
                feats=features.float().cpu().numpy() if features is not None else None,
            )
        else:
            results = Boxes(torch.empty((0, 6)), original.shape[:2]).numpy()
            tracks = tracker.update(results, original, feats=None)
        compute_seconds += time.perf_counter() - started

        for track in tracks:
            x1, y1, x2, y2, public_id, score, _cls, _index = track.tolist()
            rows.append(
                [
                    frame_number,
                    int(public_id),
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

    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{sequence}.txt").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                ",".join(
                    f"{value:.6f}" if isinstance(value, float) else str(value)
                    for value in row
                )
                + "\n"
            )
    return {
        "sequence": sequence,
        "frames": len(image_paths),
        "rows": len(rows),
        "reid_frames": reid_frames,
        "reid_detections": reid_detections,
        "compute_fps": len(image_paths) / compute_seconds,
        "pipeline_fps": len(image_paths) / (time.perf_counter() - total_started),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--head", default="runs/identity-head-v2/head.pt")
    parser.add_argument("--output", default="runs/learned-reid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-detections-for-reid",
        type=int,
        default=10,
        help="Use appearance features only on frames with at least this many detections.",
    )
    parser.add_argument("--appearance-threshold", type=float, default=0.8)
    parser.add_argument(
        "--new-track-threshold",
        type=float,
        default=0.45,
        help="Minimum detection score for starting a new BoT-SORT track.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("trained", "random", "raw"),
        default="trained",
        help="Appearance representation: trained 64D head, seeded untrained 64D head, or raw pooled YOLO features.",
    )
    parser.add_argument(
        "--disable-reid",
        action="store_true",
        help="Use the identical detector/BoT-SORT path without appearance features.",
    )
    args = parser.parse_args()
    if args.min_detections_for_reid < 1:
        raise ValueError("--min-detections-for-reid must be at least 1")
    if not 0 <= args.new_track_threshold <= 1:
        raise ValueError("--new-track-threshold must be between 0 and 1")

    device = torch.device(args.device)
    checkpoint = None
    head_path = None
    if args.disable_reid or args.feature_mode in {"random", "raw"}:
        detector_path = ROOT / "weights" / "yolo26n.pt"
        if args.feature_mode == "random":
            torch.manual_seed(0)
        model = TrackingYOLO(detector_path).to(device).eval()
    else:
        head_path = ROOT / args.head
        checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
        detector_path = ROOT / checkpoint["detector"]
        actual_detector_sha = sha256(detector_path)
        if actual_detector_sha != checkpoint["detector_sha256"]:
            raise RuntimeError(
                f"detector hash mismatch: checkpoint expects {checkpoint['detector_sha256']}, "
                f"found {actual_detector_sha}"
            )
        model = TrackingYOLO(
            detector_path,
            checkpoint["embedding_dim"],
        ).to(device).eval()
        model.embedding.load_state_dict(checkpoint["embedding"])
    actual_detector_sha = sha256(detector_path)

    output = ROOT / args.output
    stats = [
        run_sequence(
            model,
            sequence,
            output,
            device=device,
            with_reid=not args.disable_reid,
            feature_mode=args.feature_mode,
            min_detections_for_reid=args.min_detections_for_reid,
            appearance_threshold=args.appearance_threshold,
            new_track_threshold=args.new_track_threshold,
        )
        for sequence in args.sequences
    ]
    metadata = {
        "system": (
            "YOLO26n manual detector path + Ultralytics BoT-SORT"
            if args.disable_reid
            else f"YOLO26n + {args.feature_mode} appearance feature + Ultralytics BoT-SORT"
        ),
        "feature_mode": "none" if args.disable_reid else args.feature_mode,
        "feature_seed": 0 if args.feature_mode == "random" and not args.disable_reid else None,
        "head": None if head_path is None else args.head,
        "head_sha256": None if head_path is None else sha256(head_path),
        "detector_sha256": actual_detector_sha,
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "repo_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "head_training": (
            None
            if checkpoint is None
            else {
                key: checkpoint.get(key)
                for key in (
                    "repo_commit",
                    "repo_dirty",
                    "objective",
                    "steps",
                    "lr",
                    "temperature",
                    "validation_samples",
                    "data",
                    "before",
                    "after",
                )
                if checkpoint.get(key) is not None
            }
        ),
        "detector_settings": {
            "classes": [0],
            "conf": 0.1,
            "iou": 0.7,
            "imgsz": 640,
        },
        "tracker": tracker_config(
            device,
            args.appearance_threshold,
            with_reid=not args.disable_reid,
            new_track_threshold=args.new_track_threshold,
        ),
        "reid_gate": {
            "min_detections": args.min_detections_for_reid,
        },
        "sequences": stats,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
