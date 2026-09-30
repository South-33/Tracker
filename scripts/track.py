"""Track people with YOLO26n and the project-owned causal tracking policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from tracker.causal import CausalPersonTracker
from tracker.model import TrackingYOLO
from tracker.runtime import CausalTrackerRuntime
from tracker.splits import assert_research_eval

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tracker_config(
    device,
    appearance_threshold,
    with_reid=True,
    new_track_threshold=0.45,
    track_low_threshold=0.10,
    gmc_method="sparseOptFlow",
):
    config = YAML.load(check_yaml("botsort.yaml"))
    config["with_reid"] = with_reid
    config["model"] = "auto"
    config["device"] = str(device)
    config["appearance_thresh"] = appearance_threshold
    config["new_track_thresh"] = new_track_threshold
    config["track_low_thresh"] = track_low_threshold
    config["gmc_method"] = gmc_method
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
    track_low_threshold=0.10,
    detector_confidence=0.10,
    owner_checkpoint=None,
    assignment_alpha=0.2,
    assignment_cost_budget=0.00025,
    gmc_method="sparseOptFlow",
    gmc_max_corners=100,
    coast_frames=0,
    coast_min_active_tracks=0,
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
        track_low_threshold=track_low_threshold,
        gmc_method=gmc_method,
    )
    tracker = (
        BOTSORT(IterableSimpleNamespace(**config))
        if owner_checkpoint is None
        else CausalPersonTracker(
            IterableSimpleNamespace(**config),
            owner_checkpoint,
            owner_alpha=assignment_alpha,
            average_base_cost_budget=assignment_cost_budget,
            gmc_max_corners=gmc_max_corners,
            coast_frames=coast_frames,
            coast_min_active_tracks=coast_min_active_tracks,
        )
    )
    runtime = CausalTrackerRuntime(
        model,
        tracker,
        device=device,
        with_reid=with_reid,
        feature_mode=feature_mode,
        min_detections_for_reid=min_detections_for_reid,
        detector_confidence=detector_confidence,
    )
    rows = []
    total_started = time.perf_counter()
    for frame_number, path in enumerate(image_paths, 1):
        original = cv2.imread(str(path))
        if original is None:
            raise FileNotFoundError(path)
        tracks = runtime.step(original)

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
    runtime_stats = runtime.stats()
    return {
        "sequence": sequence,
        "frames": len(image_paths),
        "rows": len(rows),
        "reid_frames": runtime_stats["reid_frames"],
        "reid_detections": runtime_stats["reid_detections"],
        "compute_fps": len(image_paths) / runtime_stats["compute_seconds"],
        "pipeline_fps": len(image_paths) / (time.perf_counter() - total_started),
        "owner_tiebreak_frames": runtime_stats["owner_tiebreak_frames"],
        "owner_changed_frames": runtime_stats["owner_changed_frames"],
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
        help="Minimum detection score for starting a new track.",
    )
    parser.add_argument(
        "--track-low-threshold",
        type=float,
        default=0.10,
        help="Minimum score for low-confidence recovery detections.",
    )
    parser.add_argument(
        "--detector-confidence",
        type=float,
        default=0.10,
        help="Pre-tracker detector confidence floor.",
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
        help="Use the identical detector path with the unguarded reference tracker.",
    )
    parser.add_argument(
        "--owner-head",
        default=None,
        help="Optional owner-continuity checkpoint for guarded first-association tie-breaking.",
    )
    parser.add_argument("--assignment-alpha", type=float, default=0.2)
    parser.add_argument("--assignment-cost-budget", type=float, default=0.00025)
    parser.add_argument(
        "--gmc-method",
        choices=("sparseOptFlow", "none"),
        default="sparseOptFlow",
        help="Camera-motion compensation method.",
    )
    parser.add_argument(
        "--gmc-max-corners",
        type=int,
        default=100,
        help="Sparse optical-flow corner budget for the project-owned tracker.",
    )
    parser.add_argument("--coast-frames", type=int, default=0)
    parser.add_argument("--coast-min-active-tracks", type=int, default=0)
    args = parser.parse_args()
    assert_research_eval(args.sequences)
    if args.min_detections_for_reid < 1:
        raise ValueError("--min-detections-for-reid must be at least 1")
    if not 0 <= args.new_track_threshold <= 1:
        raise ValueError("--new-track-threshold must be between 0 and 1")
    if not 0 <= args.track_low_threshold <= 1:
        raise ValueError("--track-low-threshold must be between 0 and 1")
    if not 0 <= args.detector_confidence <= 1:
        raise ValueError("--detector-confidence must be between 0 and 1")
    if args.detector_confidence > args.track_low_threshold:
        raise ValueError(
            "--detector-confidence must be <= --track-low-threshold"
        )
    if args.assignment_alpha < 0:
        raise ValueError("--assignment-alpha must be non-negative")
    if args.assignment_cost_budget < 0:
        raise ValueError("--assignment-cost-budget must be non-negative")
    if args.gmc_max_corners < 5:
        raise ValueError("--gmc-max-corners must be at least 5")
    if args.coast_frames < 0:
        raise ValueError("--coast-frames must be non-negative")
    if args.coast_min_active_tracks < 0:
        raise ValueError("--coast-min-active-tracks must be non-negative")
    if args.owner_head and args.disable_reid:
        raise ValueError("--owner-head requires appearance features")

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
    owner_path = None if args.owner_head is None else ROOT / args.owner_head
    owner_checkpoint = (
        None
        if owner_path is None
        else torch.load(owner_path, map_location="cpu", weights_only=False)
    )
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
            track_low_threshold=args.track_low_threshold,
            detector_confidence=args.detector_confidence,
            owner_checkpoint=owner_path,
            assignment_alpha=args.assignment_alpha,
            assignment_cost_budget=args.assignment_cost_budget,
            gmc_method=args.gmc_method,
            gmc_max_corners=args.gmc_max_corners,
            coast_frames=args.coast_frames,
            coast_min_active_tracks=args.coast_min_active_tracks,
        )
        for sequence in args.sequences
    ]
    metadata = {
        "system": (
            "YOLO26n manual detector path + Ultralytics BoT-SORT"
            if args.disable_reid
            else (
                f"YOLO26n + {args.feature_mode} appearance feature + Ultralytics BoT-SORT"
                if owner_path is None
                else (
                    f"YOLO26n + {args.feature_mode} appearance feature + project-owned "
                    "causal tracker + guarded owner-continuity tie-break"
                )
            )
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
            "conf": args.detector_confidence,
            "iou": 0.7,
            "imgsz": 640,
        },
        "tracker": tracker_config(
            device,
            args.appearance_threshold,
            with_reid=not args.disable_reid,
            new_track_threshold=args.new_track_threshold,
            track_low_threshold=args.track_low_threshold,
            gmc_method=args.gmc_method,
        ),
        "reid_gate": {
            "min_detections": args.min_detections_for_reid,
        },
        "coast": {
            "frames": args.coast_frames,
            "min_active_tracks": args.coast_min_active_tracks,
        },
        "owner_continuity": (
            None
            if owner_path is None
            else {
                "checkpoint": args.owner_head,
                "checkpoint_sha256": sha256(owner_path),
                "alpha": args.assignment_alpha,
                "average_base_cost_budget": args.assignment_cost_budget,
                "gmc_max_corners": args.gmc_max_corners,
                "training": {
                    key: owner_checkpoint.get(key)
                    for key in (
                        "objective",
                        "epochs",
                        "lr",
                        "repo_commit",
                        "repo_dirty",
                        "identity_head",
                        "identity_head_sha256",
                        "train_sequences",
                        "validation_sequence",
                        "train_queries",
                        "validation_queries",
                        "before",
                        "validation",
                    )
                    if owner_checkpoint.get(key) is not None
                },
            }
        ),
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
