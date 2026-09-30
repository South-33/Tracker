"""Run causal learned association with tiny bounded anonymous-track memory."""
from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import cv2
from scipy.optimize import linear_sum_assignment
import torch
import ultralytics
from ultralytics.utils.nms import non_max_suppression

from tracker.model import TemporalYOLO
from tracker.data import letterbox, restore_boxes

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Track:
    public_id: int
    box: torch.Tensor
    latest: torch.Tensor
    prototypes: list[torch.Tensor] = field(default_factory=list)
    velocity: torch.Tensor | None = None
    age: int = 0

    def update(self, box: torch.Tensor, embedding: torch.Tensor, dt_seconds: float):
        self.velocity = (box - self.box) / max(dt_seconds, 1e-3)
        self.box = box
        self.latest = embedding
        self.age = 0
        if len(self.prototypes) < 5:
            self.prototypes.append(embedding)
            return
        similarities = torch.stack(self.prototypes) @ embedding
        if float(similarities.max()) < 0.90:
            self.prototypes.pop(0)
            self.prototypes.append(embedding)


class MemoryTracker:
    def __init__(self, model: TemporalYOLO, fps: float, max_age_seconds=5.0, birth_threshold=0.25):
        self.model = model
        self.fps = fps
        self.max_age = round(max_age_seconds * fps)
        self.birth_threshold = birth_threshold
        self.tracks: list[Track] = []
        self.next_id = 1

    def step(self, boxes: torch.Tensor, embeddings: torch.Tensor, scores: torch.Tensor):
        for track in self.tracks:
            track.age += 1
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]

        assigned = {}
        if self.tracks and len(boxes):
            proto_embeddings = []
            proto_boxes = []
            proto_velocity = []
            proto_known = []
            proto_gap = []
            proto_track = []
            for track_index, track in enumerate(self.tracks):
                for prototype in track.prototypes:
                    proto_embeddings.append(prototype)
                    proto_boxes.append(track.box)
                    if track.velocity is None:
                        proto_velocity.append(torch.zeros_like(track.box))
                        proto_known.append(0.0)
                    else:
                        proto_velocity.append(track.velocity)
                        proto_known.append(1.0)
                    proto_gap.append(track.age / self.fps)
                    proto_track.append(track_index)

            proto_embeddings = torch.stack(proto_embeddings)
            proto_boxes = torch.stack(proto_boxes)
            proto_velocity = torch.stack(proto_velocity)
            proto_known = torch.tensor(proto_known, device=boxes.device)
            proto_gap = torch.tensor(proto_gap, device=boxes.device)
            proto_track = torch.tensor(proto_track, device=boxes.device, dtype=torch.long)

            prototype_logits = self.model.association_scores(
                proto_embeddings,
                proto_boxes,
                proto_velocity,
                proto_known,
                embeddings,
                boxes,
                proto_gap,
            )
            track_logits = torch.full(
                (len(self.tracks), len(boxes)), -1e6,
                device=boxes.device, dtype=prototype_logits.dtype,
            )
            track_logits.scatter_reduce_(
                0,
                proto_track[:, None].expand(-1, len(boxes)),
                prototype_logits,
                reduce="amax",
                include_self=True,
            )
            latest = torch.stack([track.latest for track in self.tracks])
            gaps = torch.tensor([track.age / self.fps for track in self.tracks], device=boxes.device)
            absent_logits = self.model.absence_scores(latest, gaps)

            track_count, detection_count = track_logits.shape
            choices = torch.full(
                (track_count, detection_count + track_count), -1e6,
                device=boxes.device, dtype=track_logits.dtype,
            )
            choices[:, :detection_count] = track_logits
            index = torch.arange(track_count, device=boxes.device)
            choices[index, detection_count + index] = absent_logits
            rows, columns = linear_sum_assignment((-choices).detach().cpu().numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                if column >= detection_count:
                    continue
                track = self.tracks[row]
                dt_seconds = track.age / self.fps
                track.update(boxes[column], embeddings[column], dt_seconds)
                assigned[column] = track.public_id

        for index in range(len(boxes)):
            if index in assigned or float(scores[index]) < self.birth_threshold:
                continue
            embedding = embeddings[index]
            self.tracks.append(Track(self.next_id, boxes[index], embedding, [embedding]))
            assigned[index] = self.next_id
            self.next_id += 1
        return assigned


def sequence_fps(sequence_dir: Path) -> float:
    config = configparser.ConfigParser()
    config.read(sequence_dir / "seqinfo.ini")
    return float(config["Sequence"]["frameRate"])


@torch.inference_mode()
def run_sequence(model, sequence, output, *, conf, iou, max_age_seconds, birth_threshold, device):
    sequence_dir = ROOT / "data" / "dancetrack" / sequence
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No frames found for {sequence}")
    tracker = MemoryTracker(
        model,
        sequence_fps(sequence_dir),
        max_age_seconds=max_age_seconds,
        birth_threshold=birth_threshold,
    )
    rows = []
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
            prediction, conf_thres=conf, iou_thres=iou, classes=[0], max_det=300
        )[0]
        if len(detections):
            boxes = detections[:, :4]
            scores = detections[:, 4]
            embeddings = model.embed_boxes(pyramid, [boxes])
            assigned = tracker.step(boxes, embeddings, scores)
            original_boxes = restore_boxes(boxes, original.shape[:2])
            for index, public_id in assigned.items():
                x1, y1, x2, y2 = original_boxes[index].tolist()
                rows.append([
                    frame_number, public_id, x1, y1, x2 - x1, y2 - y1,
                    float(scores[index]), -1, -1, -1,
                ])
        else:
            tracker.step(
                torch.empty((0, 4), device=device),
                torch.empty((0, 64), device=device),
                torch.empty(0, device=device),
            )
        compute_seconds += time.perf_counter() - started

    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{sequence}.txt").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in row) + "\n")
    return {
        "sequence": sequence,
        "frames": len(image_paths),
        "rows": len(rows),
        "compute_fps": len(image_paths) / compute_seconds,
        "pipeline_fps": len(image_paths) / (time.perf_counter() - total_started),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--head", default="runs/sequence-head/head.pt")
    parser.add_argument("--output", default="runs/sequence-tracker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-age-seconds", type=float, default=5.0)
    parser.add_argument("--birth-threshold", type=float, default=0.25)
    args = parser.parse_args()

    head_path = ROOT / args.head
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    detector_path = ROOT / checkpoint["detector"]
    actual_detector_sha = sha256(detector_path)
    if actual_detector_sha != checkpoint["detector_sha256"]:
        raise RuntimeError(
            f"detector hash mismatch: checkpoint expects {checkpoint['detector_sha256']}, "
            f"found {actual_detector_sha}"
        )
    device = torch.device(args.device)
    model = TemporalYOLO(detector_path, checkpoint["embedding_dim"]).to(device).eval()
    model.embedding.load_state_dict(checkpoint["embedding"])
    model.association.load_state_dict(checkpoint["association"])
    model.absence.load_state_dict(checkpoint["absence"])
    output = ROOT / args.output
    stats = [
        run_sequence(
            model,
            sequence,
            output,
            conf=args.conf,
            iou=args.iou,
            max_age_seconds=args.max_age_seconds,
            birth_threshold=args.birth_threshold,
            device=device,
        )
        for sequence in args.sequences
    ]
    metadata = {
        "system": "YOLO26n + causal velocity/appearance association + bounded prototype memory",
        "head": args.head,
        "head_sha256": sha256(head_path),
        "detector_sha256": actual_detector_sha,
        "torch": str(torch.__version__),
        "ultralytics": ultralytics.__version__,
        "training": {k: checkpoint[k] for k in ("steps", "lr", "data", "before", "after")},
        "detector_settings": {"classes": [0], "conf": args.conf, "iou": args.iou, "imgsz": 640},
        "memory": {
            "prototypes": 5,
            "max_age_seconds": args.max_age_seconds,
            "birth_threshold": args.birth_threshold,
        },
        "sequences": stats,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
