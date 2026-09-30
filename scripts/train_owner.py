"""Train the tiny owner-continuity scorer on real sequential BoT-SORT states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.nms import non_max_suppression

from tracker.association import OwnerContinuityScorer, owner_pair_features
from tracker.data import letterbox, restore_boxes
from tracker.model import TrackingYOLO
from tracker.splits import CALIBRATION, TRAIN, assert_train_only
from train import label_predictions
from track import tracker_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = list(TRAIN)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_truth(sequence: str):
    truth = {}
    path = ROOT / "data" / "dancetrack" / sequence / "gt" / "gt.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.split(",")
        frame = int(row[0])
        identity = int(row[1])
        x, y, width, height = map(float, row[2:6])
        if width > 1 and height > 1:
            truth.setdefault(frame, []).append((identity, [x, y, width, height]))
    return truth


class OwnerQueryCollector(BOTSORT):
    def __init__(self, args):
        super().__init__(args)
        self.detection_labels = np.empty(0, dtype=np.int64)
        self.owner = {}
        self.queries = []

    def get_dists(self, tracks, detections):
        distances = BOTSORT.get_dists(self, tracks, detections)
        features = owner_pair_features(tracks, detections, self.frame_id)
        if features is None:
            return distances

        track_ids = np.asarray([int(track.track_id) for track in tracks], dtype=np.int64)
        for detection_index, detection in enumerate(detections):
            full_index = int(detection.idx)
            if not 0 <= full_index < len(self.detection_labels):
                continue
            identity = int(self.detection_labels[full_index])
            owner = self.owner.get(identity)
            if identity < 0 or owner is None:
                continue
            target = np.flatnonzero(track_ids == owner)
            if len(target):
                self.queries.append((features[detection_index].clone(), int(target[0])))
        return distances


@torch.inference_mode()
def collect_sequence(
    model,
    sequence,
    *,
    device,
    min_detections_for_reid,
    max_frames=None,
):
    truth = load_truth(sequence)
    paths = sorted((ROOT / "data" / "dancetrack" / sequence / "img1").glob("*.jpg"))
    if max_frames is not None:
        paths = paths[:max_frames]
    config = tracker_config(
        device,
        0.8,
        with_reid=True,
        new_track_threshold=0.45,
    )
    tracker = OwnerQueryCollector(IterableSimpleNamespace(**config))

    for frame_number, path in enumerate(paths, 1):
        original = cv2.imread(str(path))
        if original is None:
            raise FileNotFoundError(path)
        entries = truth.get(frame_number, [])
        truth_ids = torch.tensor([identity for identity, _ in entries], dtype=torch.long)
        image, truth_boxes = letterbox(
            original,
            [box for _, box in entries],
            640,
        )
        image = image.unsqueeze(0).to(device)
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
            tracker.detection_labels = (
                label_predictions(
                    boxes,
                    truth_boxes.to(device),
                    truth_ids.to(device),
                )
                .cpu()
                .numpy()
            )
            use_reid = len(detections) >= min_detections_for_reid
            features = model.embed_boxes(pyramid, [boxes]) if use_reid else None
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
            tracker.detection_labels = np.empty(0, dtype=np.int64)
            tracks = tracker.update(
                Boxes(torch.empty((0, 6)), original.shape[:2]).numpy(),
                original,
                feats=None,
            )

        for row in tracks:
            public_id = int(row[4])
            detection_index = int(row[7])
            if 0 <= detection_index < len(tracker.detection_labels):
                identity = int(tracker.detection_labels[detection_index])
                if identity >= 0:
                    tracker.owner[identity] = public_id

    print(f"{sequence}: {len(tracker.queries)} owner queries")
    return tracker.queries


def accuracy(scorer, queries, mean, std):
    correct = 0
    with torch.no_grad():
        for features, target in queries:
            logits = scorer((features - mean) / std)
            correct += int(int(logits.argmax()) == target)
    return correct / max(len(queries), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="runs/identity-head-v2/head.pt")
    parser.add_argument("--output", default="runs/owner-head-v1/head.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-detections-for-reid", type=int, default=10)
    parser.add_argument("--train-sequences", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--validation-sequence", default=CALIBRATION[0])
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to each sequence.",
    )
    args = parser.parse_args()
    assert_train_only(args.train_sequences)
    if args.validation_sequence not in CALIBRATION:
        raise ValueError(
            "--validation-sequence must be the dedicated CALIBRATION sequence"
        )

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    head_path = ROOT / args.head
    identity = torch.load(head_path, map_location="cpu", weights_only=True)
    detector_path = ROOT / identity["detector"]
    model = TrackingYOLO(detector_path, identity["embedding_dim"]).to(device).eval()
    model.embedding.load_state_dict(identity["embedding"])

    train_queries = []
    for sequence in args.train_sequences:
        train_queries.extend(
            collect_sequence(
                model,
                sequence,
                device=device,
                min_detections_for_reid=args.min_detections_for_reid,
                max_frames=args.max_frames,
            )
        )
    validation_queries = collect_sequence(
        model,
        args.validation_sequence,
        device=device,
        min_detections_for_reid=args.min_detections_for_reid,
        max_frames=args.max_frames,
    )
    if not train_queries or not validation_queries:
        raise RuntimeError("owner-continuity collection produced no usable queries")

    all_candidates = torch.cat([features for features, _ in train_queries], 0)
    mean = all_candidates.mean(0)
    std = all_candidates.std(0).clamp_min(1e-5)
    scorer = OwnerContinuityScorer()
    optimizer = AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-4)

    before = accuracy(scorer, validation_queries, mean, std)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_queries)))
        random.shuffle(order)
        total_loss = 0.0
        scorer.train()
        for index in order:
            features, target = train_queries[index]
            logits = scorer((features - mean) / std)
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target]),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        scorer.eval()
        print(
            f"epoch {epoch}: loss={total_loss / len(train_queries):.6f} "
            f"val_owner_accuracy={accuracy(scorer, validation_queries, mean, std):.6f}"
        )

    after = accuracy(scorer, validation_queries, mean, std)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "pair": scorer.state_dict(),
        "mean": mean,
        "std": std,
        "input_dim": 11,
        "features": [
            "cosine",
            "dx",
            "dy",
            "log_width",
            "log_height",
            "iou",
            "age",
            "detection_score",
            "crowding",
            "tracklet_length",
            "active",
        ],
        "objective": "preserve the previous public owner on real sequential BoT-SORT states",
        "epochs": args.epochs,
        "lr": args.lr,
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
        "identity_head": args.head,
        "identity_head_sha256": sha256(head_path),
        "train_sequences": args.train_sequences,
        "validation_sequence": args.validation_sequence,
        "train_queries": len(train_queries),
        "validation_queries": len(validation_queries),
        "before": before,
        "validation": {"owner_accuracy": after},
    }
    torch.save(checkpoint, output)
    print(json.dumps({key: value for key, value in checkpoint.items() if key != "pair" and not torch.is_tensor(value)}, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
