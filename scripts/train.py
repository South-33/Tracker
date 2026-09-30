"""Train causal YOLO memory association from 3-frame PersonPath22 sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch
from torch.optim import AdamW
import ultralytics
from ultralytics.utils.nms import non_max_suppression

from tracker.model import TemporalYOLO, causal_association_loss
from tracker.data import DanceTrackTriples, PersonPathTriples

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not len(a) or not len(b):
        return torch.empty((len(a), len(b)), device=a.device)
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (bottom_right - top_left).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-6)


def label_predictions(boxes, truth_boxes, truth_ids, min_iou=0.5):
    labels = torch.full((len(boxes),), -1, device=boxes.device, dtype=torch.long)
    if not len(boxes) or not len(truth_boxes):
        return labels
    iou = box_iou(boxes, truth_boxes)
    used_predictions, used_truth = set(), set()
    truth_ids = truth_ids.to(boxes.device)
    for index in torch.argsort(iou.flatten(), descending=True).tolist():
        prediction = index // len(truth_boxes)
        truth = index % len(truth_boxes)
        if float(iou[prediction, truth]) < min_iou:
            break
        if prediction in used_predictions or truth in used_truth:
            continue
        labels[prediction] = truth_ids[truth]
        used_predictions.add(prediction)
        used_truth.add(truth)
    return labels


def detector_frames(model, images, truth_boxes, truth_ids):
    """Use the exact runtime detector distribution during training."""
    with torch.no_grad():
        raw, pyramid = model.extract(images)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(
            prediction, conf_thres=0.1, iou_thres=0.7, classes=[0], max_det=300
        )
    boxes = [detection[:, :4] for detection in detections]
    labels = [
        label_predictions(boxes[i], truth_boxes[i], truth_ids[i]) for i in range(len(boxes))
    ]
    embeddings = model.embed_boxes(pyramid, boxes)
    embeddings = list(torch.split(embeddings, [len(box) for box in boxes]))
    return boxes, labels, embeddings


def memory_velocity(previous_boxes, previous_ids, latest_boxes, latest_ids, history_seconds):
    """Velocity state attached to latest detector-backed tracks."""
    velocity = torch.zeros_like(latest_boxes)
    known = torch.zeros(len(latest_boxes), device=latest_boxes.device)
    previous = {
        int(identity): previous_boxes[i]
        for i, identity in enumerate(previous_ids.tolist())
        if int(identity) >= 0
    }
    for i, identity in enumerate(latest_ids.tolist()):
        if int(identity) >= 0 and int(identity) in previous:
            velocity[i] = (latest_boxes[i] - previous[int(identity)]) / max(history_seconds, 1e-3)
            known[i] = 1.0
    return velocity, known


def sample_loss(model, sample, device):
    (
        image_a, gt_boxes_a, gt_ids_a,
        image_b, gt_boxes_b, gt_ids_b,
        image_c, gt_boxes_c, gt_ids_c,
        history_seconds, gap_seconds,
    ) = sample
    images = torch.stack((image_a, image_b, image_c)).to(device)
    truth_boxes = [gt_boxes_a.to(device), gt_boxes_b.to(device), gt_boxes_c.to(device)]
    truth_ids = [gt_ids_a.to(device), gt_ids_b.to(device), gt_ids_c.to(device)]
    boxes, labels, embeddings = detector_frames(model, images, truth_boxes, truth_ids)
    velocity, known = memory_velocity(
        boxes[0], labels[0], boxes[1], labels[1], history_seconds
    )
    return causal_association_loss(
        model,
        embeddings[1], boxes[1], velocity, known, labels[1],
        embeddings[2], boxes[2], labels[2], gap_seconds,
    )


@torch.no_grad()
def validate(model, dataset, device, samples=100):
    model.eval()
    losses, accuracies = [], []
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    for index in indices[: min(samples, len(indices))]:
        try:
            loss, accuracy = sample_loss(model, dataset[index], device)
        except ValueError:
            continue
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    if not losses:
        raise RuntimeError("validation produced no detector-backed sequence samples")
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="runs/sequence-head")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    personpath_videos = json.loads((ROOT / "data" / "personpath22" / "starter.json").read_text())["videos"]
    dancetrack_train = [
        "dancetrack0001", "dancetrack0002", "dancetrack0006", "dancetrack0008",
        "dancetrack0015", "dancetrack0082", "dancetrack0083",
    ]
    dancetrack_validation = ["dancetrack0012"]
    personpath = PersonPathTriples(ROOT / "data" / "personpath22", personpath_videos)
    dancetrack = DanceTrackTriples(ROOT / "data" / "dancetrack", dancetrack_train)
    train_sets = (personpath, dancetrack)
    validation = DanceTrackTriples(ROOT / "data" / "dancetrack", dancetrack_validation)
    print(
        f"train triples: personpath={len(personpath)} dancetrack={len(dancetrack)} "
        f"validation={len(validation)}"
    )

    device = torch.device(args.device)
    weights = ROOT / "weights" / "yolo26n.pt"
    model = TemporalYOLO(weights).to(device)
    optimizer = AdamW(
        list(model.embedding.parameters())
        + list(model.association.parameters())
        + list(model.absence.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    before_loss, before_accuracy = validate(model, validation, device)
    print(f"before: loss={before_loss:.4f} choice_acc={before_accuracy:.3f}")
    rng = random.Random(0)
    for step in range(1, args.steps + 1):
        dataset = train_sets[(step - 1) % len(train_sets)]
        while True:
            index = rng.randrange(len(dataset))
            try:
                loss, accuracy = sample_loss(model, dataset[index], device)
            except ValueError:
                continue
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={loss.detach().item():.4f} choice_acc={accuracy.item():.3f}")

    after_loss, after_accuracy = validate(model, validation, device)
    print(f"after: loss={after_loss:.4f} choice_acc={after_accuracy:.3f}")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embedding": model.embedding.state_dict(),
        "association": model.association.state_dict(),
        "absence": model.absence.state_dict(),
        "embedding_dim": 64,
        "detector": "weights/yolo26n.pt",
        "detector_sha256": sha256(weights),
        "repo_commit": git_commit(),
        "torch": str(torch.__version__),
        "ultralytics": ultralytics.__version__,
        "seed": 0,
        "device": args.device,
        "detector_settings": {"classes": [0], "conf": 0.1, "iou": 0.7, "imgsz": 640},
        "data": {
            "personpath_train": personpath_videos,
            "dancetrack_train": dancetrack_train,
            "dancetrack_validation": dancetrack_validation,
            "sampling": "alternate datasets; uniform random triple within each dataset",
        },
        "steps": args.steps,
        "lr": args.lr,
        "before": {"loss": before_loss, "choice_accuracy": before_accuracy},
        "after": {"loss": after_loss, "choice_accuracy": after_accuracy},
    }, output / "head.pt")


if __name__ == "__main__":
    main()
