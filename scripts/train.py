"""Train a 64D identity embedding directly from YOLO26n detector features."""
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

from tracker.data import DanceTrackPairs, PersonPathPairs
from tracker.model import TrackingYOLO, identity_retrieval_loss

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    )


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
    """Embed the same post-NMS person detections used at tracking time."""
    with torch.no_grad():
        raw, pyramid = model.extract(images)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(
            prediction,
            conf_thres=0.1,
            iou_thres=0.7,
            classes=[0],
            max_det=300,
        )
    boxes = [detection[:, :4] for detection in detections]
    labels = [
        label_predictions(boxes[i], truth_boxes[i], truth_ids[i])
        for i in range(len(boxes))
    ]
    embeddings = model.embed_boxes(pyramid, boxes)
    embeddings = list(torch.split(embeddings, [len(box) for box in boxes]))
    return labels, embeddings


def sample_loss(model, sample, device, temperature):
    (
        image_a,
        gt_boxes_a,
        gt_ids_a,
        image_b,
        gt_boxes_b,
        gt_ids_b,
        _gap_seconds,
    ) = sample
    images = torch.stack((image_a, image_b)).to(device)
    truth_boxes = [gt_boxes_a.to(device), gt_boxes_b.to(device)]
    truth_ids = [gt_ids_a.to(device), gt_ids_b.to(device)]
    labels, embeddings = detector_frames(model, images, truth_boxes, truth_ids)
    return identity_retrieval_loss(
        embeddings[0],
        labels[0],
        embeddings[1],
        labels[1],
        temperature=temperature,
    )


@torch.no_grad()
def validate(model, dataset, device, temperature, samples):
    model.eval()
    losses, reports = [], []
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    for index in indices[: min(samples, len(indices))]:
        try:
            loss, report = sample_loss(
                model,
                dataset[index],
                device,
                temperature,
            )
        except ValueError:
            continue
        losses.append(loss.item())
        reports.append(report)
    if not losses:
        raise RuntimeError("validation produced no detector-backed identity pairs")
    return {
        "loss": sum(losses) / len(losses),
        **{
            key: sum(report[key] for report in reports) / len(reports)
            for key in reports[0]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--output", default="runs/identity-head-v2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    personpath_videos = json.loads(
        (ROOT / "data" / "personpath22" / "starter.json").read_text()
    )["videos"]
    dancetrack_train = [
        "dancetrack0001",
        "dancetrack0002",
        "dancetrack0006",
        "dancetrack0008",
        "dancetrack0015",
    ]
    dancetrack_validation = ["dancetrack0012"]
    personpath = PersonPathPairs(
        ROOT / "data" / "personpath22",
        personpath_videos,
    )
    dancetrack = DanceTrackPairs(
        ROOT / "data" / "dancetrack",
        dancetrack_train,
    )
    train_sets = (personpath, dancetrack)
    validation = DanceTrackPairs(
        ROOT / "data" / "dancetrack",
        dancetrack_validation,
    )
    print(
        f"train pairs: personpath={len(personpath)} dancetrack={len(dancetrack)} "
        f"validation={len(validation)}"
    )

    device = torch.device(args.device)
    weights = ROOT / "weights" / "yolo26n.pt"
    model = TrackingYOLO(weights).to(device)
    optimizer = AdamW(
        model.embedding.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    before = validate(
        model,
        validation,
        device,
        args.temperature,
        args.validation_samples,
    )
    print(
        "before: "
        f"loss={before['loss']:.4f} "
        f"top1={before['top1_accuracy']:.3f} "
        f"pos={before['positive_cosine']:.3f} "
        f"hard_neg={before['hard_negative_cosine']:.3f}"
    )
    model.train()
    rng = random.Random(0)
    for step in range(1, args.steps + 1):
        dataset = train_sets[(step - 1) % len(train_sets)]
        while True:
            index = rng.randrange(len(dataset))
            try:
                loss, report = sample_loss(
                    model,
                    dataset[index],
                    device,
                    args.temperature,
                )
            except ValueError:
                continue
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(
                f"step={step} loss={loss.detach().item():.4f} "
                f"top1={report['top1_accuracy']:.3f} "
                f"pos={report['positive_cosine']:.3f} "
                f"hard_neg={report['hard_negative_cosine']:.3f}"
            )

    after = validate(
        model,
        validation,
        device,
        args.temperature,
        args.validation_samples,
    )
    print(
        "after: "
        f"loss={after['loss']:.4f} "
        f"top1={after['top1_accuracy']:.3f} "
        f"pos={after['positive_cosine']:.3f} "
        f"hard_neg={after['hard_negative_cosine']:.3f}"
    )

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": model.embedding.state_dict(),
            "embedding_dim": model.embedding_dim,
            "detector": "weights/yolo26n.pt",
            "detector_sha256": sha256(weights),
            "repo_commit": git_commit(),
            "repo_dirty": git_dirty(),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
            "seed": 0,
            "device": args.device,
            "objective": "symmetric identity retrieval over post-NMS detector boxes",
            "temperature": args.temperature,
            "validation_samples": args.validation_samples,
            "detector_settings": {
                "classes": [0],
                "conf": 0.1,
                "iou": 0.7,
                "imgsz": 640,
            },
            "data": {
                "personpath_train": personpath_videos,
                "dancetrack_train": dancetrack_train,
                "dancetrack_validation": dancetrack_validation,
                "sampling": "alternate datasets; uniform random pair within each dataset",
            },
            "steps": args.steps,
            "lr": args.lr,
            "before": before,
            "after": after,
        },
        output / "head.pt",
    )


if __name__ == "__main__":
    main()
