"""Contiguous human-video clips for temporal tracking training."""
from __future__ import annotations

import configparser
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


def letterbox(image, boxes, size=640):
    """Resize/pad an image and xywh boxes into a square, returning xyxy boxes."""
    if image is None:
        raise ValueError("cannot letterbox an empty image")
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = torch.full(
        (3, size, size),
        114 / 255,
        dtype=torch.float32,
    )
    rgb = (
        torch.from_numpy(resized[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .float()
        / 255
    )
    canvas[:, top : top + new_height, left : left + new_width] = rgb

    output = torch.as_tensor(boxes, dtype=torch.float32).clone()
    if len(output):
        output[:, 2] += output[:, 0]
        output[:, 3] += output[:, 1]
        output[:, [0, 2]] = output[:, [0, 2]] * scale + left
        output[:, [1, 3]] = output[:, [1, 3]] * scale + top
    return canvas, output


def xyxy_to_normalized_cxcywh(
    boxes: torch.Tensor,
    size: int,
) -> torch.Tensor:
    if not len(boxes):
        return boxes.new_empty((0, 4))
    result = boxes.clone()
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / size
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / size
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / size
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / size
    return result.clamp(0, 1)


class MOTClipDataset(Dataset):
    """Contiguous MOTChallenge-style clips with boxes and stable track IDs."""

    def __init__(
        self,
        root: str | Path,
        sequences: list[str] | tuple[str, ...],
        *,
        clip_length: int = 8,
        clip_stride: int = 4,
        frame_step: int = 1,
        size: int = 640,
    ):
        if clip_length < 2:
            raise ValueError("clip_length must be at least 2")
        if clip_stride < 1 or frame_step < 1:
            raise ValueError("clip_stride and frame_step must be positive")
        self.root = Path(root)
        self.sequences = tuple(sequences)
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.frame_step = int(frame_step)
        self.size = int(size)
        self.targets: dict[str, dict[int, list[tuple[int, list[float]]]]] = {}
        self.clips: list[tuple[str, tuple[int, ...]]] = []

        for sequence in self.sequences:
            sequence_dir = self.root / sequence
            info = configparser.ConfigParser()
            info.read(sequence_dir / "seqinfo.ini")
            if "Sequence" not in info:
                raise FileNotFoundError(sequence_dir / "seqinfo.ini")
            length = int(info["Sequence"]["seqLength"])
            images = sequence_dir / "img1"
            gt_path = sequence_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                raise FileNotFoundError(gt_path)

            expected = set(range(1, length + 1))
            available = {
                int(path.stem)
                for path in images.glob("*.jpg")
            }
            missing = sorted(expected - available)
            if missing:
                raise RuntimeError(
                    f"{sequence} is incomplete: missing {len(missing)} frames"
                )

            frame_targets: dict[int, list[tuple[int, list[float]]]] = {}
            for line in gt_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = line.split(",")
                frame = int(float(row[0]))
                identity = int(float(row[1]))
                x, y, width, height = map(float, row[2:6])
                if width <= 1 or height <= 1:
                    continue
                frame_targets.setdefault(frame, []).append(
                    (identity, [x, y, width, height])
                )
            self.targets[sequence] = frame_targets

            span = (self.clip_length - 1) * self.frame_step
            for start in range(1, length - span + 1, self.clip_stride):
                frames = tuple(
                    start + index * self.frame_step
                    for index in range(self.clip_length)
                )
                if any(frame_targets.get(frame) for frame in frames):
                    self.clips.append((sequence, frames))

        if not self.clips:
            raise ValueError("no training clips were found")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        sequence, frame_numbers = self.clips[index]
        frames = []
        boxes = []
        identities = []
        for frame_number in frame_numbers:
            path = (
                self.root
                / sequence
                / "img1"
                / f"{frame_number:08d}.jpg"
            )
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            entries = self.targets[sequence].get(frame_number, [])
            ids = torch.tensor(
                [identity for identity, _ in entries],
                dtype=torch.long,
            )
            image, xyxy = letterbox(
                image,
                [box for _, box in entries],
                self.size,
            )
            frames.append(image)
            boxes.append(
                xyxy_to_normalized_cxcywh(xyxy, self.size)
            )
            identities.append(ids)

        return {
            "sequence": sequence,
            "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
            "frames": torch.stack(frames),
            "boxes": boxes,
            "identities": identities,
        }


def collate_clips(samples: list[dict]) -> dict:
    """Collate clips while keeping variable-count targets as nested lists."""
    return {
        "sequence": [sample["sequence"] for sample in samples],
        "frame_numbers": torch.stack(
            [sample["frame_numbers"] for sample in samples]
        ),
        "frames": torch.stack([sample["frames"] for sample in samples]),
        "boxes": [sample["boxes"] for sample in samples],
        "identities": [sample["identities"] for sample in samples],
    }
