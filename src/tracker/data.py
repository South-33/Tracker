"""Minimal sequence datasets for causal person tracking."""
from __future__ import annotations

import configparser
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


def letterbox(image, boxes, size=640):
    """Resize/pad an image and xywh boxes into a YOLO-style square."""
    if image is None:
        raise ValueError("cannot letterbox an empty image")
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = torch.full((3, size, size), 114 / 255, dtype=torch.float32)
    rgb = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
    canvas[:, top : top + new_height, left : left + new_width] = rgb

    output = torch.as_tensor(boxes, dtype=torch.float32).clone()
    if len(output):
        output[:, 2] += output[:, 0]
        output[:, 3] += output[:, 1]
        output[:, [0, 2]] = output[:, [0, 2]] * scale + left
        output[:, [1, 3]] = output[:, [1, 3]] * scale + top
    return canvas, output


def restore_boxes(boxes: torch.Tensor, original_shape, size=640) -> torch.Tensor:
    """Map xyxy boxes from the letterboxed square back to the source image."""
    height, width = original_shape
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    result = boxes.clone()
    result[:, [0, 2]] = (result[:, [0, 2]] - left) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - top) / scale
    result[:, [0, 2]].clamp_(0, width)
    result[:, [1, 3]].clamp_(0, height)
    return result


class PersonPathPairs(Dataset):
    """Causal frame pairs from PersonPath22 with shared person identities."""

    def __init__(self, root: str | Path, videos: list[str], size=640, max_gap_seconds=5.0):
        self.root = Path(root)
        self.size = size
        self.video_root = self.root / "raw_data" / "starter"
        self.annotation_root = self.root / "annotation" / "anno_visible_2022"
        self.cache_root = self.root / "frames"
        self.frames = {}
        self.fps = {}
        self.pairs = []

        for video in videos:
            annotation = json.loads((self.annotation_root / f"{video}.json").read_text())
            frame_map = {}
            for entity in annotation["entities"]:
                if not entity.get("labels", {}).get("person"):
                    continue
                frame = int(entity["blob"]["frame_idx"])
                box = [float(x) for x in entity["bb"]]
                if box[2] <= 1 or box[3] <= 1:
                    continue
                frame_map.setdefault(frame, []).append((int(entity["id"]), box))
            self.frames[video] = frame_map
            self.fps[video] = float(annotation["metadata"]["fps"])
            self._ensure_frame_cache(video, frame_map)

            max_gap = round(max_gap_seconds * self.fps[video])
            frame_numbers = sorted(frame_map)
            for i, latest in enumerate(frame_numbers[:-1]):
                latest_ids = {identity for identity, _ in frame_map[latest]}
                for target in frame_numbers[i + 1 :]:
                    if target - latest > max_gap:
                        break
                    if latest_ids & {identity for identity, _ in frame_map[target]}:
                        self.pairs.append((video, latest, target))

    def _ensure_frame_cache(self, video: str, frame_map):
        target = self.cache_root / Path(video).stem
        target.mkdir(parents=True, exist_ok=True)
        missing = {frame for frame in frame_map if not (target / f"{frame:06d}.jpg").exists()}
        if not missing:
            return
        capture = cv2.VideoCapture(str(self.video_root / video))
        if not capture.isOpened():
            raise FileNotFoundError(self.video_root / video)
        frame = 0
        while missing:
            ok, image = capture.read()
            if not ok:
                break
            if frame in missing:
                cv2.imwrite(str(target / f"{frame:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                missing.remove(frame)
            frame += 1
        capture.release()
        if missing:
            raise RuntimeError(f"could not decode {len(missing)} annotated frames from {video}")

    def __len__(self):
        return len(self.pairs)

    def _frame(self, video, frame):
        image = cv2.imread(str(self.cache_root / Path(video).stem / f"{frame:06d}.jpg"))
        if image is None:
            raise FileNotFoundError(self.cache_root / Path(video).stem / f"{frame:06d}.jpg")
        entries = self.frames[video][frame]
        ids = torch.tensor([identity for identity, _ in entries], dtype=torch.long)
        image, boxes = letterbox(image, [box for _, box in entries], self.size)
        return image, boxes, ids

    def __getitem__(self, index):
        video, first, second = self.pairs[index]
        a = self._frame(video, first)
        b = self._frame(video, second)
        return (*a, *b, (second - first) / self.fps[video])


class DanceTrackPairs(Dataset):
    """Dense DanceTrack frame pairs with fixed future gaps."""

    def __init__(self, root: str | Path, sequences: list[str], size=640, gaps=(1, 2, 4, 8, 16, 32, 64, 100)):
        self.root = Path(root)
        self.size = size
        self.frames = {}
        self.fps = {}
        self.pairs = []

        for sequence in sequences:
            sequence_dir = self.root / sequence
            config = configparser.ConfigParser()
            config.read(sequence_dir / "seqinfo.ini")
            fps = float(config["Sequence"]["frameRate"])
            sequence_length = int(config["Sequence"]["seqLength"])
            self.fps[sequence] = fps

            image_frames = {
                int(path.stem) for path in (sequence_dir / "img1").glob("*.jpg")
            }
            missing_images = sorted(set(range(1, sequence_length + 1)) - image_frames)
            if missing_images:
                raise RuntimeError(
                    f"{sequence} is incomplete: missing {len(missing_images)} of "
                    f"{sequence_length} image frames (first missing: {missing_images[0]})"
                )

            frame_map = {}
            for line in (sequence_dir / "gt" / "gt.txt").read_text().splitlines():
                row = line.split(",")
                frame = int(row[0])
                identity = int(row[1])
                box = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                if box[2] > 1 and box[3] > 1:
                    frame_map.setdefault(frame, []).append((identity, box))
            self.frames[sequence] = frame_map

            for first in sorted(frame_map):
                first_ids = {identity for identity, _ in frame_map[first]}
                for gap in gaps:
                    second = first + gap
                    if second not in frame_map:
                        continue
                    if first_ids & {identity for identity, _ in frame_map[second]}:
                        self.pairs.append((sequence, first, second))

    def __len__(self):
        return len(self.pairs)

    def _frame(self, sequence, frame):
        path = self.root / sequence / "img1" / f"{frame:08d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        entries = self.frames[sequence][frame]
        ids = torch.tensor([identity for identity, _ in entries], dtype=torch.long)
        image, boxes = letterbox(image, [box for _, box in entries], self.size)
        return image, boxes, ids

    def __getitem__(self, index):
        sequence, first, second = self.pairs[index]
        a = self._frame(sequence, first)
        b = self._frame(sequence, second)
        fps = self.fps[sequence]
        return (*a, *b, (second - first) / fps)
