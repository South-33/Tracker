"""DanceTrack sequence access with explicit frame availability and source timestamps."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np

from .detector import ROOT


def index_annotations(rows, frames):
    """Group once instead of filtering the complete table for every frame."""
    truth = {number: (np.empty(0, np.int64), np.empty((0, 4), np.float32)) for number in frames}
    if not len(rows):
        return truth
    if np.any(rows[1:, 0] < rows[:-1, 0]):
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
    numbers, starts = np.unique(rows[:, 0], return_index=True)
    ends = np.append(starts[1:], len(rows))
    identities = rows[:, 1].astype(np.int64)
    boxes = rows[:, 2:6].astype(np.float32)
    for number, start, end in zip(numbers, starts, ends):
        if number in truth:
            truth[int(number)] = (identities[start:end], boxes[start:end])
    return truth


class Sequence:
    def __init__(self, record):
        self.record = record
        self.name = record["sequence"]
        self.directory = ROOT / "data" / "dancetrack" / self.name
        self.fps = record["fps"]
        self.frames = record["available_frames"]
        annotations = self.directory / "gt" / "gt.txt"
        if hashlib.sha256(annotations.read_bytes()).hexdigest() != record["gt_sha256"]:
            raise ValueError(f"Ground truth changed since the split was frozen: {self.name}")
        rows = np.loadtxt(annotations, delimiter=",", ndmin=2)
        self.truth = index_annotations(rows, self.frames)

    def read(self, number):
        if number not in self.truth:
            raise ValueError(f"Frame {number} unavailable in frozen manifest: {self.name}")
        image = cv2.imread(str(self.directory / "img1" / f"{number:08d}.jpg"))
        if image is None:
            raise FileNotFoundError(f"Missing image {self.name}:{number}")
        identities, boxes = self.truth[number]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), identities.copy(), boxes.copy()


def load_sequences(manifest, split):
    manifest = Path(manifest)
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    sequences = [Sequence(record) for record in records if record["split"] == split]
    if not sequences:
        raise ValueError(f"No {split} sequences in {manifest}")
    return sequences
