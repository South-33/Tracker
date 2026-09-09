"""DanceTrack sequence access with explicit frame availability and source timestamps."""
import json
from pathlib import Path
import cv2
import numpy as np

from .detector import ROOT


class Sequence:
    def __init__(self, record):
        self.record = record
        self.name = record["sequence"]
        self.directory = ROOT / "data" / "dancetrack" / self.name
        self.fps = record["fps"]
        self.frames = record["available_frames"]
        rows = np.loadtxt(self.directory / "gt" / "gt.txt", delimiter=",", ndmin=2)
        self.truth = {}
        for number in self.frames:
            selected = rows[rows[:, 0] == number]
            self.truth[number] = (selected[:, 1].astype(np.int64), selected[:, 2:6].astype(np.float32))

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
