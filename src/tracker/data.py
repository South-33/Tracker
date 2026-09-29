"""Tiny MOT-format data helpers."""
from pathlib import Path

import numpy as np


def load_mot(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists() or not path.read_text().strip():
        return np.empty((0, 10), dtype=np.float32)
    return np.loadtxt(path, delimiter=",", ndmin=2).astype(np.float32)


def index_annotations(rows: np.ndarray, frames):
    """Group MOT rows by frame without repeatedly scanning the full table."""
    truth = {int(number): (np.empty(0, np.int64), np.empty((0, 4), np.float32)) for number in frames}
    if not len(rows):
        return truth
    if np.any(rows[1:, 0] < rows[:-1, 0]):
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
    numbers, starts = np.unique(rows[:, 0], return_index=True)
    ends = np.append(starts[1:], len(rows))
    identities = rows[:, 1].astype(np.int64)
    boxes = rows[:, 2:6].astype(np.float32)
    for number, start, end in zip(numbers, starts, ends):
        if int(number) in truth:
            truth[int(number)] = (identities[start:end], boxes[start:end])
    return truth
