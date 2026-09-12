"""Coverage-aware diagnostics. Official HOTA/IDF1 still come from TrackEval.

Version 2 keeps the old switch-free score for reading legacy results, but uses
covered_identity_segment_rate for acceptance. A missed segment cannot succeed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from .geometry import cxcywh_to_xyxy


def score_mot_rows(truth: np.ndarray, predictions: np.ndarray, frame_count: int) -> dict:
    """Score complete MOT-format rows [frame,id,x,y,w,h,...], including empty frames."""
    if frame_count <= 0:
        raise ValueError("A sequence must contain frames")
    for rows in (truth, predictions):
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise ValueError("MOT rows need frame, id, x, y, width, height")
        if not np.isfinite(rows[:, :6]).all():
            raise ValueError("Non-finite MOT input")
        if np.any(rows[:, :2] != np.floor(rows[:, :2])):
            raise ValueError("Frame and identity values must be integers")
        if np.any(rows[:, 0] < 1) or np.any(rows[:, 0] > frame_count):
            raise ValueError("MOT rows lie outside the evaluated sequence")
        if np.any(rows[:, 4:6] <= 0):
            raise ValueError("MOT boxes must have positive size")
    metric = TrackingAccumulator()
    for frame in range(1, frame_count + 1):
        selected = [rows[rows[:, 0] == frame] for rows in (truth, predictions)]
        boxes = []
        for rows in selected:
            values = torch.tensor(rows[:, 2:6], dtype=torch.float32)
            values[:, :2] += values[:, 2:] / 2
            boxes.append(values)
        metric.update(selected[0][:, 1], boxes[0], selected[1][:, 1], boxes[1])
    return metric.finalize()


@dataclass
class SegmentState:
    open: bool = False
    public_ids: set[int] = field(default_factory=set)
    bad: bool = False
    visible_frames: int = 0
    matched_frames: int = 0


class TrackingAccumulator:
    def __init__(self, iou_threshold=.5, minimum_segment_coverage=.8):
        self.iou_threshold = float(iou_threshold)
        self.minimum_segment_coverage = float(minimum_segment_coverage)
        if not 0 < self.iou_threshold <= 1 or not 0 < self.minimum_segment_coverage <= 1:
            raise ValueError("IoU and segment coverage thresholds must be in (0, 1]")
        self.tp = self.fp = self.fn = 0
        self.id_switches = 0
        self.public_id_reuses = 0
        self.visible_segments = 0
        self.strict_segments = 0
        self.last_public_by_gt: dict[int, int] = {}
        self.owner_by_public: dict[int, int] = {}
        self.segments: dict[int, SegmentState] = {}
        self.closed_segments: list[dict] = []
        self.conflicted_public_ids: set[int] = set()

    def _close(self, identity: int):
        state = self.segments.get(identity)
        if not state or not state.open:
            return
        self.strict_segments += int(not state.bad and len(state.public_ids) <= 1)
        self.closed_segments.append({
            "identity": identity, "public_ids": set(state.public_ids), "bad": state.bad,
            "visible_frames": state.visible_frames, "matched_frames": state.matched_frames,
        })
        state.open = False
        state.public_ids.clear()
        state.bad = False
        state.visible_frames = state.matched_frames = 0

    def update(self, gt_ids, gt_boxes: torch.Tensor, public_ids, public_boxes: torch.Tensor):
        gt_ids = [int(value) for value in gt_ids]
        public_ids = [int(value) for value in public_ids]
        if len(set(gt_ids)) != len(gt_ids) or len(set(public_ids)) != len(public_ids):
            raise ValueError("Each GT/public identity may appear at most once in a frame")
        gt_boxes = gt_boxes.float().reshape(-1, 4)
        public_boxes = public_boxes.float().reshape(-1, 4)
        if len(gt_boxes) != len(gt_ids) or len(public_boxes) != len(public_ids):
            raise ValueError("Box and identity counts differ")
        visible = set(gt_ids)
        for identity in list(self.segments):
            if identity not in visible:
                self._close(identity)
        for identity in gt_ids:
            state = self.segments.setdefault(identity, SegmentState())
            if not state.open:
                state.open = True
                self.visible_segments += 1
            state.visible_frames += 1

        matches = []
        if len(gt_ids) and len(public_ids):
            iou = box_iou(cxcywh_to_xyxy(gt_boxes), cxcywh_to_xyxy(public_boxes))
            # First maximize the number of valid matches, then overlap. Invalid
            # pairs must not steal assignments before the threshold is applied.
            weights = torch.where(iou >= self.iou_threshold,
                                  1 + iou / (min(iou.shape) + 1), 0)
            row, col = linear_sum_assignment(weights.cpu().numpy(), maximize=True)
            for g, p in zip(row, col):
                if float(iou[g, p]) >= self.iou_threshold:
                    matches.append((int(g), int(p)))
        matched_gt = {g for g, _ in matches}
        matched_pred = {p for _, p in matches}
        self.tp += len(matches)
        self.fn += len(gt_ids) - len(matches)
        self.fp += len(public_ids) - len(matches)

        for gt_index, pred_index in matches:
            identity = gt_ids[gt_index]
            public = public_ids[pred_index]
            state = self.segments[identity]
            state.matched_frames += 1
            state.public_ids.add(public)
            previous = self.last_public_by_gt.get(identity)
            if previous is not None and previous != public:
                self.id_switches += 1
                state.bad = True
            self.last_public_by_gt[identity] = public
            owner = self.owner_by_public.get(public)
            if owner is None:
                self.owner_by_public[public] = identity
            elif owner != identity:
                self.public_id_reuses += 1
                self.conflicted_public_ids.add(public)
                state.bad = True

    def finalize(self):
        for identity in list(self.segments):
            self._close(identity)
        successful = [row for row in self.closed_segments if (
            not row["bad"] and len(row["public_ids"]) == 1
            and not row["public_ids"].intersection(self.conflicted_public_ids)
            and row["matched_frames"] / row["visible_frames"] >= self.minimum_segment_coverage
        )]
        visible_frames = self.tp + self.fn
        return {
            "metric_version": 2,
            "iou_threshold": self.iou_threshold,
            "minimum_segment_coverage": self.minimum_segment_coverage,
            "covered_identity_segments": len(successful),
            "covered_identity_segment_rate": len(successful) / self.visible_segments if self.visible_segments else None,
            "covered_identity_visible_frames": sum(row["visible_frames"] for row in successful),
            "covered_identity_frame_rate": sum(row["visible_frames"] for row in successful) / visible_frames if visible_frames else None,
            "untracked_segments": sum(row["matched_frames"] == 0 for row in self.closed_segments),
            "legacy_metric_warning": "strict_visible_segment_survival counts untracked segments as switch-free; never use it for acceptance. Version 2 also fixes thresholded matching; historical scores need replay.",
            "true_positives": self.tp,
            "false_positives": self.fp,
            "missed_people": self.fn,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else None,
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else None,
            "identity_switches": self.id_switches,
            "public_id_reuses": self.public_id_reuses,
            "visible_segments": self.visible_segments,
            "strict_visible_segment_survival": self.strict_segments / self.visible_segments if self.visible_segments else None,
            "public_ids_used": len(self.owner_by_public),
        }
