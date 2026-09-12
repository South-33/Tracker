"""Small causal tracking metrics used before official TrackEval promotion."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from .geometry import cxcywh_to_xyxy


@dataclass
class SegmentState:
    open: bool = False
    public_ids: set[int] = field(default_factory=set)
    bad: bool = False


class TrackingAccumulator:
    def __init__(self, iou_threshold=.5):
        self.iou_threshold = float(iou_threshold)
        self.tp = self.fp = self.fn = 0
        self.id_switches = 0
        self.public_id_reuses = 0
        self.visible_segments = 0
        self.strict_segments = 0
        self.last_public_by_gt: dict[int, int] = {}
        self.owner_by_public: dict[int, int] = {}
        self.segments: dict[int, SegmentState] = {}

    def _close(self, identity: int):
        state = self.segments.get(identity)
        if not state or not state.open:
            return
        self.strict_segments += int(not state.bad and len(state.public_ids) <= 1)
        state.open = False
        state.public_ids.clear()
        state.bad = False

    def update(self, gt_ids, gt_boxes: torch.Tensor, public_ids, public_boxes: torch.Tensor):
        gt_ids = [int(value) for value in gt_ids]
        public_ids = [int(value) for value in public_ids]
        gt_boxes = gt_boxes.float().reshape(-1, 4)
        public_boxes = public_boxes.float().reshape(-1, 4)
        visible = set(gt_ids)
        for identity in list(self.segments):
            if identity not in visible:
                self._close(identity)
        for identity in gt_ids:
            state = self.segments.setdefault(identity, SegmentState())
            if not state.open:
                state.open = True
                self.visible_segments += 1

        matches = []
        if len(gt_ids) and len(public_ids):
            iou = box_iou(cxcywh_to_xyxy(gt_boxes), cxcywh_to_xyxy(public_boxes))
            row, col = linear_sum_assignment((-iou).cpu().numpy())
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
                state.bad = True

    def finalize(self):
        for identity in list(self.segments):
            self._close(identity)
        return {
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
