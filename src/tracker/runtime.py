"""Single-frame causal runtime for the current person-tracking candidate."""
from __future__ import annotations

import time

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.utils.nms import non_max_suppression

from tracker.data import letterbox, restore_boxes


class CausalTrackerRuntime:
    """Compose perception and bounded online association behind one step call.

    The association object is intentionally injected. The guarded candidate uses
    CausalPersonTracker, so callers depend only on this causal frame interface
    rather than on a framework-owned tracking policy.
    """

    def __init__(
        self,
        model,
        association,
        *,
        device: torch.device,
        with_reid: bool,
        feature_mode: str = "trained",
        min_detections_for_reid: int = 10,
    ):
        if min_detections_for_reid < 1:
            raise ValueError("min_detections_for_reid must be at least 1")
        if feature_mode not in {"trained", "random", "raw"}:
            raise ValueError(f"unsupported feature mode: {feature_mode}")
        self.model = model
        self.association = association
        self.device = device
        self.with_reid = bool(with_reid)
        self.feature_mode = feature_mode
        self.min_detections_for_reid = int(min_detections_for_reid)
        self.frames = 0
        self.reid_frames = 0
        self.reid_detections = 0
        self.compute_seconds = 0.0

    @torch.inference_mode()
    def step(self, original: np.ndarray) -> np.ndarray:
        """Process one BGR frame and return [xyxy,id,score,class,det_index]."""
        if original is None:
            raise ValueError("frame must not be None")
        self.frames += 1
        image, _ = letterbox(original, [], 640)
        image = image.unsqueeze(0).to(self.device)

        started = time.perf_counter()
        raw, pyramid = self.model.extract(image)
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
            use_reid = (
                self.with_reid
                and len(detections) >= self.min_detections_for_reid
            )
            if use_reid and self.feature_mode == "raw":
                features = torch.nn.functional.normalize(
                    self.model.pool_boxes(pyramid, [boxes]),
                    dim=1,
                )
            else:
                features = (
                    self.model.embed_boxes(pyramid, [boxes])
                    if use_reid
                    else None
                )
            if features is not None:
                self.reid_frames += 1
                self.reid_detections += len(features)
            original_boxes = restore_boxes(boxes, original.shape[:2])
            results = Boxes(
                torch.cat((original_boxes, detections[:, 4:6]), 1).cpu(),
                original.shape[:2],
            ).numpy()
            tracks = self.association.update(
                results,
                original,
                feats=(
                    features.float().cpu().numpy()
                    if features is not None
                    else None
                ),
            )
        else:
            results = Boxes(
                torch.empty((0, 6)),
                original.shape[:2],
            ).numpy()
            tracks = self.association.update(
                results,
                original,
                feats=None,
            )

        self.compute_seconds += time.perf_counter() - started
        return tracks

    def stats(self) -> dict:
        return {
            "frames": self.frames,
            "reid_frames": self.reid_frames,
            "reid_detections": self.reid_detections,
            "compute_seconds": self.compute_seconds,
            "owner_tiebreak_frames": getattr(
                self.association,
                "owner_tiebreak_frames",
                0,
            ),
            "owner_changed_frames": getattr(
                self.association,
                "owner_changed_frames",
                0,
            ),
        }
