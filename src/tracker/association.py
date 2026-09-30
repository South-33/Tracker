from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching


class OwnerContinuityScorer(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_owner_scorer(path: str | Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scorer = OwnerContinuityScorer(checkpoint.get("input_dim", 11))
    state = checkpoint["pair"]
    if any(key.startswith("net.") for key in state):
        scorer.load_state_dict(state)
    else:
        scorer.net.load_state_dict(state)
    scorer.eval()
    return scorer, checkpoint["mean"], checkpoint["std"], checkpoint


def _pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 1) * np.maximum(a[:, 3] - a[:, 1], 1)
    area_b = np.maximum(b[:, 2] - b[:, 0], 1) * np.maximum(b[:, 3] - b[:, 1], 1)
    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection,
        1e-6,
    )


def owner_pair_features(tracks, detections, frame_id: int) -> torch.Tensor | None:
    if (
        len(tracks) < 2
        or not detections
        or any(track.smooth_feat is None for track in tracks)
        or any(detection.curr_feat is None for detection in detections)
    ):
        return None

    track_boxes = np.asarray([track.xyxy for track in tracks], dtype=np.float32)
    detection_boxes = np.asarray([detection.xyxy for detection in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    detection_features = np.asarray([detection.curr_feat for detection in detections], dtype=np.float32)

    cosine = detection_features @ track_features.T
    track_width = np.maximum(track_boxes[:, 2] - track_boxes[:, 0], 1)
    track_height = np.maximum(track_boxes[:, 3] - track_boxes[:, 1], 1)
    detection_width = np.maximum(detection_boxes[:, 2] - detection_boxes[:, 0], 1)
    detection_height = np.maximum(detection_boxes[:, 3] - detection_boxes[:, 1], 1)
    track_center_x = (track_boxes[:, 0] + track_boxes[:, 2]) / 2
    track_center_y = (track_boxes[:, 1] + track_boxes[:, 3]) / 2
    detection_center_x = (detection_boxes[:, 0] + detection_boxes[:, 2]) / 2
    detection_center_y = (detection_boxes[:, 1] + detection_boxes[:, 3]) / 2

    dx = (detection_center_x[:, None] - track_center_x[None, :]) / track_width[None, :]
    dy = (detection_center_y[:, None] - track_center_y[None, :]) / track_height[None, :]
    log_width = np.log(detection_width[:, None] / track_width[None, :])
    log_height = np.log(detection_height[:, None] / track_height[None, :])
    iou = _pair_iou(detection_boxes, track_boxes)
    age = np.asarray(
        [min(max(frame_id - track.end_frame, 0) / 30.0, 1.0) for track in tracks],
        dtype=np.float32,
    )[None, :]
    age = np.broadcast_to(age, cosine.shape)
    detection_score = np.asarray(
        [detection.score for detection in detections],
        dtype=np.float32,
    )[:, None]
    detection_score = np.broadcast_to(detection_score, cosine.shape)

    crowding = np.zeros(len(detections), dtype=np.float32)
    if len(detections) > 1:
        detection_iou = _pair_iou(detection_boxes, detection_boxes)
        np.fill_diagonal(detection_iou, 0)
        crowding = detection_iou.max(1)
    crowding = np.broadcast_to(crowding[:, None], cosine.shape)

    maturity = np.asarray(
        [min(getattr(track, "tracklet_len", 0) / 30.0, 2.0) for track in tracks],
        dtype=np.float32,
    )[None, :]
    maturity = np.broadcast_to(maturity, cosine.shape)
    active = np.asarray(
        [1.0 if track.state == TrackState.Tracked else 0.0 for track in tracks],
        dtype=np.float32,
    )[None, :]
    active = np.broadcast_to(active, cosine.shape)

    features = np.stack(
        (
            cosine,
            dx,
            dy,
            log_width,
            log_height,
            iou,
            age,
            detection_score,
            crowding,
            maturity,
            active,
        ),
        axis=-1,
    )
    return torch.from_numpy(features)


def choose_guarded_assignment(
    base_costs: np.ndarray,
    owner_logits: np.ndarray,
    *,
    match_threshold: float,
    alpha: float,
    average_base_cost_budget: float,
):
    base_matches, base_unmatched_tracks, base_unmatched_detections = matching.linear_assignment(
        base_costs,
        thresh=match_threshold,
    )
    if not len(base_matches) or owner_logits.size == 0:
        return base_matches, base_unmatched_tracks, base_unmatched_detections, False

    mean = owner_logits.mean(axis=1, keepdims=True)
    std = owner_logits.std(axis=1, keepdims=True) + 1e-6
    residual = ((owner_logits - mean) / std).T
    learned_costs = np.where(
        base_costs <= match_threshold,
        base_costs - alpha * residual,
        1.0,
    )
    learned_matches, learned_unmatched_tracks, learned_unmatched_detections = (
        matching.linear_assignment(learned_costs, thresh=match_threshold)
    )
    if len(learned_matches) != len(base_matches):
        return base_matches, base_unmatched_tracks, base_unmatched_detections, False

    base_total = sum(
        float(base_costs[int(track), int(detection)])
        for track, detection in base_matches
    )
    learned_total = sum(
        float(base_costs[int(track), int(detection)])
        for track, detection in learned_matches
    )
    average_increase = (learned_total - base_total) / len(base_matches)
    if average_increase > average_base_cost_budget:
        return base_matches, base_unmatched_tracks, base_unmatched_detections, False
    changed = not np.array_equal(np.asarray(base_matches), np.asarray(learned_matches))
    return (
        learned_matches,
        learned_unmatched_tracks,
        learned_unmatched_detections,
        changed,
    )


class GuardedOwnerBOTSORT(BOTSORT):
    def __init__(
        self,
        args,
        owner_checkpoint: str | Path,
        *,
        alpha: float = 0.2,
        average_base_cost_budget: float = 0.00025,
    ):
        super().__init__(args)
        self.owner_scorer, self.owner_mean, self.owner_std, _ = load_owner_scorer(
            owner_checkpoint
        )
        self.owner_alpha = alpha
        self.owner_average_base_cost_budget = average_base_cost_budget
        self.owner_tiebreak_frames = 0
        self.owner_changed_frames = 0

    def _first_association(self, strack_pool, detections, activated, refind):
        base_costs = BOTSORT.get_dists(self, strack_pool, detections)
        features = owner_pair_features(strack_pool, detections, self.frame_id)
        if features is None:
            matches, unmatched_tracks, unmatched_detections = matching.linear_assignment(
                base_costs,
                thresh=self.args.match_thresh,
            )
        else:
            normalized = (features - self.owner_mean) / self.owner_std
            with torch.no_grad():
                owner_logits = self.owner_scorer(normalized).numpy()
            (
                matches,
                unmatched_tracks,
                unmatched_detections,
                changed,
            ) = choose_guarded_assignment(
                base_costs,
                owner_logits,
                match_threshold=self.args.match_thresh,
                alpha=self.owner_alpha,
                average_base_cost_budget=self.owner_average_base_cost_budget,
            )
            self.owner_tiebreak_frames += 1
            self.owner_changed_frames += int(changed)

        self._apply_matches(matches, strack_pool, detections, activated, refind)
        return unmatched_tracks, unmatched_detections
