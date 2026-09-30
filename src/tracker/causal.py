from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTrack, GMC, KalmanFilterXYWH, build_encoder
from ultralytics.trackers.byte_tracker import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes
from ultralytics.trackers.utils import matching

from .association import choose_guarded_assignment, load_owner_scorer, owner_pair_features


class CausalPersonTracker:
    """Project-owned causal tracking policy with bounded track memory."""

    def __init__(
        self,
        args,
        owner_checkpoint: str | Path | dict,
        *,
        owner_alpha: float = 0.2,
        average_base_cost_budget: float = 0.00025,
        gmc_max_corners: int = 100,
    ):
        if gmc_max_corners < 5:
            raise ValueError("gmc_max_corners must be at least 5")
        self.args = args
        self.max_frames_lost = args.track_buffer
        self.gmc = GMC(method=args.gmc_method)
        if self.gmc.method == "sparseOptFlow":
            self.gmc.feature_params["maxCorners"] = int(gmc_max_corners)
        self.gmc_max_corners = int(gmc_max_corners)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = build_encoder(args.with_reid, args.model, getattr(args, "device", None))
        self.owner_scorer, self.owner_mean, self.owner_std, _ = load_owner_scorer(owner_checkpoint)
        self.owner_alpha = owner_alpha
        self.owner_average_base_cost_budget = average_base_cost_budget
        self.reset()

    def reset(self) -> None:
        self.tracked_stracks: list[BOTrack] = []
        self.lost_stracks: list[BOTrack] = []
        self.removed_stracks: list[BOTrack] = []
        self.removed_stracks_frame: list[BOTrack] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        self.gmc.reset_params()
        BOTrack.reset_id()
        self.owner_tiebreak_frames = 0
        self.owner_changed_frames = 0

    def _split_detections(self, results):
        scores = results.conf
        wh = results.xywh[:, 2:4]
        valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
        high = valid & (scores >= self.args.track_high_thresh)
        low = valid & (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)
        return results[high], results[low], high, low

    def _input_for(self, img, feats, mask):
        if feats is not None and len(feats):
            return feats[mask]
        if self.encoder is not None and getattr(self.args, "model", "auto") == "auto":
            return None
        return img

    def _init_track(self, results, aux) -> list[BOTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None and aux is not None:
            features = self.encoder(aux, bboxes)
            return [
                BOTrack(xywh, score, cls, feature)
                for xywh, score, cls, feature in zip(bboxes, results.conf, results.cls, features)
            ]
        return [BOTrack(xywh, score, cls) for xywh, score, cls in zip(bboxes, results.conf, results.cls)]

    def _get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def _apply_match(self, track, detection, activated, refind) -> None:
        if track.state == TrackState.Tracked:
            track.update(detection, self.frame_id)
            activated.append(track)
        else:
            track.re_activate(detection, self.frame_id, new_id=False)
            refind.append(track)

    def _apply_matches(self, matches, tracks, detections, activated, refind) -> None:
        for track_index, detection_index in matches:
            self._apply_match(
                tracks[int(track_index)],
                detections[int(detection_index)],
                activated,
                refind,
            )

    def _first_association(self, tracks, detections, activated, refind):
        base_costs = self._get_dists(tracks, detections)
        features = owner_pair_features(tracks, detections, self.frame_id)
        if features is None:
            matches, unmatched_tracks, unmatched_detections = matching.linear_assignment(
                base_costs, thresh=self.args.match_thresh
            )
        else:
            normalized = (features - self.owner_mean) / self.owner_std
            with torch.no_grad():
                owner_logits = self.owner_scorer(normalized).numpy()
            matches, unmatched_tracks, unmatched_detections, changed = choose_guarded_assignment(
                base_costs,
                owner_logits,
                match_threshold=self.args.match_thresh,
                alpha=self.owner_alpha,
                average_base_cost_budget=self.owner_average_base_cost_budget,
            )
            self.owner_tiebreak_frames += 1
            self.owner_changed_frames += int(changed)
        self._apply_matches(matches, tracks, detections, activated, refind)
        return unmatched_tracks, unmatched_detections

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Consume one frame and return active track rows."""
        self.frame_id += 1
        activated = []
        refind = []
        lost = []
        removed = []

        high_results, low_results, high_mask, low_mask = self._split_detections(results)
        detections = self._init_track(high_results, self._input_for(img, feats, high_mask))
        low_detections = self._init_track(low_results, self._input_for(img, feats, low_mask))
        for tracks, mask in ((detections, high_mask), (low_detections, low_mask)):
            for track, index in zip(tracks, np.flatnonzero(mask)):
                track.idx = index

        unconfirmed = []
        tracked = []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)

        pool = joint_stracks(tracked, self.lost_stracks)
        BOTrack.multi_predict(pool)
        if self.gmc.method is not None and img is not None:
            try:
                warp = self.gmc.apply(img, high_results.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            multi_gmc(pool, warp)
            multi_gmc(unconfirmed, warp)

        unmatched_tracks, unmatched_detections = self._first_association(
            pool, detections, activated, refind
        )

        remaining_tracked = [
            pool[index]
            for index in unmatched_tracks
            if pool[index].state == TrackState.Tracked
        ]
        if remaining_tracked and low_detections:
            second_costs = matching.iou_distance(remaining_tracked, low_detections)
            matches, still_unmatched, _ = matching.linear_assignment(second_costs, thresh=0.5)
            self._apply_matches(matches, remaining_tracked, low_detections, activated, refind)
        else:
            still_unmatched = list(range(len(remaining_tracked)))
        for index in still_unmatched:
            track = remaining_tracked[int(index)]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        remaining_detections = [detections[index] for index in unmatched_detections]
        if unconfirmed:
            unconfirmed_costs = self._get_dists(unconfirmed, remaining_detections)
            matches, unmatched_unconfirmed, unmatched_detection_indices = matching.linear_assignment(
                unconfirmed_costs, thresh=0.7
            )
            for track_index, detection_index in matches:
                track = unconfirmed[int(track_index)]
                track.update(remaining_detections[int(detection_index)], self.frame_id)
                activated.append(track)
            for track_index in unmatched_unconfirmed:
                track = unconfirmed[int(track_index)]
                track.mark_removed()
                removed.append(track)
        else:
            unmatched_detection_indices = list(range(len(remaining_detections)))

        for detection_index in unmatched_detection_indices:
            track = remaining_detections[int(detection_index)]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed.append(track)

        merge_track_pools(self, activated, refind, lost, removed)
        return np.asarray(
            [track.result for track in self.tracked_stracks if track.is_activated],
            dtype=np.float32,
        )
