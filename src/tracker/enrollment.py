"""Bounded private opening enrollment used before public identities are frozen."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import box_iou

from .geometry import cxcywh_to_xyxy
from .proposals import DetectionPrediction


@dataclass
class EnrollmentTrack:
    private_id: int
    box: torch.Tensor
    feature_sum: torch.Tensor
    hits: int
    age: float
    score_sum: float

    @property
    def canonical(self) -> torch.Tensor:
        return self.feature_sum / max(1, self.hits)

    @property
    def mean_score(self) -> float:
        return self.score_sum / max(1, self.hits)


def advance_enrollment(
    tracks: list[EnrollmentTrack],
    next_private_id: int,
    detection: DetectionPrediction,
    dt: float,
    *,
    score_threshold: float = .4,
    minimum_iou: float = .1,
    maximum_age: float = .5,
) -> tuple[list[EnrollmentTrack], int]:
    """Advance private opening tracklets without creating public IDs."""
    for track in tracks:
        track.age += float(dt)
    scores = detection.person_logits[0].float().sigmoid()
    candidates = torch.nonzero(
        detection.valid[0].bool() & (scores >= float(score_threshold)), as_tuple=False,
    ).flatten().tolist()
    matched_queries: set[int] = set()
    if tracks and candidates:
        left = torch.stack([track.box for track in tracks])
        index = torch.tensor(candidates, device=detection.boxes.device)
        right = detection.boxes[0, index].float()
        overlap = box_iou(cxcywh_to_xyxy(left), cxcywh_to_xyxy(right))
        track_choice = overlap.argmax(dim=1)
        proposal_choice = overlap.argmax(dim=0)
        for track_index, local_query in enumerate(track_choice.tolist()):
            if int(proposal_choice[local_query]) != track_index:
                continue
            if float(overlap[track_index, local_query]) < float(minimum_iou):
                continue
            query = int(candidates[local_query])
            track = tracks[track_index]
            track.box = detection.boxes[0, query].detach().float()
            track.feature_sum = track.feature_sum + detection.features[0, query].detach().float()
            track.hits += 1
            track.age = 0.
            track.score_sum += float(scores[query])
            matched_queries.add(query)
    for query in candidates:
        if query in matched_queries:
            continue
        tracks.append(EnrollmentTrack(
            next_private_id,
            detection.boxes[0, query].detach().float(),
            detection.features[0, query].detach().float(),
            1,
            0.,
            float(scores[query]),
        ))
        next_private_id += 1
    return [track for track in tracks if track.age <= float(maximum_age)], next_private_id


def freeze_enrollment(
    tracks: list[EnrollmentTrack],
    *,
    required_hits: int = 3,
    duplicate_iou: float = .5,
    maximum_slots: int = 64,
) -> list[EnrollmentTrack]:
    """Freeze one bounded roster, then close births permanently."""
    mature = [track for track in tracks if track.hits >= int(required_hits)]
    mature.sort(key=lambda track: (track.hits, track.mean_score), reverse=True)
    selected: list[EnrollmentTrack] = []
    for track in mature:
        if len(selected) >= int(maximum_slots):
            break
        if selected:
            overlap = box_iou(
                cxcywh_to_xyxy(track.box.reshape(1, 4)),
                cxcywh_to_xyxy(torch.stack([item.box for item in selected])),
            )[0]
            if bool((overlap >= float(duplicate_iou)).any()):
                continue
        selected.append(track)
    return selected
