"""Reciprocal local continuity and its current-frame safety features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from .geometry import cxcywh_to_xyxy
from .proposals import DetectionPrediction
from .state_split import IdentityAuthority


LOCAL_FEATURE_DIM = 16


class LocalState(NamedTuple):
    trusted: torch.Tensor
    age: torch.Tensor

    def detached(self):
        return LocalState(self.trusted.detach(), self.age.detach())


@dataclass
class LocalLinkBatch:
    slots: torch.Tensor
    queries: torch.Tensor
    features: torch.Tensor
    link_iou: torch.Tensor

    def __len__(self) -> int:
        return int(self.slots.numel())


def empty_local_state(batch: int, slots: int, *, device, dtype) -> LocalState:
    return LocalState(
        torch.full((batch, slots, 4), .5, device=device, dtype=dtype),
        torch.zeros((batch, slots), device=device, dtype=dtype),
    )


def seed_local_state(state: LocalState, slot: int, box: torch.Tensor) -> LocalState:
    trusted = state.trusted.clone()
    age = state.age.clone()
    trusted[:, slot] = box.to(trusted).reshape(1, 4)
    age[:, slot] = 0
    return LocalState(trusted, age)


def _best_margin(matrix: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 1:
        if matrix.shape[1] <= 1:
            return torch.ones(len(rows), device=matrix.device)
        competing = matrix.clone()
        competing[rows, cols] = float('-inf')
        return matrix[rows, cols] - competing.max(dim=1).values[rows]
    if matrix.shape[0] <= 1:
        return torch.ones(len(rows), device=matrix.device)
    competing = matrix.clone()
    competing[rows, cols] = float('-inf')
    return matrix[rows, cols] - competing.max(dim=0).values[cols]


def _empty_links(device) -> LocalLinkBatch:
    return LocalLinkBatch(
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty((0, LOCAL_FEATURE_DIM), dtype=torch.float32, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
    )


def _materialize_links(
    authority: IdentityAuthority,
    state: LocalState,
    detection: DetectionPrediction,
    slots: torch.Tensor,
    queries: torch.Tensor,
    track_boxes: torch.Tensor,
    proposal_boxes: torch.Tensor,
    scores: torch.Tensor,
    overlap: torch.Tensor,
    local_track: torch.Tensor,
    local_query: torch.Tensor,
) -> LocalLinkBatch:
    if local_track.numel() == 0:
        return _empty_links(detection.boxes.device)

    selected_slots = slots[local_track]
    selected_queries = queries[local_query]
    selected_iou = overlap[local_track, local_query]
    candidate_boxes = proposal_boxes[local_query]
    trusted_boxes = track_boxes[local_track]

    proposal_features = F.normalize(detection.features[0, queries].float(), dim=-1)
    canonicals = F.normalize(authority.canonical[0, slots].float(), dim=-1)
    cosine = canonicals @ proposal_features.T
    own_cosine = cosine[local_track, local_query]
    if len(slots) > 1:
        same_query = cosine[:, local_query]
        mask = torch.arange(len(slots), device=detection.boxes.device).reshape(-1, 1) != local_track.reshape(1, -1)
        other_cosine = same_query.masked_fill(~mask, float('-inf')).max(dim=0).values
        other_iou = overlap[:, local_query].masked_fill(~mask, float('-inf')).max(dim=0).values
        track_overlap = box_iou(cxcywh_to_xyxy(track_boxes), cxcywh_to_xyxy(track_boxes))
        track_overlap.fill_diagonal_(float('-inf'))
        track_crowding = track_overlap.max(dim=1).values[local_track].nan_to_num(0., neginf=0.)
    else:
        other_cosine = torch.full_like(own_cosine, -1.)
        other_iou = torch.zeros_like(selected_iou)
        track_crowding = torch.zeros_like(selected_iou)

    if len(queries) > 1:
        proposal_overlap = box_iou(cxcywh_to_xyxy(proposal_boxes), cxcywh_to_xyxy(proposal_boxes))
        proposal_overlap.fill_diagonal_(float('-inf'))
        proposal_crowding = proposal_overlap.max(dim=1).values[local_query].nan_to_num(0., neginf=0.)
    else:
        proposal_crowding = torch.zeros_like(selected_iou)

    appearance_choice = cosine.argmax(dim=1)
    appearance_agrees = (appearance_choice[local_track] == local_query).float()
    appearance_margin = _best_margin(cosine, local_track, local_query, axis=1)
    track_margin = _best_margin(overlap, local_track, local_query, axis=1)
    proposal_margin = _best_margin(overlap, local_track, local_query, axis=0)
    trusted_wh = trusted_boxes[:, 2:].clamp(min=1e-4)
    delta = (candidate_boxes[:, :2] - trusted_boxes[:, :2]).abs() / trusted_wh
    scale = torch.log(candidate_boxes[:, 2:].clamp(min=1e-4) / trusted_wh).abs()

    features = torch.stack((
        scores[selected_queries], selected_iou, track_margin, proposal_margin,
        own_cosine, own_cosine - other_cosine, appearance_margin, appearance_agrees,
        proposal_crowding, other_iou, track_crowding,
        (state.age[0, selected_slots].float() / 2.).clamp(max=1.),
        (delta[:, 0] / 2.).clamp(max=1.), (delta[:, 1] / 2.).clamp(max=1.),
        (scale[:, 0] / 2.).clamp(max=1.), (scale[:, 1] / 2.).clamp(max=1.),
    ), dim=1)
    return LocalLinkBatch(selected_slots, selected_queries, features, selected_iou)


def build_local_links(
    authority: IdentityAuthority,
    state: LocalState,
    detection: DetectionPrediction,
    *,
    proposal_score: float = .1,
    minimum_iou: float = .1,
) -> LocalLinkBatch:
    """Build at most one reciprocal best-IoU current observation per public ID."""
    if authority.canonical.shape[0] != 1:
        raise ValueError('local continuity currently supports batch size 1')
    device = detection.boxes.device
    slots = torch.nonzero(authority.valid[0, :, 0].bool(), as_tuple=False).flatten()
    scores = detection.person_logits[0].float().sigmoid()
    proposal_mask = detection.valid[0].bool() & (scores >= float(proposal_score))
    queries = torch.nonzero(proposal_mask, as_tuple=False).flatten()
    if slots.numel() == 0 or queries.numel() == 0:
        return _empty_links(device)

    track_boxes = state.trusted[0, slots].float()
    proposal_boxes = detection.boxes[0, queries].float()
    overlap = box_iou(cxcywh_to_xyxy(track_boxes), cxcywh_to_xyxy(proposal_boxes))
    track_choice = overlap.argmax(dim=1)
    proposal_choice = overlap.argmax(dim=0)
    local_track = torch.arange(len(slots), device=device)
    chosen_iou = overlap[local_track, track_choice]
    reciprocal = (proposal_choice[track_choice] == local_track) & (chosen_iou >= float(minimum_iou))
    local_track = local_track[reciprocal]
    local_query = track_choice[reciprocal]
    return _materialize_links(
        authority, state, detection, slots, queries, track_boxes, proposal_boxes,
        scores, overlap, local_track, local_query,
    )


def advance_local_state(
    state: LocalState,
    commits: dict[int, int],
    detection: DetectionPrediction,
    dt: float,
    valid: torch.Tensor,
) -> LocalState:
    trusted = state.trusted.clone()
    age = state.age.clone()
    for slot in torch.nonzero(valid[0], as_tuple=False).flatten().tolist():
        query = commits.get(int(slot))
        if query is None:
            age[0, slot] += float(dt)
            continue
        trusted[0, slot] = detection.boxes[0, query].detach().to(trusted)
        age[0, slot] = 0
    return LocalState(trusted, age)
