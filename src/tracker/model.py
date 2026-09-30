"""YOLO26n plus a tiny learned causal association head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO


class TemporalYOLO(nn.Module):
    """Frozen YOLO26n perception probe with small trainable tracking heads."""

    def __init__(self, weights: str | Path, embedding_dim: int = 64):
        super().__init__()
        self.detector = YOLO(str(weights)).model
        detect = self.detector.model[-1]
        channels = [branch[0].conv.in_channels for branch in detect.cv2]
        self.strides = [int(x) for x in detect.stride.tolist()]

        self.embedding = nn.Sequential(
            nn.Linear(sum(channels), 128),
            nn.SiLU(),
            nn.Linear(128, embedding_dim),
        )
        self.association = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 10, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.absence = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._pyramid = None
        self._hook = detect.register_forward_pre_hook(self._capture_pyramid)
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self.detector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # This experiment deliberately isolates the tracking heads. A parent
        # train() call must not change detector BatchNorm/runtime behavior.
        self.detector.eval()
        return self

    def _capture_pyramid(self, _module, args):
        self._pyramid = tuple(args[0])

    def extract(self, images: torch.Tensor):
        self._pyramid = None
        predictions = self.detector(images)
        if self._pyramid is None:
            raise RuntimeError("YOLO feature hook did not run")
        return predictions, self._pyramid

    def embed_boxes(self, pyramid, boxes_per_image: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feature, stride in zip(pyramid, self.strides):
            x = roi_align(feature, boxes_per_image, output_size=1, spatial_scale=1.0 / stride, aligned=True)
            pooled.append(x.flatten(1))
        return F.normalize(self.embedding(torch.cat(pooled, dim=1)), dim=1)

    def association_scores(
        self,
        memory_embeddings: torch.Tensor,
        memory_boxes: torch.Tensor,
        velocities: torch.Tensor,
        velocity_known: torch.Tensor,
        current_embeddings: torch.Tensor,
        current_boxes: torch.Tensor,
        gap_seconds: float | torch.Tensor,
    ) -> torch.Tensor:
        a = memory_embeddings[:, None, :].expand(-1, len(current_embeddings), -1)
        b = current_embeddings[None, :, :].expand(len(memory_embeddings), -1, -1)
        geometry = pair_geometry(
            memory_boxes, velocities, velocity_known, current_boxes, gap_seconds
        )
        features = torch.cat((a, b, (a - b).abs(), a * b, geometry), dim=-1)
        return self.association(features).squeeze(-1)

    def absence_scores(self, embeddings: torch.Tensor, gap_seconds: float | torch.Tensor) -> torch.Tensor:
        gap = _row_values(gap_seconds, len(embeddings), embeddings)
        return self.absence(torch.cat((embeddings, (gap / 5.0)[:, None]), dim=1)).squeeze(-1)


def _row_values(value, rows: int, like: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if x.ndim == 0:
        return x.expand(rows)
    if x.numel() != rows:
        raise ValueError(f"expected {rows} row values, got {x.numel()}")
    return x.reshape(rows)


def _iou(a: torch.Tensor, b: torch.Tensor, aw, ah, bw, bh):
    top_left = torch.maximum(a[..., :2], b[..., :2])
    bottom_right = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (bottom_right - top_left).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    return intersection / (aw * ah + bw * bh - intersection).clamp_min(1)


def pair_geometry(memory_boxes, velocities, velocity_known, current_boxes, gap_seconds):
    """Relative box + constant-velocity residual features for every candidate pair."""
    rows = len(memory_boxes)
    gap = _row_values(gap_seconds, rows, memory_boxes)
    known = _row_values(velocity_known, rows, memory_boxes)
    a = memory_boxes[:, None, :]
    b = current_boxes[None, :, :]
    aw = (a[..., 2] - a[..., 0]).clamp_min(1)
    ah = (a[..., 3] - a[..., 1]).clamp_min(1)
    bw = (b[..., 2] - b[..., 0]).clamp_min(1)
    bh = (b[..., 3] - b[..., 1]).clamp_min(1)
    acx = (a[..., 0] + a[..., 2]) / 2
    acy = (a[..., 1] + a[..., 3]) / 2
    bcx = (b[..., 0] + b[..., 2]) / 2
    bcy = (b[..., 1] + b[..., 3]) / 2
    iou = _iou(a, b, aw, ah, bw, bh)

    predicted = a + velocities[:, None, :] * gap[:, None, None]
    pw = (predicted[..., 2] - predicted[..., 0]).clamp_min(1)
    ph = (predicted[..., 3] - predicted[..., 1]).clamp_min(1)
    pcx = (predicted[..., 0] + predicted[..., 2]) / 2
    pcy = (predicted[..., 1] + predicted[..., 3]) / 2
    predicted_iou = _iou(predicted, b, pw, ph, bw, bh)

    return torch.stack((
        (bcx - acx) / aw,
        (bcy - acy) / ah,
        torch.log(bw / aw),
        torch.log(bh / ah),
        iou,
        gap[:, None].expand_as(iou) / 5.0,
        (bcx - pcx) / aw,
        (bcy - pcy) / ah,
        predicted_iou,
        known[:, None].expand_as(iou),
    ), dim=-1)


def causal_association_loss(
    model: TemporalYOLO,
    memory_embeddings: torch.Tensor,
    memory_boxes: torch.Tensor,
    velocities: torch.Tensor,
    velocity_known: torch.Tensor,
    memory_ids: torch.Tensor,
    current_embeddings: torch.Tensor,
    current_boxes: torch.Tensor,
    current_ids: torch.Tensor,
    gap_seconds: float,
):
    """For each valid remembered track, choose one current detection or ABSENT."""
    valid_rows = [i for i, identity in enumerate(memory_ids.tolist()) if int(identity) >= 0]
    if not valid_rows:
        raise ValueError("no detector-backed memory tracks in sample")
    current_lookup = {
        int(identity): i for i, identity in enumerate(current_ids.tolist()) if int(identity) >= 0
    }
    scores = model.association_scores(
        memory_embeddings,
        memory_boxes,
        velocities,
        velocity_known,
        current_embeddings,
        current_boxes,
        gap_seconds,
    )
    targets = torch.tensor(
        [current_lookup.get(int(memory_ids[i]), len(current_ids)) for i in valid_rows],
        device=scores.device,
    )
    logits = torch.cat((
        scores[valid_rows],
        model.absence_scores(memory_embeddings[valid_rows], gap_seconds)[:, None],
    ), dim=1)
    loss = F.cross_entropy(logits, targets)
    with torch.no_grad():
        accuracy = (logits.argmax(1) == targets).float().mean()
    return loss, accuracy
