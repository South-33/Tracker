"""YOLO26n detector features plus a tiny person-identity embedding head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO


class TrackingYOLO(nn.Module):
    """Frozen YOLO26n perception with a trainable 64D tracking embedding."""

    def __init__(self, weights: str | Path, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.detector = YOLO(str(weights)).model
        detect = self.detector.model[-1]
        channels = [branch[0].conv.in_channels for branch in detect.cv2]
        self.strides = [int(x) for x in detect.stride.tolist()]
        self.embedding = nn.Sequential(
            nn.Linear(sum(channels), 128),
            nn.SiLU(),
            nn.Linear(128, embedding_dim),
        )

        self._pyramid = None
        self._hook = detect.register_forward_pre_hook(self._capture_pyramid)
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self.detector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The current probe isolates identity representation. Detector weights
        # and BatchNorm/runtime behavior stay frozen until this signal is real.
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
        count = sum(len(boxes) for boxes in boxes_per_image)
        if count == 0:
            return pyramid[0].new_empty((0, self.embedding_dim))
        pooled = []
        for feature, stride in zip(pyramid, self.strides):
            x = roi_align(
                feature,
                boxes_per_image,
                output_size=1,
                spatial_scale=1.0 / stride,
                aligned=True,
            )
            pooled.append(x.flatten(1))
        return F.normalize(self.embedding(torch.cat(pooled, dim=1)), dim=1)


def _retrieval_direction(
    query_embeddings: torch.Tensor,
    query_ids: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_ids: torch.Tensor,
    temperature: float,
):
    candidate_lookup = {
        int(identity): index
        for index, identity in enumerate(candidate_ids.tolist())
        if int(identity) >= 0
    }
    rows, targets = [], []
    for index, identity in enumerate(query_ids.tolist()):
        identity = int(identity)
        if identity >= 0 and identity in candidate_lookup:
            rows.append(index)
            targets.append(candidate_lookup[identity])
    if not rows:
        return None

    similarity = query_embeddings[rows] @ candidate_embeddings.T
    target = torch.tensor(targets, device=similarity.device, dtype=torch.long)
    index = torch.arange(len(rows), device=similarity.device)
    positive = similarity[index, target]
    logits = similarity / temperature
    loss = F.cross_entropy(logits, target)

    row_ids = query_ids[rows]
    negative_mask = (
        (candidate_ids >= 0)[None, :]
        & (candidate_ids[None, :] != row_ids[:, None])
    )
    hard_negative = similarity.masked_fill(~negative_mask, -1.0).max(1).values
    has_negative = negative_mask.any(1)
    with torch.no_grad():
        correct = (logits.argmax(1) == target).sum()
    return {
        "loss": loss,
        "matches": len(rows),
        "hard_negatives": int(has_negative.sum()),
        "correct": correct,
        "positive_cosine": positive.sum(),
        "hard_negative_cosine": hard_negative[has_negative].sum(),
    }


def identity_retrieval_loss(
    first_embeddings: torch.Tensor,
    first_ids: torch.Tensor,
    second_embeddings: torch.Tensor,
    second_ids: torch.Tensor,
    temperature: float = 0.1,
):
    """Symmetric identity retrieval over detector-backed boxes."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    directions = [
        _retrieval_direction(
            first_embeddings,
            first_ids,
            second_embeddings,
            second_ids,
            temperature,
        ),
        _retrieval_direction(
            second_embeddings,
            second_ids,
            first_embeddings,
            first_ids,
            temperature,
        ),
    ]
    directions = [result for result in directions if result is not None]
    if not directions:
        raise ValueError("sample has no detector-backed identity match across frames")

    matches = sum(result["matches"] for result in directions)
    hard_negatives = sum(result["hard_negatives"] for result in directions)
    loss = sum(
        result["loss"] * result["matches"] for result in directions
    ) / matches
    if hard_negatives:
        hard_negative_cosine = float(
            sum(
                result["hard_negative_cosine"].detach()
                for result in directions
            )
            / hard_negatives
        )
    else:
        hard_negative_cosine = -1.0
    stats = {
        "top1_accuracy": (
            sum(int(result["correct"]) for result in directions) / matches
        ),
        "positive_cosine": float(
            sum(
                result["positive_cosine"].detach()
                for result in directions
            )
            / matches
        ),
        "hard_negative_cosine": hard_negative_cosine,
    }
    return loss, stats
