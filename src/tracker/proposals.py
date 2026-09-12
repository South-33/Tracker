"""Frozen RT-DETR proposal stream shared by active causal association mechanisms."""
from __future__ import annotations
from typing import NamedTuple
import torch


class DetectionPrediction(NamedTuple):
    boxes: torch.Tensor
    person_logits: torch.Tensor
    features: torch.Tensor
    valid: torch.Tensor


def decode_proposals(base, pyramid) -> DetectionPrediction:
    decoder = base.detector.decoder
    image_features, shapes = decoder._get_encoder_input(pyramid)
    discovery, references, _, _ = decoder._get_decoder_input(image_features, shapes)
    result = decoder.decoder(
        discovery, references, image_features, shapes, decoder.dec_bbox_head,
        decoder.dec_score_head, decoder.query_pos_head, decoder.pre_bbox_head,
        decoder.integral, decoder.up, decoder.reg_scale, return_features=True,
    )
    scores = result[1][-1][..., 0]
    return DetectionPrediction(
        result[0][-1], scores, result[-1], torch.ones_like(scores, dtype=torch.bool),
    )


def top_candidates(detection: DetectionPrediction, count: int):
    count = min(int(count), detection.person_logits.shape[1])
    indices = torch.topk(detection.person_logits.float().sigmoid(), count, dim=1).indices
    feature_index = indices.unsqueeze(-1).expand(-1, -1, detection.features.shape[-1])
    box_index = indices.unsqueeze(-1).expand(-1, -1, 4)
    return DetectionPrediction(
        torch.gather(detection.boxes, 1, box_index),
        torch.gather(detection.person_logits, 1, indices),
        torch.gather(detection.features, 1, feature_index),
        torch.gather(detection.valid, 1, indices),
    ), indices
