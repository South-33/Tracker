"""Resident supervision is fixed by sequence identity; only births use Hungarian matching."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou

from .geometry import cxcywh_to_xyxy


def assign(prediction, ids, boxes, slot_to_gt, slots):
    gt_index = {int(identity): index for index, identity in enumerate(ids)}
    assignments = [(slot, gt_index[identity]) for slot, identity in slot_to_gt.items() if identity in gt_index]
    remembered = set(slot_to_gt.values())
    newcomers = [index for index, identity in enumerate(ids) if int(identity) not in remembered]
    births = []
    if newcomers:
        predicted = prediction.boxes[0, slots:].detach().float()
        target = boxes[newcomers].float()
        probability = prediction.person_logits[0, slots:].detach().float().sigmoid()
        cost = 5 * torch.cdist(predicted, target, p=1) - 2 * generalized_box_iou(cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target)) - 2 * probability[:, None]
        row, col = linear_sum_assignment(cost.cpu().numpy())
        births = [(int(r) + slots, newcomers[int(c)]) for r, c in zip(row, col)]
        assignments.extend(births)
    return assignments, births


def supervised_loss(prediction, assignments, targets, memory, slots):
    logits = prediction.person_logits[0].float()
    labels = torch.zeros_like(logits)
    quality = torch.zeros_like(logits)
    valid_queries = torch.cat((memory.valid[0, :, 0], torch.ones_like(logits[slots:])))
    zero = prediction.boxes.sum() * 0
    bbox_loss, giou_loss = zero, zero
    if assignments:
        queries, indices = zip(*assignments)
        queries = torch.as_tensor(queries, device=logits.device)
        indices = torch.as_tensor(indices, device=logits.device)
        selected = prediction.boxes[0, queries].float()
        truth = targets[indices].float()
        labels[queries] = 1
        iou = box_iou(cxcywh_to_xyxy(selected), cxcywh_to_xyxy(truth)).diag().detach().clamp(0, 1)
        quality[queries] = iou
        bbox_loss = F.l1_loss(selected, truth, reduction="sum") / len(assignments)
        giou_loss = (1 - generalized_box_iou(cxcywh_to_xyxy(selected), cxcywh_to_xyxy(truth)).diag()).mean()
    normalizer = max(1, len(assignments))
    weight = 0.75 * logits.sigmoid().detach().pow(2) * (1 - labels) + quality * labels
    classification = (F.binary_cross_entropy_with_logits(logits, quality, reduction="none") * weight * valid_queries).sum() / normalizer
    resident_valid = memory.valid[0, :, 0]
    continuity = (F.binary_cross_entropy_with_logits(prediction.continuity_logits[0, :slots].float(), labels[:slots], reduction="none") * resident_valid).sum() / resident_valid.sum().clamp(min=1)
    gate = (F.binary_cross_entropy_with_logits(prediction.quality_logits[0].float(), quality, reduction="none") * labels).sum() / normalizer
    loss = classification + 5 * bbox_loss + 2 * giou_loss + continuity + 0.25 * gate
    parts = {"vfl": classification, "l1": bbox_loss, "giou": giou_loss, "continuity": continuity, "quality": gate}
    return loss, parts
