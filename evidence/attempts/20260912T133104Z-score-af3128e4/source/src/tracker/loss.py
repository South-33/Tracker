"""Training-only detector-to-GT assignment.

Hungarian assignment is legal here because it creates supervision only. It is not
part of the deployed tracker or causal runtime state transition.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from .geometry import cxcywh_to_xyxy


def assign(prediction, ids, boxes, slot_to_gt, slots, separate_birth=False):
    gt_index = {int(identity): index for index, identity in enumerate(ids)}
    assignments = [
        (slot, gt_index[identity]) for slot, identity in slot_to_gt.items()
        if identity in gt_index
    ]
    remembered = set(slot_to_gt.values())
    candidates = (
        list(range(len(ids))) if separate_birth else
        [index for index, identity in enumerate(ids) if int(identity) not in remembered]
    )
    births = []
    if candidates:
        predicted = prediction.boxes[0, slots:].detach().float()
        target = boxes[candidates].float()
        probability = prediction.person_logits[0, slots:].detach().float().sigmoid()
        cost = (
            5 * torch.cdist(predicted, target, p=1)
            - 2 * generalized_box_iou(cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target))
            - 2 * probability[:, None]
        )
        row, col = linear_sum_assignment(cost.cpu().numpy())
        discoveries = [
            (int(r) + slots, candidates[int(c)]) for r, c in zip(row, col)
        ]
        assignments.extend(discoveries)
        births = [
            (query, index) for query, index in discoveries
            if int(ids[index]) not in remembered
        ]
    return assignments, births
