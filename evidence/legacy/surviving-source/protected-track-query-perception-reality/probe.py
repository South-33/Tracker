from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou

from tracker.ancestry import load_clean_detector
from tracker.data import load_sequences
from tracker.detections import FrozenDetections
from tracker.geometry import cxcywh_to_xyxy


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest-train-full-dense-no0096.json"
CHECKPOINT = ROOT / "runs" / "expanded-0002" / "last.pt"
OUTPUT = ROOT / "runs" / "protected-track-query-perception-reality.json"
WEIGHTS = ROOT / "runs" / "protected-track-query-perception-reality.pt"
SCORE = 0.1
OWNER_IOU = 0.5
CANDIDATES = 8
SAFE_MARGIN = 0.10
STEPS = 5000
LR = 2e-4
STABILITY_TRACK_QUERIES = 16
STABILITY_FRAMES_PER_SEQUENCE = 100


class QueryAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def inverse_sigmoid(boxes: torch.Tensor) -> torch.Tensor:
    return torch.logit(boxes.clamp(1e-4, 1 - 1e-4))


def proposal_owners(detection, ids: np.ndarray, target: torch.Tensor):
    scores = detection.person_logits[0].float().sigmoid()
    selected = torch.nonzero(detection.valid[0].bool() & (scores >= SCORE), as_tuple=False).flatten()
    if selected.numel() == 0 or len(ids) == 0:
        empty = torch.empty(0, dtype=torch.long, device=target.device)
        return selected, empty
    overlap = box_iou(cxcywh_to_xyxy(detection.boxes[0, selected].float()), cxcywh_to_xyxy(target.float()))
    best_iou, best_gt = overlap.max(dim=1)
    labels = torch.as_tensor(ids, device=target.device, dtype=torch.long)[best_gt]
    labels = torch.where(best_iou >= OWNER_IOU, labels, torch.full_like(labels, -1))
    return selected, labels


def build_rows(sequences, detections):
    grouped = defaultdict(list)
    counts = defaultdict(lambda: {"hard": 0, "safe": 0, "missing_owner": 0})
    for sequence in sequences:
        previous_number = None
        previous_payload = None
        for number in sequence.frames:
            current = detections.get(sequence, number)
            if previous_number is None or number != previous_number + 1:
                previous_number, previous_payload = number, current
                continue
            prev_det, _, prev_ids, prev_target = previous_payload
            cur_det, _, cur_ids, cur_target = current
            prev_indices, prev_labels = proposal_owners(prev_det, prev_ids, prev_target)
            cur_indices, cur_labels = proposal_owners(cur_det, cur_ids, cur_target)
            prev_gt = {int(owner): i for i, owner in enumerate(prev_ids.tolist())}
            cur_gt = {int(owner): i for i, owner in enumerate(cur_ids.tolist())}
            for owner in set(prev_gt).intersection(cur_gt):
                prev_matches = torch.nonzero(prev_labels == owner, as_tuple=False).flatten()
                if prev_matches.numel() == 0 or cur_indices.numel() == 0:
                    continue
                gt_box = prev_target[prev_gt[owner]].float()
                candidate_indices = prev_indices[prev_matches]
                candidate_boxes = prev_det.boxes[0, candidate_indices].float()
                previous_iou = box_iou(
                    cxcywh_to_xyxy(candidate_boxes), cxcywh_to_xyxy(gt_box.reshape(1, 4))
                ).squeeze(1)
                prev_query = candidate_indices[previous_iou.argmax()]
                trusted_box = prev_det.boxes[0, prev_query].float()

                spatial = box_iou(
                    cxcywh_to_xyxy(trusted_box.reshape(1, 4)),
                    cxcywh_to_xyxy(cur_det.boxes[0, cur_indices].float()),
                ).squeeze(0)
                k = min(CANDIDATES, int(spatial.numel()))
                if k == 0:
                    continue
                top = torch.topk(spatial, k=k).indices
                cloud_labels = cur_labels[top]
                if not bool((cloud_labels == owner).any()):
                    counts[sequence.name]["missing_owner"] += 1
                    continue
                baseline_safe = int(cloud_labels[0].item()) == owner
                margin = float(spatial[top[0]] - spatial[top[1]]) if k > 1 else 1.0
                kind = "hard" if not baseline_safe else ("safe" if margin <= SAFE_MARGIN else None)
                if kind is None:
                    continue
                grouped[(sequence.name, number, kind)].append({
                    "owner": int(owner),
                    "feature": prev_det.features[0, prev_query].detach().cpu().half(),
                    "trusted_box": trusted_box.detach().cpu().half(),
                    "target_box": cur_target[cur_gt[owner]].detach().cpu().half(),
                })
                counts[sequence.name][kind] += 1
            previous_number, previous_payload = number, current
    return grouped, {key: value for key, value in counts.items()}


def prepare_decoder_inputs(base, pyramid):
    decoder = base.detector.decoder
    memory, shapes = decoder._get_encoder_input(pyramid)
    discovery, references, _, _ = decoder._get_decoder_input(memory, shapes)
    return decoder, memory, shapes, discovery, references


def decode_with_tracks(base, pyramid, adapter, features, trusted_boxes):
    use_amp = features.device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            decoder, memory, shapes, discovery, references = prepare_decoder_inputs(base, pyramid)
        track_content = adapter(features.float()).to(discovery.dtype)
        track_refs = inverse_sigmoid(trusted_boxes.float()).to(references.dtype)
        content = torch.cat((discovery.detach(), track_content.unsqueeze(0)), dim=1)
        refs = torch.cat((references.detach(), track_refs.unsqueeze(0)), dim=1)
        normal = discovery.shape[1]
        total = content.shape[1]
        mask = torch.zeros((total, total), dtype=torch.bool, device=content.device)
        mask[:normal, normal:] = True
        result = decoder.decoder(
            content, refs, memory.detach(), shapes,
            decoder.dec_bbox_head, decoder.dec_score_head, decoder.query_pos_head,
            decoder.pre_bbox_head, decoder.integral, decoder.up, decoder.reg_scale,
            attn_mask=mask,
        )
    return result, normal


def track_loss(pred_boxes, pred_logits, target_boxes):
    pred = pred_boxes.float()
    target = target_boxes.float()
    l1 = F.l1_loss(pred, target)
    giou = generalized_box_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target)).diag()
    cls = F.binary_cross_entropy_with_logits(pred_logits.float(), torch.ones_like(pred_logits.float()))
    return 5.0 * l1 + 2.0 * (1 - giou).mean() + 0.25 * cls


@torch.no_grad()
def evaluate_rows(base, feature_cache, sequence, groups, adapter, device):
    by_frame = defaultdict(list)
    for (name, number, kind), rows in groups.items():
        if name == sequence.name:
            by_frame[number].extend((kind, row) for row in rows)
    counts = {"hard": 0, "hard_correct": 0, "safe": 0, "safe_correct": 0, "all": 0, "all_correct": 0}
    for number in sorted(by_frame):
        items = by_frame[number]
        pyramid, ids, target = feature_cache.get(sequence, number)
        features = torch.stack([row["feature"] for _, row in items]).to(device)
        trusted = torch.stack([row["trusted_box"] for _, row in items]).to(device)
        result, normal = decode_with_tracks(base, pyramid, adapter, features, trusted)
        pred = result[0][-1][0, normal:].float()
        overlap = box_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target.float()))
        best_iou, best_gt = overlap.max(dim=1)
        assigned = torch.as_tensor(ids, device=device, dtype=torch.long)[best_gt]
        for index, (kind, row) in enumerate(items):
            correct = bool(best_iou[index] >= OWNER_IOU and int(assigned[index]) == row["owner"])
            counts[kind] += 1
            counts[f"{kind}_correct"] += int(correct)
            counts["all"] += 1
            counts["all_correct"] += int(correct)
    return {
        **counts,
        "hard_recovery": counts["hard_correct"] / max(1, counts["hard"]),
        "safe_accuracy": counts["safe_correct"] / max(1, counts["safe"]),
        "accuracy": counts["all_correct"] / max(1, counts["all"]),
    }


@torch.no_grad()
def detector_stability(base, feature_cache, sequences, adapter, device):
    total_overlap = 0.0
    samples = 0
    baseline_visible = mixed_visible = visible_rows = 0
    max_box = max_logit = 0.0
    for sequence in sequences:
        frames = list(sequence.frames)
        if len(frames) > STABILITY_FRAMES_PER_SEQUENCE:
            positions = np.linspace(0, len(frames) - 1, STABILITY_FRAMES_PER_SEQUENCE).round().astype(int)
            frames = [frames[index] for index in positions]
        for number in frames:
            pyramid, ids, target = feature_cache.get(sequence, number)
            use_amp = device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                decoder, memory, shapes, discovery, references = prepare_decoder_inputs(base, pyramid)
                plain = decoder.decoder(
                    discovery, references, memory, shapes,
                    decoder.dec_bbox_head, decoder.dec_score_head, decoder.query_pos_head,
                    decoder.pre_bbox_head, decoder.integral, decoder.up, decoder.reg_scale,
                )
                dummy_features = torch.zeros((STABILITY_TRACK_QUERIES, base.dimension), device=device)
                dummy_boxes = torch.full((STABILITY_TRACK_QUERIES, 4), 0.5, device=device)
                mixed, normal = decode_with_tracks(base, pyramid, adapter, dummy_features, dummy_boxes)
            plain_boxes = plain[0][-1][0].float()
            plain_logits = plain[1][-1][0, :, 0].float()
            mixed_boxes = mixed[0][-1][0, :normal].float()
            mixed_logits = mixed[1][-1][0, :normal, 0].float()
            max_box = max(max_box, float((plain_boxes - mixed_boxes).abs().max()))
            max_logit = max(max_logit, float((plain_logits - mixed_logits).abs().max()))
            plain_top = set(torch.topk(plain_logits, 96).indices.cpu().tolist())
            mixed_top = set(torch.topk(mixed_logits, 96).indices.cpu().tolist())
            total_overlap += len(plain_top & mixed_top) / len(plain_top | mixed_top)
            samples += 1
            if len(ids):
                plain_selected = plain_boxes[torch.topk(plain_logits, 96).indices]
                mixed_selected = mixed_boxes[torch.topk(mixed_logits, 96).indices]
                plain_iou = box_iou(cxcywh_to_xyxy(target.float()), cxcywh_to_xyxy(plain_selected)).max(dim=1).values
                mixed_iou = box_iou(cxcywh_to_xyxy(target.float()), cxcywh_to_xyxy(mixed_selected)).max(dim=1).values
                visible_rows += len(ids)
                baseline_visible += int((plain_iou >= OWNER_IOU).sum())
                mixed_visible += int((mixed_iou >= OWNER_IOU).sum())
    baseline_rate = baseline_visible / max(1, visible_rows)
    mixed_rate = mixed_visible / max(1, visible_rows)
    return {
        "frames": samples,
        "track_queries": STABILITY_TRACK_QUERIES,
        "mean_top96_query_set_jaccard": total_overlap / max(1, samples),
        "max_box_abs_diff": max_box,
        "max_person_logit_abs_diff": max_logit,
        "visible_rows": visible_rows,
        "baseline_top96_gt_coverage": baseline_rate,
        "mixed_top96_gt_coverage": mixed_rate,
        "coverage_delta": mixed_rate - baseline_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(args.device)

    base, metadata = load_clean_detector(CHECKPOINT, device=device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    detections = FrozenDetections(
        base, metadata["size"], device,
        ROOT / "data" / "features", ROOT / "data" / "detections",
        candidate_count=96, cache_mib=256,
    )
    train_sequences = load_sequences(MANIFEST, "train")
    dev_sequences = load_sequences(MANIFEST, "dev")

    cache_path = Path(__file__).with_name("rows.pt")
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_groups, train_counts = cached["train_groups"], cached["train_counts"]
        dev_groups, dev_counts = cached["dev_groups"], cached["dev_counts"]
    else:
        train_groups, train_counts = build_rows(train_sequences, detections)
        dev_groups, dev_counts = build_rows(dev_sequences, detections)
        torch.save({
            "train_groups": train_groups, "train_counts": train_counts,
            "dev_groups": dev_groups, "dev_counts": dev_counts,
        }, cache_path)

    hard_keys = [key for key in train_groups if key[2] == "hard"]
    safe_keys = [key for key in train_groups if key[2] == "safe"]
    sequence_map = {sequence.name: sequence for sequence in train_sequences}
    adapter = QueryAdapter(base.dimension).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=1e-4)
    curve = []
    for step in range(1, STEPS + 1):
        key = random.choice(hard_keys if random.random() < 0.5 else safe_keys)
        rows = train_groups[key]
        sequence = sequence_map[key[0]]
        pyramid, _, _ = detections.features.get(sequence, key[1])
        features = torch.stack([row["feature"] for row in rows]).to(device)
        trusted = torch.stack([row["trusted_box"] for row in rows]).to(device)
        target = torch.stack([row["target_box"] for row in rows]).to(device)
        optimizer.zero_grad(set_to_none=True)
        result, normal = decode_with_tracks(base, pyramid, adapter, features, trusted)
        loss = track_loss(
            result[0][-1][0, normal:], result[1][-1][0, normal:, 0], target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 500 == 0:
            point = {"step": step, "loss": float(loss.detach())}
            curve.append(point)
            print(json.dumps(point), flush=True)

    per_sequence = {
        sequence.name: evaluate_rows(base, detections.features, sequence, dev_groups, adapter, device)
        for sequence in dev_sequences
    }
    combined_counts = {
        key: sum(result[key] for result in per_sequence.values())
        for key in ("hard", "hard_correct", "safe", "safe_correct", "all", "all_correct")
    }
    combined = {
        **combined_counts,
        "hard_recovery": combined_counts["hard_correct"] / max(1, combined_counts["hard"]),
        "safe_accuracy": combined_counts["safe_correct"] / max(1, combined_counts["safe"]),
        "accuracy": combined_counts["all_correct"] / max(1, combined_counts["all"]),
    }
    stability = detector_stability(base, detections.features, dev_sequences, adapter, device)
    criteria = {
        "hard_recovery_min": 0.45,
        "safe_accuracy_min": 0.90,
        "top96_jaccard_min": 0.99,
        "coverage_delta_min": -0.005,
    }
    passed = (
        combined["hard_recovery"] >= criteria["hard_recovery_min"]
        and combined["safe_accuracy"] >= criteria["safe_accuracy_min"]
        and stability["mean_top96_query_set_jaccard"] >= criteria["top96_jaccard_min"]
        and stability["coverage_delta"] >= criteria["coverage_delta_min"]
    )
    result = {
        "scope": "frozen non-gate dev Reality probe for learned protected track queries inside the existing RT-DETR decoder",
        "settings": {
            "steps": STEPS, "lr": LR, "candidate_cloud": CANDIDATES,
            "safe_margin": SAFE_MARGIN, "owner_iou": OWNER_IOU,
            "sampling": "50% hard frame / 50% low-margin safe frame over all train sequences",
            "trainable": "263424-parameter zero-init residual protected-query adapter; detector/decoder frozen",
            "ordinary_query_rule": "normal rows cannot attend protected rows; measured FP16 stability is part of the pass criteria",
        },
        "criteria": criteria,
        "decision": "pass" if passed else "fail",
        "train_counts": train_counts,
        "dev_counts": dev_counts,
        "curve": curve,
        "per_sequence": per_sequence,
        "combined": combined,
        "detector_stability": stability,
        "cache": detections.stats(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "adapter": adapter.state_dict(),
        "metadata": {"dimension": base.dimension, "steps": STEPS, "lr": LR, "criteria": criteria},
    }, WEIGHTS)
    print(json.dumps({"decision": result["decision"], "combined": combined, "detector_stability": stability}, indent=2), flush=True)


if __name__ == "__main__":
    main()
