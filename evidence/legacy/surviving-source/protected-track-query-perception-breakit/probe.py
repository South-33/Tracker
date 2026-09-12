from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from tracker.ancestry import load_clean_detector
from tracker.data import load_sequences
from tracker.detections import FrozenDetections
from tracker.geometry import cxcywh_to_xyxy


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest-train-full-dense-no0096.json"
CHECKPOINT = ROOT / "runs" / "expanded-0002" / "last.pt"
WEIGHTS = ROOT / "runs" / "protected-track-query-perception-reality.pt"
REALITY_ROWS = ROOT / "runs" / "protected-track-query-perception-reality" / "rows.pt"
REALITY_SOURCE = ROOT / "runs" / "protected-track-query-perception-reality" / "probe.py"
OUTPUT = ROOT / "runs" / "protected-track-query-perception-breakit.json"

SCORE = 0.1
OWNER_IOU = 0.5
AGREEMENT_IOU = 0.5
FIRST_GAP = 2
MAX_GAP = 5


def load_reality_module():
    spec = importlib.util.spec_from_file_location("protected_query_reality", REALITY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Reality source: {REALITY_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_key(sequence: str, ambiguous_frame: int, kind: str, owner: int) -> tuple[str, int, str, int]:
    return sequence, int(ambiguous_frame), kind, int(owner)


def make_attempts(dev_groups, sequences):
    available = {sequence.name: set(sequence.frames) for sequence in sequences}
    attempts = defaultdict(list)
    events = {}
    for (sequence, ambiguous_frame, kind), rows in dev_groups.items():
        anchor_frame = int(ambiguous_frame) - 1
        for row in rows:
            key = event_key(sequence, ambiguous_frame, kind, row["owner"])
            if key in events:
                raise ValueError(f"duplicate ambiguous event: {key}")
            events[key] = {
                "sequence": sequence,
                "ambiguous_frame": int(ambiguous_frame),
                "anchor_frame": anchor_frame,
                "kind": kind,
                "owner": int(row["owner"]),
                "feature": row["feature"],
                "trusted_box": row["trusted_box"],
            }
            for gap in range(FIRST_GAP, MAX_GAP + 1):
                number = anchor_frame + gap
                if number in available[sequence]:
                    attempts[(sequence, number)].append((key, gap))
    return events, attempts


def owner_index(ids: np.ndarray, owner: int) -> int | None:
    matches = np.nonzero(ids == int(owner))[0]
    return int(matches[0]) if len(matches) else None


@torch.no_grad()
def evaluate_attempts(base, detections, sequences, adapter, reality, events, attempts, device):
    sequence_map = {sequence.name: sequence for sequence in sequences}
    records = defaultdict(list)
    gap_stats = {
        gap: {
            "attempts": 0,
            "owner_visible": 0,
            "direct_correct": 0,
            "certified": 0,
            "certified_safe": 0,
            "certified_wrong": 0,
            "absent_attempts": 0,
            "absent_certified": 0,
        }
        for gap in range(FIRST_GAP, MAX_GAP + 1)
    }

    for (sequence_name, number), items in sorted(attempts.items()):
        sequence = sequence_map[sequence_name]
        detection, _, ids, target = detections.get(sequence, number)
        pyramid, _, _ = detections.features.get(sequence, number)
        features = torch.stack([events[key]["feature"] for key, _ in items]).to(device)
        trusted = torch.stack([events[key]["trusted_box"] for key, _ in items]).to(device)
        decoded, normal = reality.decode_with_tracks(base, pyramid, adapter, features, trusted)
        query_boxes = decoded[0][-1][0, normal:].float()
        query_scores = decoded[1][-1][0, normal:, 0].float().sigmoid()

        proposal_indices, proposal_labels = reality.proposal_owners(detection, ids, target)
        proposal_boxes = detection.boxes[0, proposal_indices].float() if proposal_indices.numel() else None

        for i, (key, gap) in enumerate(items):
            event = events[key]
            owner = event["owner"]
            visible_index = owner_index(ids, owner)
            visible = visible_index is not None
            direct_correct = False
            if visible:
                direct_iou = box_iou(
                    cxcywh_to_xyxy(query_boxes[i].reshape(1, 4)),
                    cxcywh_to_xyxy(target[visible_index].float().reshape(1, 4)),
                )[0, 0]
                direct_correct = bool(direct_iou >= OWNER_IOU)

            certified = False
            certified_safe = False
            local_label = -1
            agreement_iou = 0.0
            if proposal_boxes is not None and proposal_boxes.numel():
                trusted_box = event["trusted_box"].to(device=device, dtype=torch.float32)
                spatial = box_iou(
                    cxcywh_to_xyxy(trusted_box.reshape(1, 4)),
                    cxcywh_to_xyxy(proposal_boxes),
                ).squeeze(0)
                local = int(spatial.argmax())
                local_box = proposal_boxes[local]
                local_label = int(proposal_labels[local])
                agreement_iou = float(box_iou(
                    cxcywh_to_xyxy(query_boxes[i].reshape(1, 4)),
                    cxcywh_to_xyxy(local_box.reshape(1, 4)),
                )[0, 0])
                certified = bool(query_scores[i] >= SCORE and agreement_iou >= AGREEMENT_IOU)
                certified_safe = bool(certified and visible and local_label == owner)

            record = {
                "gap": gap,
                "visible": visible,
                "direct_correct": direct_correct,
                "query_score": float(query_scores[i]),
                "certified": certified,
                "certified_safe": certified_safe,
                "certified_wrong": bool(certified and not certified_safe),
                "local_label": local_label,
                "agreement_iou": agreement_iou,
            }
            records[key].append(record)

            stats = gap_stats[gap]
            stats["attempts"] += 1
            stats["owner_visible"] += int(visible)
            stats["direct_correct"] += int(direct_correct)
            stats["certified"] += int(certified)
            stats["certified_safe"] += int(certified_safe)
            stats["certified_wrong"] += int(certified and not certified_safe)
            stats["absent_attempts"] += int(not visible)
            stats["absent_certified"] += int((not visible) and certified)

    return records, gap_stats


def summarize_first_accept(events, records, max_gap: int):
    by_kind = {
        kind: {
            "events": 0,
            "recoverable_events": 0,
            "safe_accepts": 0,
            "wrong_accepts": 0,
            "unresolved": 0,
        }
        for kind in ("safe", "hard")
    }
    accepted_gaps = []
    for key, event in events.items():
        kind = event["kind"]
        eligible = [row for row in records.get(key, []) if row["gap"] <= max_gap]
        stats = by_kind[kind]
        stats["events"] += 1
        stats["recoverable_events"] += int(any(row["visible"] for row in eligible))
        first = next((row for row in sorted(eligible, key=lambda row: row["gap"]) if row["certified"]), None)
        if first is None:
            stats["unresolved"] += 1
        elif first["certified_safe"]:
            stats["safe_accepts"] += 1
            accepted_gaps.append(first["gap"])
        else:
            stats["wrong_accepts"] += 1

    for stats in by_kind.values():
        accepted = stats["safe_accepts"] + stats["wrong_accepts"]
        stats["accepted_precision"] = stats["safe_accepts"] / max(1, accepted)
        stats["recovery_rate"] = stats["safe_accepts"] / max(1, stats["recoverable_events"])
        stats["wrong_event_rate"] = stats["wrong_accepts"] / max(1, stats["events"])

    combined_safe = sum(stats["safe_accepts"] for stats in by_kind.values())
    combined_wrong = sum(stats["wrong_accepts"] for stats in by_kind.values())
    combined = {
        "safe_accepts": combined_safe,
        "wrong_accepts": combined_wrong,
        "accepted_precision": combined_safe / max(1, combined_safe + combined_wrong),
        "mean_safe_accept_gap": float(np.mean(accepted_gaps)) if accepted_gaps else None,
    }
    return {"max_gap": max_gap, "by_kind": by_kind, "combined": combined}


def summarize_gaps(gap_stats):
    result = {}
    for gap, stats in gap_stats.items():
        certified = stats["certified"]
        result[str(gap)] = {
            **stats,
            "direct_accuracy_when_visible": stats["direct_correct"] / max(1, stats["owner_visible"]),
            "certified_precision": stats["certified_safe"] / max(1, certified),
            "certificate_rate": certified / max(1, stats["attempts"]),
            "absent_accept_rate": stats["absent_certified"] / max(1, stats["absent_attempts"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    reality = load_reality_module()
    base, metadata = load_clean_detector(CHECKPOINT, device=device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    adapter = reality.QueryAdapter(base.dimension).to(device)
    payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    detections = FrozenDetections(
        base, metadata["size"], device,
        ROOT / "data" / "features", ROOT / "data" / "detections",
        candidate_count=96, cache_mib=256,
    )
    dev_sequences = load_sequences(MANIFEST, "dev")
    cached = torch.load(REALITY_ROWS, map_location="cpu", weights_only=False)
    dev_groups = cached["dev_groups"]
    events, attempts = make_attempts(dev_groups, dev_sequences)
    records, gap_stats = evaluate_attempts(
        base, detections, dev_sequences, adapter, reality, events, attempts, device
    )
    by3 = summarize_first_accept(events, records, 3)
    by5 = summarize_first_accept(events, records, 5)
    gaps = summarize_gaps(gap_stats)

    absent_attempts = sum(stats["absent_attempts"] for stats in gap_stats.values())
    absent_certified = sum(stats["absent_certified"] for stats in gap_stats.values())
    absent_accept_rate = absent_certified / max(1, absent_attempts)
    criteria = {
        "accepted_precision_by5_min": 0.99,
        "safe_recovery_by3_min": 0.70,
        "safe_recovery_by5_min": 0.80,
        "absent_accept_rate_max": 0.01,
    }
    passed = (
        by5["combined"]["accepted_precision"] >= criteria["accepted_precision_by5_min"]
        and by3["by_kind"]["safe"]["recovery_rate"] >= criteria["safe_recovery_by3_min"]
        and by5["by_kind"]["safe"]["recovery_rate"] >= criteria["safe_recovery_by5_min"]
        and absent_accept_rate <= criteria["absent_accept_rate_max"]
    )
    result = {
        "scope": "frozen non-gate dev break-it probe: force an ambiguous-frame veto, keep the protected anchor immutable, and ask whether later protected-query/local-proposal agreement safely reanchors without a private trajectory",
        "mechanism": {
            "anchor": "same trusted feature and trusted box from frame t for every retry; no intermediate update",
            "attempts": "frames t+2 through t+5 after deliberately withholding the ambiguous t+1 observation",
            "certificate": "query person score >= 0.1 and query box IoU >= 0.5 with the stale-anchor track-perspective best current detector proposal",
            "publication": "the current detector proposal, never the stale anchor or raw query box",
        },
        "criteria": criteria,
        "decision": "pass" if passed else "fail",
        "events": {"total": len(events), "safe": sum(e["kind"] == "safe" for e in events.values()), "hard": sum(e["kind"] == "hard" for e in events.values())},
        "per_gap": gaps,
        "first_accept_by_gap3": by3,
        "first_accept_by_gap5": by5,
        "absence": {
            "attempts": absent_attempts,
            "certified": absent_certified,
            "accept_rate": absent_accept_rate,
        },
        "cache": detections.stats(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "by3": by3,
        "by5": by5,
        "absence": result["absence"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
