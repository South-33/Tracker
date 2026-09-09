"""Official ByteTrack baseline on exactly the shared detector's person boxes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from tracker.cli import load_model
from tracker.data import load_sequences
from tracker.detector import ROOT, load_detector
from tracker.geometry import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest-debug.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Use the fine-tuned candidate's detector weights, with no resident queries")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--track-threshold", type=float, default=.5)
    parser.add_argument("--match-threshold", type=float, default=.8)
    parser.add_argument("--buffer-seconds", type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "third_party/ByteTrack-source"))
    from yolox.tracker.byte_tracker import BYTETracker
    sequence = next(item for item in load_sequences(args.manifest, args.split) if item.name == args.sequence)
    if sequence.frames != list(range(1, len(sequence.frames) + 1)):
        raise ValueError("Baseline needs contiguous frames; skipped-frame dynamics need an explicit protocol")
    tracker = BYTETracker(SimpleNamespace(track_thresh=args.track_threshold, match_thresh=args.match_threshold,
        track_buffer=round(args.buffer_seconds * 30), mot20=False), frame_rate=sequence.fps)
    model = load_model(args).detector if args.checkpoint else load_detector(device=args.device, size=args.size)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    rows, timings = [], []
    start = time.monotonic()
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream, torch.inference_mode():
        for number in sequence.frames:
            rgb, _, _ = sequence.read(number)
            before = time.perf_counter()
            image, transform = preprocess(rgb, args.size, args.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                prediction = model(image)
            boxes = transform.restore_xyxy(prediction["pred_boxes"][0].float().cpu().numpy())
            scores = prediction["pred_logits"][0, :, 0].sigmoid().float().cpu().numpy()
            detections = np.column_stack((boxes, scores))
            detections = detections[(scores > .1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            # Input boxes are already restored; identical source and model sizes prevent a second scaling.
            shape = rgb.shape[:2]
            tracks = tracker.update(detections, shape, shape)
            observations = []
            for track in tracks:
                box = track.tlbr.tolist()
                observations.append({"id": int(track.track_id), "box": box, "person_score": float(track.score),
                                     "continuity_score": None, "status": "baseline"})
                x1, y1, x2, y2 = box
                rows.append(f"{number},{track.track_id},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},{track.score:.5f},-1,-1,-1")
            timings.append((time.perf_counter() - before) * 1000)
            stream.write(json.dumps({"frame": number, "observations": observations}) + "\n")
    (directory / f"{sequence.name}.txt").write_text("\n".join(rows) + "\n")
    metadata = {**vars(args), "method": "official ByteTrack", "run_id": str(uuid.uuid4()), "frames": len(timings),
        "source_loop_fps": len(timings) / (time.monotonic() - start),
        "frame_processing_ms": dict(zip(["p50", "p95", "p99"], np.percentile(timings[10:] or timings, [50, 95, 99]).tolist())),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "baseline_source": json.loads((ROOT / "third_party/ByteTrack-source/source.json").read_text()),
        "limit": "Laptop contiguous-prefix baseline. No Nano performance or long-gap generalization claim."}
    weight_path = Path(args.checkpoint) if args.checkpoint else ROOT / "weights/rtv4_s.pth"
    with weight_path.open("rb") as stream:
        metadata["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (directory / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
