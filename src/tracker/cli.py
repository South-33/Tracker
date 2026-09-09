"""Setup verification, training, and uncut sequence replay."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .detector import ROOT, load_detector
from .geometry import preprocess
from .model import RecurrentTracker, fill_slot


def load_model(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint:
        args.slots = checkpoint["metadata"]["slots"]
        args.size = checkpoint["metadata"]["size"]
    model = RecurrentTracker(load_detector(device=args.device, size=args.size), args.slots).to(args.device).eval()
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


def verify(args):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = load_model(args)
    image = torch.rand(1, 3, args.size, args.size, device=args.device)
    state = model.empty_memory()
    dt = torch.tensor([1 / 15], device=args.device)
    with torch.no_grad():
        original = model.detector(image)
        actual = model(image, *state, dt)
    error = (actual.boxes[:, args.slots:] - original["pred_boxes"]).abs().max().item()
    score_error = (actual.person_logits[:, args.slots:] - original["pred_logits"][..., 0]).abs().max().item()
    if error > 1e-4 or score_error > 1e-3:
        raise AssertionError(f"Empty resident memory changed detector outputs: {error}, {score_error}")
    feature = actual.features[0, args.slots].detach().clone().requires_grad_(True)
    state = fill_slot(state, 0, feature, actual.boxes[0, args.slots].detach())
    remembered = model(image, *state, dt)
    objective = remembered.features[0, 0].square().mean() + remembered.person_logits[0, 0]
    objective.backward()
    gradient = feature.grad.abs().sum().item()
    if not np.isfinite(gradient) or gradient <= 0:
        raise AssertionError(f"Temporal feature gradient failed: {gradient}")
    effect = (remembered.person_logits[0, 0] - actual.person_logits[0, 0]).abs().item()
    result = {"pretrained_load": "strict", "empty_memory_box_max_error": error, "empty_memory_person_logit_max_error": score_error,
              "previous_feature_gradient_l1": gradient, "state_effect_person_logit": effect,
              "parameters": sum(p.numel() for p in model.parameters()), "device": args.device,
              "cuda": torch.version.cuda, "checkpoint": args.checkpoint,
              "limit": "Structural correctness on random images, not tracking accuracy or Nano performance."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def inspect(args):
    torch.set_num_threads(1)
    detector = load_detector(device=args.device, size=args.size)
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    image, transform = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.size, args.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        prediction = detector(image)
    boxes = transform.restore_xyxy(prediction["pred_boxes"][0].float().cpu().numpy())
    scores = prediction["pred_logits"][0, :, 0].sigmoid().float().cpu().numpy()
    records = []
    for box, score in zip(boxes, scores):
        if score < args.threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (30, 220, 30), 2)
        cv2.putText(bgr, f"{score:.2f}", (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 220, 30), 1)
        records.append({"box": box.tolist(), "score": float(score)})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), bgr):
        raise RuntimeError("Failed to save detector preview")
    destination.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} people; {destination}")


def replay(args):
    from .data import load_sequences
    from .runtime import Runtime
    torch.set_num_threads(1)
    model = load_model(args)
    runtime = Runtime(model, args.size, args.threshold, args.birth_threshold, args.continuity_threshold)
    sequences = load_sequences(args.manifest, args.split)
    sequence = next((item for item in sequences if item.name == args.sequence), None)
    if sequence is None:
        raise ValueError(f"Sequence not in selected manifest split: {args.sequence}")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(directory / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sequence.fps,
                             (sequence.record["width"], sequence.record["height"])) if args.video else None
    if writer is not None and not writer.isOpened():
        raise RuntimeError("Video writer failed to open")
    latencies, rows = [], []
    start = time.monotonic()
    try:
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            last = sequence.frames[0] - 1
            for number in sequence.frames:
                rgb, _, _ = sequence.read(number)
                before = time.perf_counter()
                observations = runtime.step(rgb, (number - last) / sequence.fps)
                latencies.append((time.perf_counter() - before) * 1000)
                last = number
                stream.write(json.dumps({"frame": number, "observations": observations}) + "\n")
                for observation in observations:
                    x1, y1, x2, y2 = observation["box"]
                    rows.append(f"{number},{observation['id']},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},{observation['person_score']:.5f},-1,-1,-1")
                    if writer is not None:
                        color = ((observation["id"] * 73) % 200 + 30, (observation["id"] * 31) % 200 + 30, (observation["id"] * 127) % 200 + 30)
                        cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(rgb, str(observation["id"]), (int(x1), max(20, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
                if writer is not None:
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if number % 30 == 0:
                    print(f"{sequence.name}:{number}, {len(observations)} observations", flush=True)
    finally:
        if writer is not None:
            writer.release()
    (directory / f"{sequence.name}.txt").write_text("\n".join(rows) + "\n")
    warm = latencies[min(10, len(latencies) - 1):]
    result = {"run_id": runtime.run_id, "sequence": sequence.name, "checkpoint": args.checkpoint,
              "frames": len(latencies), "source_loop_fps": len(latencies) / (time.monotonic() - start),
              "frame_processing_ms": dict(zip(["p50", "p95", "p99"], np.percentile(warm, [50, 95, 99]).tolist())),
              "allocated_ids": runtime.next_identity - 1, "capacity_rejections": runtime.capacity_rejections,
              "device": args.device, "threshold": args.threshold, "birth_threshold": args.birth_threshold,
              "continuity_threshold": args.continuity_threshold,
              "limit": "Laptop development replay; sequence subset; not a Nano benchmark. Untrained if checkpoint is null."}
    (directory / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def export(args):
    model = load_model(args)
    model.detector.deploy()
    image = torch.zeros(1, 3, args.size, args.size, device=args.device)
    state = model.empty_memory()
    dt = torch.tensor([1 / 15], device=args.device)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (image, *state, dt), str(destination), opset_version=17, dynamo=False,
        input_names=["image", "recent", "reliable", "boxes", "age", "valid", "dt"],
        output_names=["pred_boxes", "person_logits", "features", "continuity_logits", "quality_logits"])
    import onnx
    onnx.checker.check_model(onnx.load(destination))
    print(f"Fixed B1 neural graph exported to {destination}. Memory update/ID wrapper is outside this graph.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "inspect", "train", "replay", "export"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--size", type=int, default=640)
        sub.add_argument("--slots", type=int, default=64)
        if name != "train":
            sub.add_argument("--checkpoint")
        if name in ("train", "replay"):
            sub.add_argument("--manifest", default=str(ROOT / "data/manifest-120.json"))
        if name == "verify":
            sub.add_argument("--output", default="runs/verify.json")
        elif name == "inspect":
            sub.add_argument("--image", required=True)
            sub.add_argument("--threshold", type=float, default=.4)
            sub.add_argument("--output", default="runs/detector.jpg")
        elif name == "train":
            sub.add_argument("--output", default="runs/pilot")
            sub.add_argument("--debug", action="store_true")
            sub.add_argument("--steps", type=int, default=500)
            sub.add_argument("--unroll", type=int, default=4)
            sub.add_argument("--lr", type=float, default=2e-4)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--save-every", type=int, default=50)
            sub.add_argument("--resume")
        elif name == "replay":
            sub.add_argument("--sequence", required=True)
            sub.add_argument("--split", default="holdout", choices=["train", "dev", "holdout"])
            sub.add_argument("--output", default="runs/replay")
            sub.add_argument("--threshold", type=float, default=.4)
            sub.add_argument("--birth-threshold", type=float, default=.6)
            sub.add_argument("--continuity-threshold", type=float, default=.5)
            sub.add_argument("--video", action="store_true")
        elif name == "export":
            sub.add_argument("--output", default="runs/candidate.onnx")
    args = parser.parse_args()
    if args.command == "train":
        from .train import train
        train(args)
    else:
        {"verify": verify, "inspect": inspect, "replay": replay, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
