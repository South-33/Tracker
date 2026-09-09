"""Compare independent PyTorch and ONNX feedback rollouts on actual RGB frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from tracker.cli import load_model
from tracker.data import load_sequences
from tracker.model import Memory, Prediction, RecurrentTracker
from tracker.runtime import Runtime


class OnnxModel:
    def __init__(self, path, slots=64, dimension=256):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.slots, self.dimension = slots, dimension
        self.input_names = [item.name for item in self.session.get_inputs()]
        expected = ["image", "recent", "reliable", "boxes", "age", "valid", "dt"]
        if self.input_names != expected:
            raise ValueError(f"State inputs lost or changed during export: {self.input_names}")

    def eval(self):
        return self

    def empty_memory(self):
        return Memory(torch.zeros(1, self.slots, self.dimension), torch.zeros(1, self.slots, self.dimension),
            torch.full((1, self.slots, 4), .5), torch.zeros(1, self.slots, 1), torch.zeros(1, self.slots, 1))

    def __call__(self, *inputs):
        values = self.session.run(None, {name: value.detach().cpu().numpy().astype(np.float32) for name, value in zip(self.input_names, inputs)})
        return Prediction(*(torch.from_numpy(value) for value in values))

    update_memory = RecurrentTracker.update_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", default="data/manifest-debug.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence", default="dancetrack0001")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", default="runs/onnx-parity.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.device, args.size, args.slots = "cpu", 640, 64
    reference = load_model(args)
    reference.detector.deploy()
    original = Runtime(reference, args.size)
    exported = Runtime(OnnxModel(args.onnx, args.slots), args.size)
    sequence = next(item for item in load_sequences(args.manifest, args.split) if item.name == args.sequence)
    report = {"passed": True, "precision": "FP32 CPU PyTorch vs ONNX Runtime CPU", "sequence": args.sequence,
              "scope": "Independent feedback state, identical RGB and thresholds; not TensorRT or FP16 parity.", "frames": []}
    for number in sequence.frames[:args.frames]:
        rgb, _, _ = sequence.read(number)
        first = {ob["id"]: ob for ob in original.step(rgb, 1 / sequence.fps)}
        second = {ob["id"]: ob for ob in exported.step(rgb, 1 / sequence.fps)}
        same_ids = first.keys() == second.keys()
        error = max([np.max(np.abs(np.asarray(first[key]["box"]) - second[key]["box"])) for key in first.keys() & second.keys()] or [0.])
        score_error = max([abs(first[key]["person_score"] - second[key]["person_score"]) for key in first.keys() & second.keys()] or [0.])
        state_error = float((original.memory.recent - exported.memory.recent).abs().max())
        report["frames"].append({"frame": number, "same_public_ids": same_ids, "box_error_pixels": float(error),
                                 "person_score_error": score_error, "recent_state_max_error": state_error})
        report["passed"] &= same_ids and error <= .5 and score_error <= .005
    report["thresholds"] = {"box_error_pixels": .5, "person_score_error": .005, "public_ids": "exact equality"}
    for key, path in (("checkpoint_sha256", args.checkpoint), ("onnx_sha256", args.onnx), ("manifest_sha256", args.manifest)):
        with Path(path).open("rb") as stream:
            report[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "frames": len(report["frames"]),
        "max_box_error_pixels": max(row["box_error_pixels"] for row in report["frames"]), "report": str(destination)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
