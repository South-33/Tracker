"""Export the frozen YOLO perception path and feature pyramid to ONNX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from tracker.model import TrackingYOLO

ROOT = Path(__file__).resolve().parents[1]


class ExportablePerception(nn.Module):
    """Prediction tensor plus the three feature maps used by ROI appearance."""

    def __init__(self, model: TrackingYOLO):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        raw, pyramid = self.model.extract(images)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        return prediction, pyramid[0], pyramid[1], pyramid[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="runs/identity-head-v2/head.pt")
    parser.add_argument("--output", default="runs/perception-640.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX Runtime numerical comparison.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        ROOT / args.head,
        map_location="cpu",
        weights_only=True,
    )
    model = TrackingYOLO(
        ROOT / checkpoint["detector"],
        checkpoint["embedding_dim"],
    ).cpu().eval()
    model.embedding.load_state_dict(checkpoint["embedding"])
    exportable = ExportablePerception(model).eval()

    torch.manual_seed(0)
    sample = torch.rand(1, 3, 640, 640)
    with torch.inference_mode():
        expected = exportable(sample)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exportable,
        sample,
        output,
        input_names=["images"],
        output_names=["prediction", "p3", "p4", "p5"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    report = {
        "output": args.output,
        "bytes": output.stat().st_size,
        "input": [1, 3, 640, 640],
        "outputs": {
            "prediction": list(expected[0].shape),
            "p3": list(expected[1].shape),
            "p4": list(expected[2].shape),
            "p5": list(expected[3].shape),
        },
        "opset": args.opset,
    }

    if not args.skip_validation:
        session = ort.InferenceSession(
            str(output),
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(None, {"images": sample.numpy()})
        validation = {}
        for name, reference, value in zip(
            ("prediction", "p3", "p4", "p5"),
            expected,
            actual,
        ):
            diff = np.abs(reference.detach().numpy() - value)
            validation[name] = {
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        report["validation"] = validation

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
