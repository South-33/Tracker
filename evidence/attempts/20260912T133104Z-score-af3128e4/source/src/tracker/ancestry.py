"""Load only the clean detector ancestry needed by the current architecture."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .detector import load_detector


class DetectorAncestry(nn.Module):
    """Minimal wrapper expected by feature caching and the active tracker."""

    def __init__(self, detector, slots):
        super().__init__()
        self.detector = detector
        self.slots = int(slots)
        self.dimension = int(detector.decoder.hidden_dim)

    def encode(self, images):
        return self.detector.encoder(self.detector.backbone(images))


def load_clean_detector(checkpoint_path, device="cuda"):
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    detector = load_detector(device=device, size=metadata["size"])
    prefix = "detector."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"checkpoint contains no {prefix!r} state: {path}")
    detector.load_state_dict(state, strict=True)
    return DetectorAncestry(detector, metadata["slots"]).to(device), metadata
