"""Thin integration with the pinned, independently downloaded RT-DETRv4 source."""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def load_detector(weights=None, device="cuda", size=640):
    source = ROOT / "third_party" / "RT-DETRv4"
    if not (source / "engine").is_dir():
        raise FileNotFoundError("Run python scripts/bootstrap.py first")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from engine.core import YAMLConfig
    config = YAMLConfig(str(source / "configs/rtv4/rtv4_hgnetv2_s_coco.yml"))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    config.yaml_cfg["eval_spatial_size"] = [size, size]
    # Keep the original projector for strict loading, then remove unused training-only weights.
    model = config.model
    path = Path(weights) if weights else ROOT / "weights" / "rtv4_s.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained detector required: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model.decoder.num_denoising = 0
    model.encoder.feature_projector = None
    for index in model.encoder.use_encoder_idx:
        name = f"pos_embed{index}"
        position = getattr(model.encoder, name)
        delattr(model.encoder, name)
        model.encoder.register_buffer(name, position, persistent=False)
    model.eval().to(device)
    return model


class DetectorOutput(nn.Module):
    """Fixed query order; COCO contiguous class zero is person."""
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, images):
        result = self.detector(images)
        return result["pred_boxes"], result["pred_logits"][..., 0].sigmoid()
