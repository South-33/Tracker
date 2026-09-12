"""The same RGB letterbox transform is used by training, replay, and export checks."""
from dataclasses import dataclass
import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    size: int
    scale_x: float
    scale_y: float
    left: int
    top: int

    def normalize_xywh(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes *= np.array([self.scale_x, self.scale_y, self.scale_x, self.scale_y])
        boxes[:, :2] += [self.left, self.top]
        return boxes / self.size

    def restore_xyxy(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        centers, sizes = boxes[:, :2].copy(), boxes[:, 2:].copy()
        boxes[:, :2], boxes[:, 2:] = centers - sizes / 2, centers + sizes / 2
        boxes *= self.size
        boxes -= [self.left, self.top, self.left, self.top]
        boxes /= [self.scale_x, self.scale_y, self.scale_x, self.scale_y]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.height)
        return boxes


def preprocess(rgb, size=640, device="cpu"):
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(rgb, (resized_w, resized_h))
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    transform = Letterbox(width, height, size, resized_w / width, resized_h / height, left, top)
    return tensor, transform


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)
