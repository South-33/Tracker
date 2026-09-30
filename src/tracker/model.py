"""YOLO26n visual features with bounded recurrent neural track slots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO


class TemporalSlotHead(nn.Module):
    """Update persistent anonymous person slots from current visual tokens."""

    def __init__(
        self,
        feature_channels: tuple[int, ...],
        *,
        slots: int = 64,
        memory_dim: int = 128,
        attention_heads: int = 4,
        pool_sizes: tuple[int, ...] = (20, 10, 5),
    ):
        super().__init__()
        if len(feature_channels) != len(pool_sizes):
            raise ValueError("one pool size is required for each feature level")
        if memory_dim % attention_heads:
            raise ValueError("memory_dim must be divisible by attention_heads")
        self.slots = int(slots)
        self.memory_dim = int(memory_dim)
        self.pool_sizes = tuple(int(x) for x in pool_sizes)

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, memory_dim, 1)
            for channels in feature_channels
        )
        self.scale_embedding = nn.Parameter(
            torch.zeros(len(feature_channels), memory_dim)
        )
        self.initial_slots = nn.Parameter(
            torch.randn(slots, memory_dim) * 0.02
        )
        self.slot_norm = nn.LayerNorm(memory_dim)
        self.token_norm = nn.LayerNorm(memory_dim)
        self.attention = nn.MultiheadAttention(
            memory_dim,
            attention_heads,
            batch_first=True,
        )
        self.update = nn.GRUCell(memory_dim, memory_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, memory_dim * 2),
            nn.SiLU(),
            nn.Linear(memory_dim * 2, memory_dim),
        )
        self.output_norm = nn.LayerNorm(memory_dim)
        self.alive = nn.Linear(memory_dim, 1)
        self.box = nn.Linear(memory_dim, 4)

    def initial_memory(
        self,
        batch_size: int,
        *,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        memory = self.initial_slots
        if device is not None or dtype is not None:
            memory = memory.to(device=device, dtype=dtype)
        return memory.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def visual_tokens(
        self,
        pyramid: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        tokens = []
        for index, (feature, projection, size) in enumerate(
            zip(pyramid, self.projections, self.pool_sizes)
        ):
            projected = projection(feature)
            pooled = F.adaptive_avg_pool2d(projected, (size, size))
            level = pooled.flatten(2).transpose(1, 2)
            level = level + self.scale_embedding[index][None, None, :]
            tokens.append(level)
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        pyramid: tuple[torch.Tensor, ...],
        memory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.visual_tokens(pyramid)
        batch_size = tokens.shape[0]
        if memory is None:
            memory = self.initial_memory(
                batch_size,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        if memory.shape != (batch_size, self.slots, self.memory_dim):
            raise ValueError(
                "memory must have shape "
                f"[B, {self.slots}, {self.memory_dim}]"
            )

        context, _ = self.attention(
            self.slot_norm(memory),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        flat_context = context.reshape(-1, self.memory_dim)
        flat_memory = memory.reshape(-1, self.memory_dim)
        updated = self.update(flat_context, flat_memory).view_as(memory)
        updated = updated + self.ffn(updated)
        output = self.output_norm(updated)
        return {
            "alive_logits": self.alive(output).squeeze(-1),
            "boxes_cxcywh": self.box(output).sigmoid(),
            "memory": updated,
        }


class TemporalYOLO(nn.Module):
    """Pretrained YOLO26n perception plus a recurrent track-slot head."""

    def __init__(
        self,
        weights: str | Path | nn.Module = "yolo26n.pt",
        *,
        slots: int = 64,
        memory_dim: int = 128,
        freeze_visual: bool = True,
    ):
        super().__init__()
        self.freeze_visual = bool(freeze_visual)
        self.detector = (
            weights
            if isinstance(weights, nn.Module)
            else YOLO(str(weights)).model
        )
        detect = self.detector.model[-1]
        channels = tuple(
            branch[0].conv.in_channels
            for branch in detect.cv2
        )
        self.temporal = TemporalSlotHead(
            channels,
            slots=slots,
            memory_dim=memory_dim,
        )
        self._pyramid = None
        self._hook = detect.register_forward_pre_hook(self._capture_pyramid)
        self.set_visual_trainable(not freeze_visual)

    @property
    def slots(self) -> int:
        return self.temporal.slots

    @property
    def memory_dim(self) -> int:
        return self.temporal.memory_dim

    def _capture_pyramid(self, _module, args):
        self._pyramid = tuple(args[0])

    def set_visual_trainable(self, enabled: bool) -> None:
        self.freeze_visual = not enabled
        for parameter in self.detector.parameters():
            parameter.requires_grad_(enabled)
        if self.freeze_visual:
            self.detector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual:
            self.detector.eval()
        return self

    def extract_pyramid(self, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._pyramid = None
        if self.freeze_visual:
            with torch.no_grad():
                self.detector(frame)
        else:
            self.detector(frame)
        if self._pyramid is None:
            raise RuntimeError("YOLO feature hook did not run")
        return self._pyramid

    def step(
        self,
        frame: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pyramid = self.extract_pyramid(frame)
        return self.temporal(pyramid, memory)

    def forward(
        self,
        clip: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        if clip.ndim != 5:
            raise ValueError("clip must have shape [B, T, C, H, W]")
        outputs = []
        state = memory
        for time_index in range(clip.shape[1]):
            result = self.step(clip[:, time_index], state)
            state = result["memory"]
            outputs.append(result)
        return outputs


def _new_identity_assignment(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    free_slots: list[int],
) -> list[tuple[int, int]]:
    """Training-only matching for identities when they first enter a clip."""
    if not len(target_boxes):
        return []
    if len(target_boxes) > len(free_slots):
        raise ValueError("clip contains more identities than available slots")
    slot_tensor = torch.as_tensor(
        free_slots,
        device=predicted_boxes.device,
        dtype=torch.long,
    )
    cost = torch.cdist(
        predicted_boxes[slot_tensor].detach(),
        target_boxes.detach(),
        p=1,
    )
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    return [
        (free_slots[int(row)], int(column))
        for row, column in zip(rows, columns)
    ]


def temporal_slot_loss(
    outputs: list[dict[str, torch.Tensor]],
    boxes: list[list[torch.Tensor]],
    identities: list[list[torch.Tensor]],
    *,
    positive_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sequence loss with a fixed neural slot for each identity in a clip.

    Hungarian matching is used only during training when an identity first
    appears. Once assigned, that identity keeps the same slot for the remainder
    of the clip. Inference has no Hungarian or external ID association.
    """
    if not outputs:
        raise ValueError("outputs cannot be empty")
    batch_size = outputs[0]["alive_logits"].shape[0]
    if len(boxes) != batch_size or len(identities) != batch_size:
        raise ValueError("target batch size does not match model outputs")

    mappings: list[dict[int, int]] = [dict() for _ in range(batch_size)]
    total_presence = outputs[0]["alive_logits"].new_zeros(())
    total_box = outputs[0]["alive_logits"].new_zeros(())
    visible_count = 0
    frame_count = 0

    for time_index, output in enumerate(outputs):
        logits = output["alive_logits"]
        predicted_boxes = output["boxes_cxcywh"]
        target_alive = torch.zeros_like(logits)

        for batch_index in range(batch_size):
            frame_boxes = boxes[batch_index][time_index].to(
                predicted_boxes.device
            )
            frame_ids = identities[batch_index][time_index].to(
                predicted_boxes.device
            )
            mapping = mappings[batch_index]

            new_positions = [
                position
                for position, identity in enumerate(frame_ids.tolist())
                if int(identity) not in mapping
            ]
            if new_positions:
                used = set(mapping.values())
                free = [
                    slot
                    for slot in range(predicted_boxes.shape[1])
                    if slot not in used
                ]
                new_boxes = frame_boxes[new_positions]
                assignments = _new_identity_assignment(
                    predicted_boxes[batch_index],
                    new_boxes,
                    free,
                )
                for slot, column in assignments:
                    identity = int(frame_ids[new_positions[column]])
                    mapping[identity] = slot

            visible_slots = []
            visible_targets = []
            for position, identity in enumerate(frame_ids.tolist()):
                identity = int(identity)
                slot = mapping.get(identity)
                if slot is None:
                    continue
                target_alive[batch_index, slot] = 1.0
                visible_slots.append(slot)
                visible_targets.append(frame_boxes[position])

            if visible_slots:
                slot_tensor = torch.tensor(
                    visible_slots,
                    device=predicted_boxes.device,
                    dtype=torch.long,
                )
                target_tensor = torch.stack(visible_targets)
                total_box = total_box + F.l1_loss(
                    predicted_boxes[batch_index, slot_tensor],
                    target_tensor,
                    reduction="sum",
                )
                visible_count += len(visible_slots)

        total_presence = total_presence + F.binary_cross_entropy_with_logits(
            logits,
            target_alive,
            pos_weight=logits.new_tensor(positive_weight),
            reduction="mean",
        )
        frame_count += 1

    presence_loss = total_presence / max(frame_count, 1)
    box_loss = total_box / max(visible_count * 4, 1)
    loss = presence_loss + 5.0 * box_loss
    return loss, {
        "loss": float(loss.detach()),
        "presence_loss": float(presence_loss.detach()),
        "box_loss": float(box_loss.detach()),
        "visible_targets": float(visible_count),
    }
