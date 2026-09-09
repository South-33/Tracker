"""Fixed resident queries share the pretrained decoder with discovery queries."""
from typing import NamedTuple
import torch
from torch import nn


class Memory(NamedTuple):
    recent: torch.Tensor
    reliable: torch.Tensor
    boxes: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor

    def detached(self):
        return Memory(*(value.detach() for value in self))


class Prediction(NamedTuple):
    boxes: torch.Tensor
    person_logits: torch.Tensor
    features: torch.Tensor
    continuity_logits: torch.Tensor
    quality_logits: torch.Tensor


class RecurrentTracker(nn.Module):
    def __init__(self, detector, slots=64):
        super().__init__()
        self.detector = detector
        self.slots = slots
        dimension = detector.decoder.hidden_dim
        self.dimension = dimension
        self.memory_adapter = nn.Sequential(nn.Linear(dimension * 2 + 2, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        nn.init.zeros_(self.memory_adapter[-1].weight)
        nn.init.zeros_(self.memory_adapter[-1].bias)
        self.continuity = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.ReLU(), nn.Linear(dimension, 1))
        self.quality = nn.Linear(dimension, 1)
        for part in (detector.backbone, detector.encoder):
            part.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen feature BN statistics and the detector's final-only decoder path are intentional.
        # eval() preserves autograd. Temporal gradients are only cut at explicit training boundaries.
        self.detector.eval()
        return self

    def empty_memory(self, device=None, dtype=None):
        parameter = next(self.parameters())
        options = {"device": device or parameter.device, "dtype": dtype or parameter.dtype}
        return Memory(torch.zeros(1, self.slots, self.dimension, **options),
                      torch.zeros(1, self.slots, self.dimension, **options),
                      torch.full((1, self.slots, 4), 0.5, **options),
                      torch.zeros(1, self.slots, 1, **options),
                      torch.zeros(1, self.slots, 1, **options))

    def encode(self, images):
        return self.detector.encoder(self.detector.backbone(images))

    def decode(self, pyramid, memory, dt):
        decoder = self.detector.decoder
        image_features, shapes = decoder._get_encoder_input(pyramid)
        discovery, references, _, _ = decoder._get_decoder_input(image_features, shapes)
        times = torch.cat((memory.age.clamp(0, 10) / 10, dt.reshape(1, 1, 1).expand_as(memory.age).clamp(0, 10) / 10), -1)
        resident = memory.recent + self.memory_adapter(torch.cat((memory.recent, memory.reliable, times), -1))
        resident = resident * memory.valid
        boxes = memory.boxes.clamp(0.0001, 0.9999)
        resident_refs = torch.log(boxes / (1 - boxes))
        content = torch.cat((resident, discovery), 1)
        refs = torch.cat((resident_refs, references), 1)
        invalid = torch.cat((memory.valid[0, :, 0] < 0.5, torch.zeros(discovery.shape[1], device=discovery.device, dtype=torch.bool)))
        mask = invalid.unsqueeze(0).expand(content.shape[1], -1)
        result = decoder.decoder(content, refs, image_features, shapes, decoder.dec_bbox_head,
            decoder.dec_score_head, decoder.query_pos_head, decoder.pre_bbox_head,
            decoder.integral, decoder.up, decoder.reg_scale, attn_mask=mask, return_features=True)
        features = result[-1]
        logits = result[1][-1][..., 0]
        history = torch.cat((memory.reliable, torch.zeros_like(discovery)), 1)
        continuity = self.continuity(torch.cat((features, history), -1)).squeeze(-1)
        quality = self.quality(features).squeeze(-1)
        return Prediction(result[0][-1], logits, features, continuity, quality)

    def forward(self, images, recent, reliable, boxes, age, valid, dt):
        return self.decode(self.encode(images), Memory(recent, reliable, boxes, age, valid), dt)

    def update_memory(self, memory, prediction, observed, dt):
        """Caller supplies a visibility decision, never an appearance/geometry association."""
        observed = observed.reshape(1, self.slots, 1).to(memory.recent.dtype) * memory.valid
        current = prediction.features[:, :self.slots]
        gate = prediction.quality_logits[:, :self.slots].sigmoid().unsqueeze(-1)
        recent = memory.recent * (1 - observed) + current * observed
        update = gate * observed
        reliable = memory.reliable * (1 - update) + current * update
        boxes = memory.boxes * (1 - observed) + prediction.boxes[:, :self.slots] * observed
        age = (memory.age + dt.reshape(1, 1, 1)) * (1 - observed)
        return Memory(recent, reliable, boxes, age, memory.valid)


def fill_slot(memory, slot, feature, box):
    """Differentiable slot replacement also clears all previous occupant state."""
    index = torch.arange(memory.valid.shape[1], device=memory.valid.device)
    mask = (index == slot).reshape(1, -1, 1).to(memory.recent.dtype)
    recent = memory.recent * (1 - mask) + feature.reshape(1, 1, -1) * mask
    reliable = memory.reliable * (1 - mask) + feature.reshape(1, 1, -1) * mask
    boxes = memory.boxes * (1 - mask) + box.reshape(1, 1, 4) * mask
    return Memory(recent, reliable, boxes, memory.age * (1 - mask), torch.maximum(memory.valid, mask))


def expire_memory(memory, maximum_age=10.0):
    valid = memory.valid * (memory.age <= maximum_age).to(memory.valid.dtype)
    return Memory(memory.recent * valid, memory.reliable * valid,
                  memory.boxes * valid + 0.5 * (1 - valid), memory.age * valid, valid)
