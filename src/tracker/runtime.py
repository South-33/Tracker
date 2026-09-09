"""Anonymous ID lifecycle around one neural call. No matching/re-identification in this wrapper."""
import uuid
import torch

from .geometry import preprocess
from .model import expire_memory, fill_slot


class Runtime:
    def __init__(self, model, size=640, person_threshold=.4, birth_threshold=.6, continuity_threshold=.5, maximum_age=10.):
        self.model = model.eval()
        self.size = size
        self.person_threshold = person_threshold
        self.birth_threshold = birth_threshold
        self.continuity_threshold = continuity_threshold
        self.maximum_age = maximum_age
        self.reset()

    def reset(self):
        self.memory = self.model.empty_memory()
        self.identities = {}
        self.next_identity = 1
        self.run_id = str(uuid.uuid4())
        self.capacity_rejections = 0

    @torch.inference_mode()
    def step(self, rgb, dt_seconds):
        if not 0 < dt_seconds < float("inf"):
            raise ValueError("dt_seconds must be finite and positive")
        device = self.memory.recent.device
        image, transform = preprocess(rgb, self.size, device)
        dt = torch.tensor([dt_seconds], device=device)
        self.memory = expire_memory(self.memory, self.maximum_age - dt_seconds)
        valid = self.memory.valid[0, :, 0].cpu().bool().tolist()
        self.identities = {slot: identity for slot, identity in self.identities.items() if valid[slot]}
        enabled = device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
            prediction = self.model(image, *self.memory, dt)
            scores = prediction.person_logits[0].sigmoid().float().cpu()
            continuity = prediction.continuity_logits[0].sigmoid().float().cpu()
            observed = torch.tensor([valid[slot] and scores[slot] >= self.person_threshold and continuity[slot] >= self.continuity_threshold
                                     for slot in range(self.model.slots)], device=device)
            self.memory = self.model.update_memory(self.memory, prediction, observed, dt)
            output = [(slot, self.identities[slot], float(continuity[slot])) for slot in self.identities if observed[slot]]
            # Ranking decides which births fit in capacity; it does not reconnect an old identity.
            for relative in torch.argsort(scores[self.model.slots:], descending=True).tolist():
                query = relative + self.model.slots
                if scores[query] < self.birth_threshold:
                    break
                available = [slot for slot in range(self.model.slots) if slot not in self.identities]
                if not available:
                    self.capacity_rejections += 1
                    continue
                slot = available[0]
                identity = self.next_identity
                self.next_identity += 1
                self.identities[slot] = identity
                self.memory = fill_slot(self.memory, slot, prediction.features[0, query], prediction.boxes[0, query])
                output.append((query, identity, None))
        self.memory = self.memory.detached()
        boxes = transform.restore_xyxy(prediction.boxes[0].float().cpu().numpy())
        return [{"id": identity, "box": boxes[query].tolist(), "person_score": float(scores[query]),
                 "continuity_score": confidence, "status": "new" if confidence is None else "continued"}
                for query, identity, confidence in output]
