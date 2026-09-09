"""Cache deterministic frozen image features on disk with a bounded GPU working set."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch

from .geometry import preprocess


class FrozenFeatures:
    def __init__(self, model, size, device, directory=None, cache_mib=512):
        self.model, self.size, self.device = model, size, device
        self.limit = max(0, int(cache_mib * 2**20)) if directory else 0
        self.items = OrderedDict()
        self.bytes = 0
        self.hits = self.disk_hits = self.encodes = 0
        for module in (model.detector.backbone, model.detector.encoder):
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("Feature caching requires frozen eval-mode backbone and encoder")
        self.directory = None
        if directory:
            digest = hashlib.sha256()
            settings = {"size": size, "torch": torch.__version__, "precision": "cuda_amp_fp16" if str(device).startswith("cuda") else "cpu_fp32",
                        "preprocess": "RGB-letterbox-114-rounded-scales-v1"}
            digest.update(json.dumps(settings, sort_keys=True).encode())
            for name, module in (("backbone", model.detector.backbone), ("encoder", model.detector.encoder)):
                for key, value in module.state_dict().items():
                    digest.update(f"{name}.{key}".encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            self.directory = Path(directory) / digest.hexdigest()[:24]
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    def get(self, sequence, number):
        image_path = sequence.directory / "img1" / f"{number:08d}.jpg"
        stat = image_path.stat()
        key = f"{sequence.name}-{number}-{stat.st_size}-{stat.st_mtime_ns}-{sequence.record['gt_sha256'][:16]}"
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        path = self.directory / f"{key}.pt" if self.directory else None
        if path and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            pyramid = [value.to(self.device) for value in payload["pyramid"]]
            ids = payload["ids"].numpy()
            target = payload["target"].to(self.device)
            self.disk_hits += 1
        else:
            rgb, ids, boxes = sequence.read(number)
            image, transform = preprocess(rgb, self.size, self.device)
            # no_grad rather than inference_mode: decoder parameter gradients need to save these tensors.
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                pyramid = [value.detach() for value in self.model.encode(image)]
            target = torch.as_tensor(transform.normalize_xywh(boxes), device=self.device)
            self.encodes += 1
            if path:
                payload = {"pyramid": [value.cpu() for value in pyramid], "ids": torch.from_numpy(ids), "target": target.cpu(),
                           "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
                temporary = path.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(path)
        result = (pyramid, ids, target)
        size = sum(value.numel() * value.element_size() for value in pyramid) + target.numel() * target.element_size() + ids.nbytes
        if size <= self.limit:
            while self.bytes + size > self.limit and self.items:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = (result, size)
            self.bytes += size
        return result

    def stats(self):
        return {"gpu_cache_hits": self.hits, "disk_cache_hits": self.disk_hits, "fresh_encodes": self.encodes,
                "resident_mib": self.bytes / 2**20, "limit_mib": self.limit / 2**20,
                "directory": str(self.directory) if self.directory else None}
