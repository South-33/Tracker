from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from tracker.features import FrozenFeatures


class FrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Module()
        self.detector.backbone = nn.Linear(1, 1).requires_grad_(False).eval()
        self.detector.encoder = nn.Linear(1, 1).requires_grad_(False).eval()
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        return [image * self.detector.backbone.weight.reshape(1, 1, 1, 1)]


def sequence(tmp_path):
    directory = tmp_path / "sequence"
    (directory / "img1").mkdir(parents=True)
    (directory / "img1/00000001.jpg").write_bytes(b"fixture")
    return SimpleNamespace(name="fixture", directory=directory, record={"gt_sha256": "frozen-test-labels"},
        read=lambda number: (np.full((16, 16, 3), 128, np.uint8), np.array([1], np.int64), np.array([[0, 0, 16, 16]], np.float32)))


def test_disk_features_are_exact_and_invalidate_for_new_weights(tmp_path):
    model, source = FrozenModel(), sequence(tmp_path)
    cache = FrozenFeatures(model, 16, "cpu", tmp_path / "cache", cache_mib=0)
    first = cache.get(source, 1)
    second = cache.get(source, 1)
    assert torch.equal(first[0][0], second[0][0])
    assert torch.equal(first[2], second[2])
    assert model.calls == 1 and cache.disk_hits == 1 and cache.bytes == 0
    with torch.no_grad():
        model.detector.backbone.weight.add_(1)
    changed = FrozenFeatures(model, 16, "cpu", tmp_path / "cache", cache_mib=0)
    assert changed.directory != cache.directory
    changed.get(source, 1)
    assert model.calls == 2


def test_cache_rejects_a_trainable_encoder():
    model = FrozenModel()
    model.detector.encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenFeatures(model, 16, "cpu")
