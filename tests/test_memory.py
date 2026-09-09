import torch
from tracker.model import Memory, fill_slot, expire_memory
from tracker.geometry import preprocess
import numpy as np


def memory(slots=2, dim=4):
    return Memory(torch.zeros(1, slots, dim), torch.zeros(1, slots, dim),
                  torch.full((1, slots, 4), .5), torch.zeros(1, slots, 1), torch.zeros(1, slots, 1))


def test_slot_reuse_clears_appearance_and_retains_gradient():
    old = fill_slot(memory(), 0, torch.ones(4), torch.ones(4) * .2)
    new_feature = torch.full((4,), 3., requires_grad=True)
    new = fill_slot(old, 0, new_feature, torch.ones(4) * .7)
    assert torch.equal(new.reliable[0, 0], new_feature)
    assert new.recent[0, 1].sum() == 0
    new.reliable.sum().backward()
    assert torch.equal(new_feature.grad, torch.ones(4))


def test_expiry_clears_old_occupant():
    state = fill_slot(memory(), 0, torch.ones(4), torch.ones(4) * .2)
    state = state._replace(age=torch.tensor([[[10.1], [0.]]]))
    expired = expire_memory(state)
    assert expired.valid.sum() == 0
    assert expired.recent.sum() == 0
    assert torch.all(expired.boxes == .5)


def test_letterbox_round_trip_uses_actual_rounded_scales():
    image = np.zeros((721, 1281, 3), dtype=np.uint8)
    _, transform = preprocess(image)
    boxes = np.array([[112, 58, 97, 204], [0, 0, 1281, 721]], dtype=np.float32)
    normalized = transform.normalize_xywh(boxes)
    actual = transform.restore_xyxy(normalized)
    expected = boxes.copy()
    expected[:, 2:] += expected[:, :2]
    np.testing.assert_allclose(actual, expected, atol=.001)
