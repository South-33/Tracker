import numpy as np
import torch
from torch import nn
from tracker.model import Memory, Prediction, RecurrentTracker
from tracker.runtime import Runtime


class ScriptedNetwork(nn.Module):
    """A controlled observation stream exercises actual lifecycle logic without learned weights."""
    slots = 1

    def __init__(self):
        super().__init__()
        self.frame = 0

    def empty_memory(self):
        return Memory(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), torch.full((1, 1, 4), .5),
                      torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))

    def forward(self, image, recent, reliable, boxes, age, valid, dt):
        logits = torch.tensor([[-10., 10.]]) if self.frame in (0, 2) else torch.full((1, 2), -10.)
        self.frame += 1
        return Prediction(torch.full((1, 2, 4), .3), logits, torch.ones(1, 2, 4),
                          torch.full((1, 2), 10.), torch.full((1, 2), 10.))

    update_memory = RecurrentTracker.update_memory


def test_public_ids_do_not_repeat_after_expiry_and_hidden_frames_have_no_boxes():
    runner = Runtime(ScriptedNetwork(), size=32, maximum_age=1)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    first = runner.step(image, .1)
    hidden = runner.step(image, .2)
    second = runner.step(image, 1.1)
    assert first[0]["id"] == 1
    assert hidden == []
    assert second[0]["id"] == 2
    assert runner.memory.recent.grad_fn is None


def test_capacity_is_bounded():
    network = ScriptedNetwork()
    runner = Runtime(network, size=32)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    runner.step(image, .1)
    network.frame = 2
    assert runner.step(image, .1) == []
    assert runner.capacity_rejections == 1
    assert len(runner.identities) == 1
