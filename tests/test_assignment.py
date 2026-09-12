from typing import NamedTuple

import torch

from tracker.loss import assign


class Prediction(NamedTuple):
    boxes: torch.Tensor
    person_logits: torch.Tensor


def predictions():
    return Prediction(
        torch.tensor([[[.8, .5, .1, .2], [.2, .5, .1, .2],
                       [.5, .5, .1, .2], [.2, .5, .1, .2]]]),
        torch.zeros(1, 4),
    )


def test_crossed_boxes_do_not_rematch_resident_identities():
    ids = [101, 102, 103]
    boxes = torch.tensor([[.2, .5, .1, .2], [.8, .5, .1, .2], [.5, .5, .1, .2]])
    assignment, births = assign(predictions(), ids, boxes, {0: 101, 1: 102}, 2)
    assert (0, 0) in assignment and (1, 1) in assignment
    assert births == [(2, 2)]


def test_separate_birth_supervises_every_current_person():
    prediction = Prediction(
        torch.tensor([[[.2, .5, .1, .2], [.8, .5, .1, .2], [.5, .5, .1, .2]]]),
        torch.zeros(1, 3),
    )
    ids = [101, 102, 103]
    boxes = torch.tensor([[.2, .5, .1, .2], [.8, .5, .1, .2], [.5, .5, .1, .2]])
    assignment, births = assign(prediction, ids, boxes, {}, 0, separate_birth=True)
    assert {target for _, target in assignment} == {0, 1, 2}
    assert {target for _, target in births} == {0, 1, 2}
