import torch
from tracker.loss import assign, supervised_loss
from tracker.model import Prediction
from test_memory import memory


def predictions():
    boxes = torch.tensor([[[.8, .5, .1, .2], [.2, .5, .1, .2], [.5, .5, .1, .2], [.2, .5, .1, .2]]], requires_grad=True)
    return Prediction(boxes, torch.zeros(1, 4, requires_grad=True), torch.zeros(1, 4, 4),
                      torch.zeros(1, 4, requires_grad=True), torch.zeros(1, 4, requires_grad=True))


def test_crossed_boxes_do_not_rematch_resident_identities():
    prediction = predictions()
    ids = [101, 102, 103]
    boxes = torch.tensor([[.2, .5, .1, .2], [.8, .5, .1, .2], [.5, .5, .1, .2]])
    assignment, births = assign(prediction, ids, boxes, {0: 101, 1: 102}, 2)
    assert (0, 0) in assignment and (1, 1) in assignment
    assert births == [(2, 2)]


def test_missing_resident_is_retained_and_return_is_not_a_birth():
    prediction = predictions()
    boxes = torch.tensor([[.2, .5, .1, .2]])
    assignment, births = assign(prediction, [101], boxes, {0: 101, 1: 102}, 2)
    assert assignment == [(0, 0)] and births == []
    assignment, births = assign(prediction, [102], boxes, {0: 101, 1: 102}, 2)
    assert assignment == [(1, 0)] and births == []


def test_empty_ground_truth_still_penalizes_false_observations():
    prediction = predictions()
    state = memory()._replace(valid=torch.ones(1, 2, 1))
    loss, _ = supervised_loss(prediction, [], torch.empty(0, 4), state, 2)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert prediction.person_logits.grad.abs().sum() > 0
