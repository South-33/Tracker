import torch
from tracker.tracking_metrics import TrackingAccumulator


def box(x):
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 4)


def test_misses_do_not_break_strict_segment_but_switches_do():
    metric = TrackingAccumulator(.5)
    metric.update([1], box([[.2,.2,.1,.1]]), [10], box([[.2,.2,.1,.1]]))
    metric.update([1], box([[.3,.2,.1,.1]]), [], box([]))
    metric.update([1], box([[.4,.2,.1,.1]]), [11], box([[.4,.2,.1,.1]]))
    metric.update([], box([]), [], box([]))
    summary = metric.finalize()
    assert summary["recall"] == 2/3
    assert summary["identity_switches"] == 1
    assert summary["strict_visible_segment_survival"] == 0


def test_public_id_reuse_is_counted():
    metric = TrackingAccumulator(.5)
    metric.update([1], box([[.2,.2,.1,.1]]), [7], box([[.2,.2,.1,.1]]))
    metric.update([], box([]), [], box([]))
    metric.update([2], box([[.7,.7,.1,.1]]), [7], box([[.7,.7,.1,.1]]))
    summary = metric.finalize()
    assert summary["public_id_reuses"] == 1
    assert summary["strict_visible_segment_survival"] == .5
