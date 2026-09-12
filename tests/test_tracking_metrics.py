import torch
import pytest
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
    assert summary["covered_identity_segment_rate"] == 0


def test_silence_has_zero_covered_identity_success():
    metric = TrackingAccumulator()
    for _ in range(10):
        metric.update([1], box([[.2,.2,.1,.1]]), [], box([]))
    summary = metric.finalize()
    assert summary["recall"] == 0
    assert summary["covered_identity_segment_rate"] == 0
    assert summary["covered_identity_frame_rate"] == 0
    assert summary["untracked_segments"] == 1


@pytest.mark.parametrize("hits,expected", [(1, 0), (7, 0), (8, 1), (10, 1)])
def test_coverage_threshold_requires_sustained_tracking(hits, expected):
    metric = TrackingAccumulator()
    for frame in range(10):
        metric.update([1], box([[.2,.2,.1,.1]]), [10] if frame < hits else [],
                      box([[.2,.2,.1,.1]]) if frame < hits else box([]))
    result = metric.finalize()
    assert result["covered_identity_segment_rate"] == expected
    assert result["covered_identity_frame_rate"] == expected
    assert metric.finalize() == result


def test_perfect_coverage_with_switch_is_not_success():
    metric = TrackingAccumulator()
    for public in [10, 10, 11, 11]:
        metric.update([1], box([[.2,.2,.1,.1]]), [public], box([[.2,.2,.1,.1]]))
    result = metric.finalize()
    assert result["recall"] == 1
    assert result["covered_identity_segment_rate"] == 0


def test_long_missed_person_is_visible_in_both_denominators():
    metric = TrackingAccumulator()
    metric.update([1], box([[.2,.2,.1,.1]]), [10], box([[.2,.2,.1,.1]]))
    for _ in range(9):
        metric.update([2], box([[.7,.7,.1,.1]]), [], box([]))
    result = metric.finalize()
    assert result["covered_identity_segment_rate"] == .5
    assert result["covered_identity_frame_rate"] == .1


def test_reentry_identity_change_fails_returned_segment():
    metric = TrackingAccumulator()
    metric.update([1], box([[.2,.2,.1,.1]]), [10], box([[.2,.2,.1,.1]]))
    metric.update([], box([]), [], box([]))
    metric.update([1], box([[.2,.2,.1,.1]]), [11], box([[.2,.2,.1,.1]]))
    assert metric.finalize()["covered_identity_segment_rate"] == .5


def test_empty_sequence_is_not_perfect():
    result = TrackingAccumulator().finalize()
    assert result["covered_identity_segment_rate"] is None


def test_below_threshold_pairs_cannot_steal_valid_match(monkeypatch):
    import tracker.tracking_metrics as metrics
    monkeypatch.setattr(metrics, "box_iou", lambda *args: torch.tensor([[.51,.49],[.49,0.]]))
    metric = TrackingAccumulator()
    boxes = box([[.2,.2,.1,.1], [.7,.7,.1,.1]])
    metric.update([1, 2], boxes, [10, 20], boxes)
    assert metric.finalize()["true_positives"] == 1


def test_mot_scoring_keeps_unreported_frames_in_denominator():
    import numpy as np
    from tracker.tracking_metrics import score_mot_rows
    truth = np.array([[frame, 1, 10, 20, 30, 40] for frame in range(1, 11)])
    predictions = np.array([[1, 7, 10, 20, 30, 40]])
    result = score_mot_rows(truth, predictions, 10)
    assert result["recall"] == .1
    assert result["covered_identity_segment_rate"] == 0


def test_mot_scoring_rejects_predictions_outside_the_run():
    import numpy as np
    from tracker.tracking_metrics import score_mot_rows
    truth = np.array([[1, 1, 10, 20, 30, 40]])
    with pytest.raises(ValueError, match="outside"):
        score_mot_rows(truth, np.array([[2, 7, 10, 20, 30, 40]]), 1)
