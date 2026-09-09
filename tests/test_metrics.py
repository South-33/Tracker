"""The patched official metrics must penalize an identity swap with perfect boxes."""
import sys
import numpy as np
import pytest
from tracker.detector import ROOT


def test_official_identity_metrics_penalize_a_swap():
    sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
    import trackeval
    base = {"num_timesteps": 4, "num_gt_ids": 2, "num_tracker_ids": 2,
            "num_gt_dets": 8, "num_tracker_dets": 8,
            "gt_ids": [np.array([0, 1])] * 4,
            "tracker_ids": [np.array([0, 1])] * 4,
            "similarity_scores": [np.eye(2)] * 4}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.Identity({"PRINT_CONFIG": False}), trackeval.metrics.CLEAR({"PRINT_CONFIG": False})]
    good = [metric.eval_sequence(base) for metric in metrics]
    swapped = {**base, "tracker_ids": [np.array([0, 1])] * 2 + [np.array([1, 0])] * 2}
    bad = [metric.eval_sequence(swapped) for metric in metrics]
    assert good[0]["HOTA"].mean() == pytest.approx(1.)
    assert good[1]["IDF1"] == pytest.approx(1.)
    assert good[2]["IDSW"] == 0
    assert bad[1]["IDF1"] == pytest.approx(.5)
    assert bad[2]["IDSW"] == 2
    assert bad[0]["HOTA"].mean() < good[0]["HOTA"].mean()
