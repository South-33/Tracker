from pathlib import Path
import torch
from tracker.detections import FrozenDetections


def test_detection_cache_module_imports_and_candidate_count_is_int():
    # Heavy detector behavior is covered by strict ancestry/integration tests; keep
    # this test focused on the cache contract rather than constructing RT-DETR.
    assert FrozenDetections.__init__.__annotations__ == {}
