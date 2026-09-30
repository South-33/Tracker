import unittest

import numpy as np
import torch

from tracker.data import letterbox, restore_boxes
from tracker.model import pair_geometry


class LetterboxTests(unittest.TestCase):
    def test_boxes_round_trip(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        source_xywh = [[20.0, 10.0, 50.0, 40.0]]

        _, boxed = letterbox(image, source_xywh, size=640)
        restored = restore_boxes(boxed, image.shape[:2], size=640)

        expected_xyxy = torch.tensor([[20.0, 10.0, 70.0, 50.0]])
        self.assertTrue(torch.allclose(restored, expected_xyxy, atol=1e-4))

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            letterbox(None, [])


class GeometryTests(unittest.TestCase):
    def test_constant_velocity_prediction_hits_current_box(self):
        memory = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        current = torch.tensor([[10.0, 0.0, 20.0, 10.0]])
        velocity = torch.tensor([[10.0, 0.0, 10.0, 0.0]])

        features = pair_geometry(
            memory,
            velocity,
            torch.tensor([1.0]),
            current,
            gap_seconds=1.0,
        )

        self.assertEqual(tuple(features.shape), (1, 1, 10))
        self.assertAlmostEqual(features[0, 0, 8].item(), 1.0, places=6)
        self.assertAlmostEqual(features[0, 0, 6].item(), 0.0, places=6)
        self.assertAlmostEqual(features[0, 0, 7].item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
