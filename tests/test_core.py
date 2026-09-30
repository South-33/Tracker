import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tracker.data import DanceTrackPairs, letterbox, restore_boxes
from tracker.model import identity_retrieval_loss


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


class IdentityLossTests(unittest.TestCase):
    def test_perfect_embeddings_retrieve_the_same_identity(self):
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        first_ids = torch.tensor([10, 20])
        second_ids = torch.tensor([20, 10])

        loss, report = identity_retrieval_loss(
            first,
            first_ids,
            second,
            second_ids,
            temperature=1.0,
        )

        self.assertLess(loss.item(), 0.32)
        self.assertEqual(report["top1_accuracy"], 1.0)
        self.assertEqual(report["positive_cosine"], 1.0)
        self.assertEqual(report["hard_negative_cosine"], 0.0)

    def test_pair_without_shared_detector_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            identity_retrieval_loss(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([1]),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([2]),
            )


class DanceTrackDataTests(unittest.TestCase):
    def test_incomplete_sequence_fails_during_dataset_setup(self):
        with TemporaryDirectory() as directory:
            sequence = Path(directory) / "dancetrack0001"
            (sequence / "img1").mkdir(parents=True)
            (sequence / "gt").mkdir()
            (sequence / "img1" / "00000001.jpg").touch()
            (sequence / "gt" / "gt.txt").write_text(
                "1,1,0,0,10,10,1,1,1\n", encoding="utf-8"
            )
            (sequence / "seqinfo.ini").write_text(
                "[Sequence]\nframeRate=30\nseqLength=2\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                DanceTrackPairs(directory, ["dancetrack0001"])


if __name__ == "__main__":
    unittest.main()
