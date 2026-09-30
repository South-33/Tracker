import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from types import SimpleNamespace
from ultralytics.engine.results import Boxes

from tracker.association import OwnerContinuityScorer, choose_guarded_assignment
from tracker.causal import CausalPersonTracker
from tracker.data import DanceTrackPairs, letterbox, restore_boxes
from tracker.model import identity_retrieval_loss
from tracker.runtime import CausalTrackerRuntime
from tracker.splits import (
    CALIBRATION,
    CONSUMED_HOLDOUT,
    DEV,
    RESERVED_HOLDOUT,
    TRAIN,
    assert_research_eval,
    assert_train_only,
    role_of,
)
from tracker.system import TrackerConfig


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
            temperature=0.1,
        )

        self.assertLess(loss.item(), 0.001)
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


class GuardedAssignmentTests(unittest.TestCase):
    def test_tiebreak_respects_base_cost_budget(self):
        base = np.array([[0.20, 0.21], [0.21, 0.20]], dtype=np.float32)
        logits = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)

        strict, *_ = choose_guarded_assignment(
            base,
            logits,
            match_threshold=0.8,
            alpha=0.2,
            average_base_cost_budget=0.005,
        )
        relaxed, *_ = choose_guarded_assignment(
            base,
            logits,
            match_threshold=0.8,
            alpha=0.2,
            average_base_cost_budget=0.02,
        )

        self.assertEqual({tuple(row) for row in strict}, {(0, 0), (1, 1)})
        self.assertEqual({tuple(row) for row in relaxed}, {(0, 1), (1, 0)})

    def test_tiebreak_cannot_introduce_invalid_edge(self):
        base = np.array([[0.20, 0.81], [0.81, 0.20]], dtype=np.float32)
        logits = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=np.float32)

        matches, *_ = choose_guarded_assignment(
            base,
            logits,
            match_threshold=0.8,
            alpha=1.0,
            average_base_cost_budget=1.0,
        )

        self.assertEqual({tuple(row) for row in matches}, {(0, 0), (1, 1)})


class CausalRuntimeTests(unittest.TestCase):
    def test_rejects_invalid_feature_mode(self):
        with self.assertRaises(ValueError):
            CausalTrackerRuntime(
                SimpleNamespace(),
                SimpleNamespace(),
                device=torch.device("cpu"),
                with_reid=True,
                feature_mode="invalid",
            )

    def test_starts_with_empty_causal_counters(self):
        runtime = CausalTrackerRuntime(
            SimpleNamespace(),
            SimpleNamespace(),
            device=torch.device("cpu"),
            with_reid=False,
        )
        self.assertEqual(
            runtime.stats(),
            {
                "frames": 0,
                "reid_frames": 0,
                "reid_detections": 0,
                "compute_seconds": 0.0,
                "owner_tiebreak_frames": 0,
                "owner_changed_frames": 0,
            },
        )

    def test_rejects_invalid_detector_confidence(self):
        with self.assertRaises(ValueError):
            CausalTrackerRuntime(
                SimpleNamespace(),
                SimpleNamespace(),
                device=torch.device("cpu"),
                with_reid=False,
                detector_confidence=1.1,
            )


class SplitSafetyTests(unittest.TestCase):
    def test_roles_are_disjoint(self):
        self.assertFalse(set(TRAIN) & set(DEV))
        self.assertFalse(set(TRAIN) & set(CALIBRATION))
        self.assertFalse(set(TRAIN) & set(CONSUMED_HOLDOUT))
        self.assertFalse(set(DEV) & set(RESERVED_HOLDOUT))

    def test_holdout_cannot_be_used_for_training(self):
        with self.assertRaises(ValueError):
            assert_train_only([CONSUMED_HOLDOUT[0]])

    def test_role_labels_are_explicit(self):
        self.assertEqual(role_of(TRAIN[0]), "train")
        self.assertEqual(role_of(CALIBRATION[0]), "calibration")
        self.assertEqual(role_of(DEV[0]), "dev")
        self.assertEqual(role_of(CONSUMED_HOLDOUT[0]), "consumed_holdout")
        self.assertEqual(role_of(RESERVED_HOLDOUT[0]), "reserved_holdout")

    def test_research_eval_rejects_holdout(self):
        with self.assertRaises(ValueError):
            assert_research_eval([RESERVED_HOLDOUT[0]])


class TrackerConfigTests(unittest.TestCase):
    def test_current_candidate_defaults_are_explicit(self):
        config = TrackerConfig()
        self.assertEqual(config.track_low_thresh, 0.05)
        self.assertEqual(config.new_track_thresh, 0.45)
        self.assertEqual(config.detector_confidence, 0.05)
        self.assertEqual(config.track_buffer, 30)
        self.assertEqual(config.gmc_method, "sparseOptFlow")
        self.assertEqual(config.gmc_max_corners, 100)
        self.assertEqual(config.min_detections_for_reid, 10)
        self.assertEqual(config.owner_average_base_cost_budget, 0.00025)


class CausalPersonTrackerTests(unittest.TestCase):
    def test_keeps_identity_across_simple_two_frame_track(self):
        with TemporaryDirectory() as directory:
            owner_path = Path(directory) / "owner.pt"
            scorer = OwnerContinuityScorer()
            torch.save(
                {
                    "pair": scorer.state_dict(),
                    "mean": torch.zeros(11),
                    "std": torch.ones(11),
                },
                owner_path,
            )
            config = SimpleNamespace(
                track_buffer=30,
                gmc_method=None,
                proximity_thresh=0.5,
                appearance_thresh=0.8,
                with_reid=False,
                model="auto",
                device="cpu",
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                fuse_score=True,
                match_thresh=0.8,
                new_track_thresh=0.45,
            )
            tracker = CausalPersonTracker(config, owner_path)

            first = Boxes(
                torch.tensor([[10.0, 10.0, 30.0, 40.0, 0.9, 0.0]]),
                (100, 100),
            ).numpy()
            second = Boxes(
                torch.tensor([[11.0, 10.0, 31.0, 40.0, 0.9, 0.0]]),
                (100, 100),
            ).numpy()

            first_output = tracker.update(first)
            second_output = tracker.update(second)

            self.assertEqual(first_output.shape, (1, 8))
            self.assertEqual(second_output.shape, (1, 8))
            self.assertEqual(int(first_output[0, 4]), int(second_output[0, 4]))

    def test_sparse_flow_uses_reduced_corner_budget(self):
        with TemporaryDirectory() as directory:
            owner_path = Path(directory) / "owner.pt"
            scorer = OwnerContinuityScorer()
            torch.save(
                {
                    "pair": scorer.state_dict(),
                    "mean": torch.zeros(11),
                    "std": torch.ones(11),
                },
                owner_path,
            )
            config = SimpleNamespace(
                track_buffer=30,
                gmc_method="sparseOptFlow",
                proximity_thresh=0.5,
                appearance_thresh=0.8,
                with_reid=False,
                model="auto",
                device="cpu",
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                fuse_score=True,
                match_thresh=0.8,
                new_track_thresh=0.45,
            )
            tracker = CausalPersonTracker(config, owner_path)
            self.assertEqual(tracker.gmc.feature_params["maxCorners"], 100)

            with self.assertRaises(ValueError):
                CausalPersonTracker(config, owner_path, gmc_max_corners=4)


if __name__ == "__main__":
    unittest.main()
