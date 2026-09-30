import unittest

import torch

from tracker.data import xyxy_to_normalized_cxcywh
from tracker.model import TemporalSlotHead, temporal_slot_loss
from tracker.splits import (
    BENCHMARK_ONLY_DATASETS,
    CALIBRATION,
    CONSUMED_HOLDOUT,
    DEV,
    RESERVED_HOLDOUT,
    TRAIN,
    TRAIN_DATASETS,
    assert_dataset_can_train,
    assert_research_eval,
    assert_train_only,
)


class TemporalSlotHeadTests(unittest.TestCase):
    def test_bounded_memory_and_output_shapes(self):
        head = TemporalSlotHead(
            (16, 32, 64),
            slots=12,
            memory_dim=32,
            attention_heads=4,
            pool_sizes=(8, 4, 2),
        )
        pyramid = (
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
        )
        first = head(pyramid)
        self.assertEqual(first["alive_logits"].shape, (2, 12))
        self.assertEqual(first["boxes_cxcywh"].shape, (2, 12, 4))
        self.assertEqual(first["memory"].shape, (2, 12, 32))

        second = head(pyramid, first["memory"])
        self.assertEqual(second["memory"].shape, first["memory"].shape)
        self.assertFalse(torch.equal(second["memory"], first["memory"]))

    def test_memory_shape_is_strict(self):
        head = TemporalSlotHead(
            (8, 16, 32),
            slots=4,
            memory_dim=16,
            attention_heads=4,
            pool_sizes=(4, 2, 1),
        )
        pyramid = (
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 16, 4, 4),
            torch.randn(1, 32, 2, 2),
        )
        with self.assertRaises(ValueError):
            head(pyramid, torch.zeros(1, 5, 16))


class TemporalLossTests(unittest.TestCase):
    def test_sequence_loss_is_finite_and_differentiable(self):
        alive0 = torch.zeros(1, 4, requires_grad=True)
        alive1 = torch.zeros(1, 4, requires_grad=True)
        box0 = torch.full((1, 4, 4), 0.5, requires_grad=True)
        box1 = torch.full((1, 4, 4), 0.5, requires_grad=True)
        outputs = [
            {
                "alive_logits": alive0,
                "boxes_cxcywh": box0,
                "memory": torch.zeros(1, 4, 8),
            },
            {
                "alive_logits": alive1,
                "boxes_cxcywh": box1,
                "memory": torch.zeros(1, 4, 8),
            },
        ]
        boxes = [[
            torch.tensor([[0.3, 0.4, 0.2, 0.3]]),
            torch.tensor([[0.32, 0.4, 0.2, 0.3]]),
        ]]
        identities = [[
            torch.tensor([7]),
            torch.tensor([7]),
        ]]
        loss, report = temporal_slot_loss(outputs, boxes, identities)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(report["visible_targets"], 2.0)
        loss.backward()
        self.assertIsNotNone(alive0.grad)
        self.assertIsNotNone(box0.grad)


class DataTests(unittest.TestCase):
    def test_xyxy_conversion(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
        converted = xyxy_to_normalized_cxcywh(boxes, 100)
        expected = torch.tensor([[0.2, 0.4, 0.2, 0.4]])
        self.assertTrue(torch.allclose(converted, expected))


class SplitSafetyTests(unittest.TestCase):
    def test_training_and_benchmark_datasets_are_disjoint(self):
        self.assertFalse(
            set(TRAIN_DATASETS) & set(BENCHMARK_ONLY_DATASETS)
        )

    def test_benchmark_dataset_cannot_train(self):
        with self.assertRaises(ValueError):
            assert_dataset_can_train("mot17")

    def test_training_is_separate_from_dev_and_holdout(self):
        self.assertFalse(set(TRAIN) & set(CALIBRATION))
        self.assertFalse(set(TRAIN) & set(DEV))
        self.assertFalse(set(TRAIN) & set(CONSUMED_HOLDOUT))
        self.assertFalse(set(TRAIN) & set(RESERVED_HOLDOUT))

    def test_train_guard_rejects_non_training_sequence(self):
        with self.assertRaises(ValueError):
            assert_train_only([DEV[0]])

    def test_research_guard_rejects_consumed_holdout(self):
        with self.assertRaises(ValueError):
            assert_research_eval([CONSUMED_HOLDOUT[0]])


if __name__ == "__main__":
    unittest.main()
