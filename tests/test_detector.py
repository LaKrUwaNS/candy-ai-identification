"""Unit and integration tests for detection engine, data structures, and model loading."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch

from core.detector import (
    DetectionItem,
    DetectionResult,
    InferenceSpeed,
    detect_candies,
    draw_bounding_boxes,
)
from core.model_loader import (
    extract_class_mapping,
    get_compute_device,
    get_model,
    validate_class_mapping,
)
from utils.file_handler import check_model_exists, resolve_model_path


class TestDetectorStructures(unittest.TestCase):
    def test_detection_item(self):
        item = DetectionItem(
            detection_id=1,
            class_id=15,
            candy_category="snickers",
            confidence=0.9234,
            confidence_percent="92.3%",
            bbox=(10, 20, 110, 220),
            box_width=100,
            box_height=200,
            color_rgb=(120, 53, 15),
            color_hex="#78350f",
        )
        self.assertEqual(item.detection_id, 1)
        self.assertEqual(item.candy_category, "snickers")
        d_dict = item.to_dict()
        self.assertEqual(d_dict["box_width"], 100)
        self.assertEqual(d_dict["color_hex"], "#78350f")

    def test_inference_speed_calculation(self):
        speed = InferenceSpeed(preprocess_ms=2.5, inference_ms=120.3, postprocess_ms=3.2)
        self.assertEqual(speed.total_ms, 126.0)
        speed_dict = speed.to_dict()
        self.assertEqual(speed_dict["inference_ms"], 120.3)

    def test_detection_result_subscript_access(self):
        dummy_img = Image.new("RGB", (50, 50))
        speed = InferenceSpeed(1.0, 10.0, 1.0)
        res = DetectionResult(
            annotated_image=dummy_img,
            original_image=dummy_img,
            detections=[],
            candy_counts={"Mars": 1},
            total_candies=1,
            unique_categories=1,
            inference_speed=speed,
            class_mapping={0: "Mars"},
        )
        # Property access
        self.assertEqual(res.total_candies, 1)
        # Dict backwards-compatibility access
        self.assertEqual(res["total_candies"], 1)
        self.assertEqual(res["candy_counts"], {"Mars": 1})
        self.assertEqual(res.get("non_existent", "default"), "default")

    def test_draw_bounding_boxes_empty(self):
        img = Image.new("RGB", (100, 100), color=(10, 20, 30))
        annotated = draw_bounding_boxes(img, [])
        self.assertEqual(annotated.size, (100, 100))


class TestModelLoadingIntegration(unittest.TestCase):
    def test_model_file_exists_and_resolves(self):
        exists, msg = check_model_exists()
        self.assertTrue(exists, f"Model file check failed: {msg}")
        path = resolve_model_path()
        self.assertTrue(path.is_file(), f"Resolved path '{path}' is not a valid file.")

    def test_model_class_mapping_twenty_classes(self):
        model = get_model()
        self.assertIsNotNone(model)
        mapping = extract_class_mapping(model)
        self.assertEqual(len(mapping), 20, f"Expected 20 classes, got {len(mapping)}")
        valid, val_msg = validate_class_mapping(mapping)
        self.assertTrue(valid, val_msg)

    def test_detect_candies_blank_image(self):
        dummy_img = Image.new("RGB", (640, 640), color=(0, 0, 0))
        result = detect_candies(dummy_img, confidence=0.25, iou=0.45, imgsz=640)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.total_candies, 0)
        self.assertEqual(result.unique_categories, 0)
        self.assertEqual(len(result.detections), 0)


if __name__ == "__main__":
    unittest.main()

