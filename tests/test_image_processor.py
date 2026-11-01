"""Unit tests for core image processing, conversions, and normalization."""

import unittest
from io import BytesIO
import numpy as np
from PIL import Image

from core.image_processor import (
    create_preview_thumbnail,
    preprocess_image,
)
from utils.image_utils import cv2_to_pil, pil_to_bytes, pil_to_cv2


class TestImageProcessor(unittest.TestCase):
    def test_preprocess_pil_image(self):
        pil_img = Image.new("RGB", (200, 150), color=(50, 100, 150))
        processed = preprocess_image(pil_img)
        self.assertEqual(processed.size, (200, 150))
        self.assertEqual(processed.mode, "RGB")

    def test_preprocess_numpy_array(self):
        # 3-channel RGB array
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 128]
        processed = preprocess_image(arr)
        self.assertEqual(processed.size, (100, 100))
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.getpixel((10, 10)), (255, 0, 128))

        # Grayscale 2D array
        gray_arr = np.ones((80, 80), dtype=np.uint8) * 120
        processed_gray = preprocess_image(gray_arr)
        self.assertEqual(processed_gray.size, (80, 80))
        self.assertEqual(processed_gray.mode, "RGB")

    def test_preprocess_bytes(self):
        img = Image.new("RGB", (120, 80), color=(10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        processed = preprocess_image(raw_bytes)
        self.assertEqual(processed.size, (120, 80))
        self.assertEqual(processed.mode, "RGB")

    def test_cv2_conversions(self):
        original = Image.new("RGB", (50, 50), color=(200, 100, 50))
        cv2_bgr = pil_to_cv2(original)
        self.assertEqual(cv2_bgr.shape, (50, 50, 3))
        # In BGR, blue is first: (50, 100, 200)
        self.assertEqual(tuple(cv2_bgr[0, 0]), (50, 100, 200))

        restored_pil = cv2_to_pil(cv2_bgr)
        self.assertEqual(restored_pil.getpixel((0, 0)), (200, 100, 50))

    def test_thumbnail_preserves_aspect_ratio(self):
        wide_img = Image.new("RGB", (800, 400), color=(0, 0, 0))
        thumb = create_preview_thumbnail(wide_img, size=(160, 160))
        self.assertEqual(thumb.size, (160, 80))

        tall_img = Image.new("RGB", (400, 800), color=(0, 0, 0))
        thumb_tall = create_preview_thumbnail(tall_img, size=(160, 160))
        self.assertEqual(thumb_tall.size, (80, 160))


if __name__ == "__main__":
    unittest.main()

