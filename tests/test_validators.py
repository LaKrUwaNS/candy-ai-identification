"""Unit tests for parameter, image, and threshold validators."""

import unittest
from io import BytesIO
from PIL import Image

from utils.validators import (
    validate_image_bytes,
    validate_image_dimensions,
    validate_image_format,
    validate_thresholds,
)


class TestValidators(unittest.TestCase):
    def test_image_format_valid(self):
        valid_files = ["candy.jpg", "photo.JPEG", "scan.png", "item.webp"]
        for f in valid_files:
            ok, msg = validate_image_format(f)
            self.assertTrue(ok, f"Expected {f} to be valid: {msg}")

    def test_image_format_invalid(self):
        invalid_files = ["file.txt", "doc.pdf", "animation.gif", "script.py", ""]
        for f in invalid_files:
            ok, msg = validate_image_format(f)
            self.assertFalse(ok, f"Expected {f} to be rejected")
            self.assertGreater(len(msg), 0)

    def test_image_dimensions_bounds(self):
        # Valid dimensions
        ok, _ = validate_image_dimensions(640, 480)
        self.assertTrue(ok)

        # Too small
        ok, msg = validate_image_dimensions(5, 5)
        self.assertFalse(ok)
        self.assertIn("smaller than minimum", msg)

        # Too large
        ok, msg = validate_image_dimensions(9000, 9000)
        self.assertFalse(ok)
        self.assertIn("exceed maximum", msg)

    def test_image_bytes_empty_and_corrupt(self):
        # Empty bytes
        ok, msg = validate_image_bytes(b"")
        self.assertFalse(ok)
        self.assertIn("empty", msg)

        # Corrupted bytes
        ok, msg = validate_image_bytes(b"not_an_image_random_noise_data")
        self.assertFalse(ok)
        self.assertIn("Corrupted or invalid", msg)

        # Valid image bytes
        img = Image.new("RGB", (100, 100), color=(255, 128, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        ok, msg = validate_image_bytes(buf.getvalue())
        self.assertTrue(ok, f"Failed on valid image bytes: {msg}")

    def test_thresholds(self):
        # Valid defaults
        ok, _ = validate_thresholds(0.25, 0.45)
        self.assertTrue(ok)

        # Valid boundaries
        ok, _ = validate_thresholds(0.05, 0.10)
        self.assertTrue(ok)
        ok, _ = validate_thresholds(1.00, 0.95)
        self.assertTrue(ok)

        # Invalid confidence
        ok, msg = validate_thresholds(0.01, 0.45)
        self.assertFalse(ok)
        self.assertIn("Confidence", msg)

        ok, msg = validate_thresholds(1.05, 0.45)
        self.assertFalse(ok)
        self.assertIn("Confidence", msg)

        # Invalid IoU
        ok, msg = validate_thresholds(0.25, 0.05)
        self.assertFalse(ok)
        self.assertIn("IoU", msg)


if __name__ == "__main__":
    unittest.main()

