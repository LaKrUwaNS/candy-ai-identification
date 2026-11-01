"""Validation utilities for images, thresholds, and files in Candy AI."""

import os
from io import BytesIO
from typing import Tuple
from PIL import Image

from config.settings import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_CONFIDENCE,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IOU,
    MIN_CONFIDENCE,
    MIN_IMAGE_DIMENSION,
    MIN_IOU,
)


def validate_image_format(filename: str) -> Tuple[bool, str]:
    """Validate whether file extension is among allowed formats."""
    if not filename:
        return False, "Filename cannot be empty."
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        return False, f"Unsupported image extension '{ext}'. Supported formats: {allowed}."
    return True, ""


def validate_image_dimensions(width: int, height: int) -> Tuple[bool, str]:
    """Validate image resolution bounds."""
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return (
            False,
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}).",
        )
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return (
            False,
            f"Image dimensions ({width}x{height}) exceed maximum allowed ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}).",
        )
    return True, ""


def validate_image_bytes(data: bytes) -> Tuple[bool, str]:
    """Verify raw byte data integrity, size, and decodability."""
    if not data or len(data) == 0:
        return False, "Image file data is empty."
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        return False, f"File exceeds maximum allowed size ({max_mb} MB)."
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            dim_ok, dim_msg = validate_image_dimensions(img.width, img.height)
            if not dim_ok:
                return False, dim_msg
    except Exception as exc:
        return False, f"Corrupted or invalid image: {str(exc)}"
    return True, ""


def validate_thresholds(confidence: float, iou: float) -> Tuple[bool, str]:
    """Validate confidence and IoU threshold boundaries."""
    if not (MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE):
        return (
            False,
            f"Confidence threshold ({confidence}) must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}.",
        )
    if not (MIN_IOU <= iou <= MAX_IOU):
        return (
            False,
            f"IoU threshold ({iou}) must be between {MIN_IOU} and {MAX_IOU}.",
        )
    return True, ""

