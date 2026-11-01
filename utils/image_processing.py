"""Image processing utilities for Candy AI.
Handles image loading, validation, format conversions, and thumbnails.
"""

from io import BytesIO
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import cv2

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def validate_image_extension(filename: str) -> Tuple[bool, str]:
    """Validate file extension against supported formats."""
    if not filename:
        return False, "Filename is empty."
    import os
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_FORMATS:
        return (
            False,
            f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )
    return True, ""


def validate_image_bytes(data: bytes) -> Tuple[bool, str]:
    """Validate image bytes for integrity and dimensions."""
    if not data or len(data) == 0:
        return False, "Uploaded image file is empty."
    if len(data) > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB).",
        )
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        # Re-open after verify() to test decoding
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            if width < 10 or height < 10:
                return False, f"Image dimensions ({width}x{height}) are too small."
            if width > 8192 or height > 8192:
                return False, f"Image dimensions ({width}x{height}) exceed maximum allowed limit."
    except Exception as exc:
        return False, f"Corrupted or invalid image file: {str(exc)}"
    return True, ""


def validate_image(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file object from Streamlit."""
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = getattr(uploaded_file, "name", "")
    ext_valid, ext_err = validate_image_extension(filename)
    if not ext_valid:
        return False, ext_err

    bytes_data = uploaded_file.getvalue()
    return validate_image_bytes(bytes_data)


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Load and normalize uploaded image into RGB PIL Image."""
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data))
    # Correct orientation based on EXIF tag if present
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert RGB PIL Image to OpenCV BGR ndarray."""
    rgb_arr = np.array(image)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr


def cv2_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to RGB PIL Image."""
    rgb_arr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_arr)


def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (160, 160)) -> Image.Image:
    """Create an aspect-ratio-preserved thumbnail for session storage."""
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb

