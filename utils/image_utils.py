"""Image format conversion and manipulation utilities."""

from io import BytesIO
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from config.settings import DEFAULT_THUMBNAIL_SIZE


def bytes_to_pil(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL Image with EXIF orientation correction."""
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encode PIL Image into in-memory bytes."""
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert RGB PIL Image to OpenCV BGR numpy array."""
    rgb_array = np.array(image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to RGB PIL Image."""
    rgb_array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def create_thumbnail(
    image: Image.Image,
    size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> Image.Image:
    """Generate high-quality aspect-ratio-preserved thumbnail."""
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb

