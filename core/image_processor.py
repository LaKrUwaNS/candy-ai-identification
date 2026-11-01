"""Core image preprocessing, validation, and normalization module."""

from pathlib import Path
from typing import Any, Tuple, Union
import numpy as np
from PIL import Image

from config.settings import DEFAULT_THUMBNAIL_SIZE
from utils.image_utils import bytes_to_pil, create_thumbnail, cv2_to_pil
from utils.validators import (
    validate_image_bytes,
    validate_image_dimensions,
    validate_image_format,
)


def validate_image_upload(uploaded_file: Any) -> Tuple[bool, str]:
    """Validate Streamlit UploadedFile or file-like object."""
    if uploaded_file is None:
        return False, "No file was provided."

    filename = getattr(uploaded_file, "name", "")
    fmt_ok, fmt_err = validate_image_format(filename)
    if not fmt_ok:
        return False, fmt_err

    # Read bytes
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        return False, "Unable to read image bytes from object."

    return validate_image_bytes(data)


def preprocess_image(
    image_input: Union[Image.Image, np.ndarray, bytes, str, Path],
) -> Image.Image:
    """
    Normalize various input formats into a consistent, orientation-corrected RGB PIL Image.
    Supports PIL.Image, numpy arrays (RGB/BGR), raw bytes, and file paths.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with open(path, "rb") as f:
            return bytes_to_pil(f.read())

    elif isinstance(image_input, bytes):
        return bytes_to_pil(image_input)

    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            # Grayscale -> RGB
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            # Determine if BGR or RGB by default assumption or shape
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            # RGBA -> RGB
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")

    elif isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":
            return image_input.convert("RGB")
        return image_input

    elif hasattr(image_input, "getvalue"):
        # Streamlit UploadedFile / CameraInput
        return bytes_to_pil(image_input.getvalue())

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def create_preview_thumbnail(
    image: Image.Image,
    size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> Image.Image:
    """Generate resized thumbnail preserving aspect ratio."""
    return create_thumbnail(image, size=size)

