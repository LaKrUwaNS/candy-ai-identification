"""Core computer vision and inference package for Candy AI."""

from core.candy_counter import (
    count_candies,
    format_ascii_report,
    format_candy_summary_str,
    get_top_candy,
)
from core.detector import (
    DetectionItem,
    DetectionResult,
    InferenceSpeed,
    detect_candies,
    draw_bounding_boxes,
)
from core.image_processor import (
    create_preview_thumbnail,
    preprocess_image,
    validate_image_upload,
)
from core.model_loader import (
    extract_class_mapping,
    get_compute_device,
    get_device_info,
    get_model,
    load_raw_model,
    validate_class_mapping,
)

__all__ = [
    "get_model",
    "load_raw_model",
    "get_compute_device",
    "get_device_info",
    "extract_class_mapping",
    "validate_class_mapping",
    "detect_candies",
    "draw_bounding_boxes",
    "DetectionItem",
    "DetectionResult",
    "InferenceSpeed",
    "preprocess_image",
    "validate_image_upload",
    "create_preview_thumbnail",
    "count_candies",
    "get_top_candy",
    "format_ascii_report",
    "format_candy_summary_str",
]

