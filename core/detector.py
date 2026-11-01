"""Core object detection engine for Candy AI.
Manages YOLO inference, bounding box extraction, and custom visualization.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from config.settings import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    get_color_for_class,
    get_hex_for_class,
)
from core.candy_counter import count_candies
from core.model_loader import extract_class_mapping, get_compute_device, get_model
from utils.helpers import format_percentage


@dataclass
class DetectionItem:
    """Individual candy detection object localized within an image."""
    detection_id: int
    class_id: int
    candy_category: str
    confidence: float
    confidence_percent: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    box_width: int
    box_height: int
    color_rgb: Tuple[int, int, int]
    color_hex: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection item to dictionary representation."""
        return asdict(self)


@dataclass
class InferenceSpeed:
    """Inference execution latency benchmarks in milliseconds."""
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        """Total latency from pre-processing to post-processing."""
        return round(self.preprocess_ms + self.inference_ms + self.postprocess_ms, 1)

    def to_dict(self) -> Dict[str, float]:
        """Convert inference speed metrics to dictionary."""
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class DetectionResult:
    """Complete result package produced by YOLO candy detection inference."""
    annotated_image: Image.Image
    original_image: Image.Image
    detections: List[DetectionItem]
    candy_counts: Dict[str, int]
    total_candies: int
    unique_categories: int
    inference_speed: InferenceSpeed
    class_mapping: Dict[int, str]

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-style subscript access for backwards compatibility."""
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "inference_speed" and isinstance(val, InferenceSpeed):
                return val.to_dict()
            if key == "detections" and isinstance(val, list):
                # Return list of dicts if accessed as dict for backwards compatibility
                return [d.to_dict() if isinstance(d, DetectionItem) else d for d in val]
            return val
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get method."""
        try:
            return self[key]
        except KeyError:
            return default


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[DetectionItem],
) -> Image.Image:
    """
    Render high-contrast, professional bounding boxes with corner accents
    and category label pills onto the image.
    """
    if not detections:
        return image.copy()

    # Convert to OpenCV BGR
    rgb_arr = np.array(image.convert("RGB"))
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_arr.shape[:2]

    # Dynamic scaling based on image dimension
    base_dim = max(img_h, img_w)
    box_thickness = max(2, int(round(base_dim / 350.0)))
    font_scale = max(0.45, base_dim / 1100.0)
    font_thickness = max(1, int(round(box_thickness * 0.7)))

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        rgb_color = det.color_rgb
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Main bounding rectangle
        cv2.rectangle(bgr_arr, (x1, y1), (x2, y2), bgr_color, box_thickness)

        # Corner accent brackets
        corner_len = max(8, int(min(x2 - x1, y2 - y1) * 0.2))
        corner_thick = box_thickness + 1
        # Top-left
        cv2.line(bgr_arr, (x1, y1), (x1 + corner_len, y1), bgr_color, corner_thick)
        cv2.line(bgr_arr, (x1, y1), (x1, y1 + corner_len), bgr_color, corner_thick)
        # Top-right
        cv2.line(bgr_arr, (x2, y1), (x2 - corner_len, y1), bgr_color, corner_thick)
        cv2.line(bgr_arr, (x2, y1), (x2, y1 + corner_len), bgr_color, corner_thick)
        # Bottom-left
        cv2.line(bgr_arr, (x1, y2), (x1 + corner_len, y2), bgr_color, corner_thick)
        cv2.line(bgr_arr, (x1, y2), (x1, y2 - corner_len), bgr_color, corner_thick)
        # Bottom-right
        cv2.line(bgr_arr, (x2, y2), (x2 - corner_len, y2), bgr_color, corner_thick)
        cv2.line(bgr_arr, (x2, y2), (x2, y2 - corner_len), bgr_color, corner_thick)

        # Label Pill
        label_text = f"#{det.detection_id} {det.candy_category} ({det.confidence_percent})"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        padding = 4
        pill_h = text_h + baseline + padding * 2
        pill_w = text_w + padding * 2

        # Position pill above box if room, otherwise inside top of box
        if y1 - pill_h >= 0:
            pill_y1 = y1 - pill_h
            pill_y2 = y1
        else:
            pill_y1 = y1
            pill_y2 = y1 + pill_h

        pill_x1 = max(0, x1)
        pill_x2 = min(img_w, pill_x1 + pill_w)

        # Draw filled pill background
        cv2.rectangle(bgr_arr, (pill_x1, pill_y1), (pill_x2, pill_y2), bgr_color, -1)

        # Determine contrast text color
        luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
        text_color = (255, 255, 255) if luminance < 140 else (20, 20, 20)

        # Draw text inside pill
        text_y = pill_y2 - baseline - padding + 2
        cv2.putText(
            bgr_arr,
            label_text,
            (pill_x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # Convert back to RGB PIL Image
    annotated_rgb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)


def detect_candies(
    image: Image.Image,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMAGE_SIZE,
) -> DetectionResult:
    """
    Execute YOLOv8 inference on candy image and package structured results.
    """
    model = get_model()
    device = get_compute_device()
    class_mapping = extract_class_mapping(model)

    # Ensure input image is RGB PIL Image
    pil_img = image.convert("RGB")
    img_w, img_h = pil_img.size

    # Run prediction
    results = model.predict(
        source=pil_img,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]
    detections: List[DetectionItem] = []
    detected_category_names: List[str] = []

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy_coords = boxes.xyxy.cpu().numpy()
        conf_scores = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(cls_ids)):
            cid = int(cls_ids[i])
            conf_val = float(conf_scores[i])
            cname = class_mapping.get(cid, f"Class_{cid}")
            detected_category_names.append(cname)

            x1, y1, x2, y2 = [int(round(c)) for c in xyxy_coords[i]]
            # Clamp inside boundaries
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            color_rgb = get_color_for_class(cid)
            color_hex = get_hex_for_class(cid)

            item = DetectionItem(
                detection_id=i + 1,
                class_id=cid,
                candy_category=cname,
                confidence=round(conf_val, 4),
                confidence_percent=format_percentage(conf_val),
                bbox=(x1, y1, x2, y2),
                box_width=x2 - x1,
                box_height=y2 - y1,
                color_rgb=color_rgb,
                color_hex=color_hex,
            )
            detections.append(item)

    # Tally candy counts
    candy_counts = count_candies(detected_category_names)

    # Draw annotated boxes
    annotated_image = draw_bounding_boxes(pil_img, detections)

    # Extract speed
    speed_info = getattr(result, "speed", {}) or {}
    speed = InferenceSpeed(
        preprocess_ms=round(speed_info.get("preprocess", 0.0), 1),
        inference_ms=round(speed_info.get("inference", 0.0), 1),
        postprocess_ms=round(speed_info.get("postprocess", 0.0), 1),
    )

    return DetectionResult(
        annotated_image=annotated_image,
        original_image=pil_img,
        detections=detections,
        candy_counts=candy_counts,
        total_candies=len(detections),
        unique_categories=len(candy_counts),
        inference_speed=speed,
        class_mapping=class_mapping,
    )

