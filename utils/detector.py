"""Detector module for Candy AI.
Manages YOLOv8 model loading, device selection, and inference operations.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# Relative model path specified in requirements
MODEL_PATH = "Model/best.pt"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# 20 distinct accessible colors for the 20 candy classes (RGB format)
CLASS_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 168, 150),    # 0: Bounty (Deep Turquoise)
    (217, 119, 6),    # 1: Dare_caramel (Warm Amber)
    (139, 69, 19),    # 2: Dare_cocoa (Chocolate Brown)
    (245, 158, 11),   # 3: MMs_peanut (Peanut Gold)
    (239, 68, 68),    # 4: MMs_regular (Bright Red)
    (190, 18, 60),    # 5: Mars (Crimson Ruby)
    (234, 179, 8),    # 6: Twix_caramel (Golden Caramel)
    (180, 83, 9),     # 7: Twix_clasic (Bronze Gold)
    (100, 116, 139),  # 8: Twix_white (Slate Silver)
    (59, 130, 246),   # 9: airheads (Electric Blue)
    (249, 115, 22),   # 10: eugenia (Orange Peel)
    (16, 185, 129),   # 11: gummy_worms (Lime Emerald)
    (99, 102, 241),   # 12: milky_way (Royal Indigo)
    (139, 92, 246),   # 13: nerds (Vivid Violet)
    (244, 63, 94),    # 14: skittles (Bright Coral)
    (120, 53, 15),    # 15: snickers (Saddle Brown)
    (202, 138, 4),    # 16: snickers_creamy (Cream Ochre)
    (236, 72, 153),   # 17: starbust (Hot Pink)
    (2, 132, 199),    # 18: three_musketeers (Midnight Blue)
    (153, 27, 27),    # 19: twizzlers (Cherry Red)
]


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to Hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def get_model_file_path() -> Path:
    """
    Resolve model path robustly using pathlib.
    Works whether run from workspace root or pages directory.
    """
    # 1. Direct path check from current working directory
    cwd_path = Path(MODEL_PATH).resolve()
    if cwd_path.is_file():
        return cwd_path

    # 2. Check relative to this file's parent's parent (workspace root)
    root_path = (Path(__file__).resolve().parent.parent / MODEL_PATH).resolve()
    if root_path.is_file():
        return root_path

    # 3. Fallback: Search for Model/best.pt in parent hierarchy
    curr = Path(__file__).resolve().parent
    for _ in range(3):
        candidate = curr / "Model" / "best.pt"
        if candidate.is_file():
            return candidate.resolve()
        curr = curr.parent

    # Return expected path even if missing so caller can produce descriptive error
    return cwd_path


def check_model_exists() -> Tuple[bool, str]:
    """Check whether the model file exists and return path or error message."""
    target_path = get_model_file_path()
    if target_path.is_file():
        return True, str(target_path)
    return False, (
        f"Model file not found at '{MODEL_PATH}' (resolved to '{target_path}'). "
        "Please ensure 'Model/best.pt' is present in your project directory."
    )


def get_inference_device() -> str:
    """Detect available computation device (GPU with fallback to CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource(show_spinner="Loading YOLOv8 Candy Detection Model...")
def load_model() -> YOLO:
    """
    Load the existing YOLO model from Model/best.pt.
    Cache the model to prevent repeated loading across reruns and pages.
    """
    exists, model_path_str = check_model_exists()
    if not exists:
        raise FileNotFoundError(model_path_str)

    model = YOLO(model_path_str)
    return model


def get_candy_classes(model: Optional[YOLO] = None) -> Dict[int, str]:
    """
    Return class mapping directly from model.names as source of truth.
    Falls back to loading model if not provided.
    """
    if model is None:
        model = load_model()
    # Ultralytics model.names is a dict mapping int -> str
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {}


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Return consistent RGB color for a given class ID."""
    idx = class_id % len(CLASS_COLOR_PALETTE)
    return CLASS_COLOR_PALETTE[idx]


def draw_detection_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
) -> Image.Image:
    """
    Draw crisp, professional bounding boxes with category labels and confidence pills.
    Operates on RGB image and returns annotated RGB PIL Image.
    """
    if not detections:
        return image.copy()

    # Convert PIL Image to BGR for OpenCV drawing
    rgb_array = np.array(image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_array.shape[:2]

    # Dynamically scale thickness based on image resolution
    base_dim = max(img_h, img_w)
    box_thickness = max(2, int(round(base_dim / 350.0)))
    font_scale = max(0.45, base_dim / 1100.0)
    font_thickness = max(1, int(round(box_thickness * 0.7)))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        class_name = det["candy_category"]
        conf_pct = det["confidence_percent"]
        det_id = det["detection_id"]

        # Color in BGR for OpenCV
        rgb_color = det.get("color_rgb", get_color_for_class(class_id))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Main bounding box
        cv2.rectangle(bgr_array, (x1, y1), (x2, y2), bgr_color, box_thickness)

        # Subtle corner accent brackets for modern tech aesthetic
        corner_len = max(8, int(min(x2 - x1, y2 - y1) * 0.2))
        corner_thick = box_thickness + 1
        # Top-left
        cv2.line(bgr_array, (x1, y1), (x1 + corner_len, y1), bgr_color, corner_thick)
        cv2.line(bgr_array, (x1, y1), (x1, y1 + corner_len), bgr_color, corner_thick)
        # Top-right
        cv2.line(bgr_array, (x2, y1), (x2 - corner_len, y1), bgr_color, corner_thick)
        cv2.line(bgr_array, (x2, y1), (x2, y1 + corner_len), bgr_color, corner_thick)
        # Bottom-left
        cv2.line(bgr_array, (x1, y2), (x1 + corner_len, y2), bgr_color, corner_thick)
        cv2.line(bgr_array, (x1, y2), (x1, y2 - corner_len), bgr_color, corner_thick)
        # Bottom-right
        cv2.line(bgr_array, (x2, y2), (x2 - corner_len, y2), bgr_color, corner_thick)
        cv2.line(bgr_array, (x2, y2), (x2, y2 - corner_len), bgr_color, corner_thick)

        # Label pill
        label_text = f"#{det_id} {class_name} ({conf_pct})"
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
        cv2.rectangle(bgr_array, (pill_x1, pill_y1), (pill_x2, pill_y2), bgr_color, -1)

        # Determine text contrast (white vs dark text based on luminance)
        luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
        text_color = (255, 255, 255) if luminance < 140 else (20, 20, 20)

        # Draw label text
        text_y = pill_y2 - baseline - padding + 2
        cv2.putText(
            bgr_array,
            label_text,
            (pill_x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # Convert back to RGB PIL Image
    annotated_rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)


def detect_candies(
    image: Union[Image.Image, np.ndarray, str, Path],
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
) -> Dict[str, Any]:
    """
    Run YOLO inference using the existing trained model at Model/best.pt.

    Parameters:
    - image: PIL Image, numpy array (RGB or BGR), or file path
    - confidence: Detection confidence threshold [0.0 - 1.0]
    - iou: Non-Maximum Suppression IoU threshold [0.0 - 1.0]
    - imgsz: Input image size for inference (e.g. 640)

    Returns dictionary containing:
    - annotated_image: PIL Image with bounding boxes and labels
    - detections: List of detection details (ID, category, confidence, coords, color)
    - candy_counts: Dict mapping candy category to count
    - total_candies: Total number of detected candies
    - unique_categories: Number of unique candy categories
    - inference_speed: Dict with preprocess, inference, and postprocess timing in ms
    - class_mapping: Dict mapping class_id -> class_name
    """
    model = load_model()
    device = get_inference_device()

    # Normalize image to RGB PIL Image
    if isinstance(image, (str, Path)):
        pil_image = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        # If ndarray, determine if BGR or RGB
        if image.ndim == 2:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        elif image.shape[2] == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image[:, :, :3])
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Run inference with ultralytics
    # Use conf, iou, imgsz, device parameters
    results = model.predict(
        source=pil_image,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]
    class_mapping = get_candy_classes(model)

    detections: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # Extract coordinates, confidence scores, and class IDs
        xyxy_coords = boxes.xyxy.cpu().numpy()
        conf_scores = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(cls_ids)):
            cid = int(cls_ids[i])
            conf_val = float(conf_scores[i])
            cname = class_mapping.get(cid, f"Class_{cid}")
            x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy_coords[i]]

            # Ensure coordinates stay within image boundaries
            img_w, img_h = pil_image.size
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            color_rgb = get_color_for_class(cid)
            color_hex = rgb_to_hex(color_rgb)

            detection_item = {
                "detection_id": i + 1,
                "class_id": cid,
                "candy_category": cname,
                "confidence": round(conf_val, 4),
                "confidence_percent": f"{conf_val * 100:.1f}%",
                "bbox": [x1, y1, x2, y2],
                "box_width": x2 - x1,
                "box_height": y2 - y1,
                "color_rgb": color_rgb,
                "color_hex": color_hex,
            }
            detections.append(detection_item)

        # Tally candy counts
        raw_counts = Counter(d["candy_category"] for d in detections)
        # Sort by count descending, then alphabetically
        category_counts = dict(
            sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))
        )

    # Draw custom annotated bounding boxes
    annotated_image = draw_detection_boxes(pil_image, detections)

    speed_info = getattr(result, "speed", {}) or {}
    inference_speed = {
        "preprocess_ms": round(speed_info.get("preprocess", 0.0), 1),
        "inference_ms": round(speed_info.get("inference", 0.0), 1),
        "postprocess_ms": round(speed_info.get("postprocess", 0.0), 1),
    }

    return {
        "annotated_image": annotated_image,
        "original_image": pil_image,
        "detections": detections,
        "candy_counts": category_counts,
        "total_candies": len(detections),
        "unique_categories": len(category_counts),
        "inference_speed": inference_speed,
        "class_mapping": class_mapping,
    }

