"""Session detection history manager for Candy AI.
Maintains session-based detection logs, thumbnails, and exports.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
from PIL import Image
import streamlit as st

from utils.image_processing import create_thumbnail

SESSION_HISTORY_KEY = "candy_ai_history"
SESSION_IMAGES_KEY = "candy_ai_images"


def init_history() -> None:
    """Initialize session state history containers if not present."""
    if SESSION_HISTORY_KEY not in st.session_state:
        st.session_state[SESSION_HISTORY_KEY] = []
    if SESSION_IMAGES_KEY not in st.session_state:
        st.session_state[SESSION_IMAGES_KEY] = {}


def get_history() -> List[Dict[str, Any]]:
    """Return all session detection records, sorted newest first."""
    init_history()
    return st.session_state[SESSION_HISTORY_KEY]


def get_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single history record by its UUID."""
    history = get_history()
    for record in history:
        if record["id"] == record_id:
            return record
    return None


def get_stored_image(image_id: str) -> Optional[Image.Image]:
    """Retrieve full annotated image from session storage."""
    init_history()
    return st.session_state[SESSION_IMAGES_KEY].get(image_id)


def add_history_record(
    filename: str,
    detection_result: Dict[str, Any],
    confidence: float,
    iou: float,
) -> str:
    """
    Append a new detection entry to the session history.
    Stores lightweight metadata and thumbnails in session list,
    and stores full images in session image cache.
    """
    init_history()

    record_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    annotated_img: Image.Image = detection_result["annotated_image"]
    thumbnail = create_thumbnail(annotated_img, size=(160, 160))

    # Cache full images
    st.session_state[SESSION_IMAGES_KEY][record_id] = {
        "annotated": annotated_img,
        "original": detection_result.get("original_image"),
    }

    speed = detection_result.get("inference_speed", {})
    inference_ms = speed.get("inference_ms", 0.0)

    # Build clean serializable record
    record = {
        "id": record_id,
        "filename": filename,
        "timestamp": timestamp_str,
        "total_candies": detection_result.get("total_candies", 0),
        "unique_categories": detection_result.get("unique_categories", 0),
        "candy_counts": detection_result.get("candy_counts", {}),
        "detections": detection_result.get("detections", []),
        "thumbnail": thumbnail,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "inference_ms": inference_ms,
    }

    # Prepend newest record
    st.session_state[SESSION_HISTORY_KEY].insert(0, record)
    return record_id


def clear_history() -> None:
    """Clear all session history records and cached images."""
    st.session_state[SESSION_HISTORY_KEY] = []
    st.session_state[SESSION_IMAGES_KEY] = {}


def history_to_dataframe() -> pd.DataFrame:
    """Convert history records to a clean Pandas DataFrame for table display or CSV export."""
    history = get_history()
    if not history:
        return pd.DataFrame(
            columns=[
                "ID",
                "Timestamp",
                "Filename",
                "Total Candies",
                "Unique Categories",
                "Identified Candies",
                "Confidence Threshold",
                "IoU Threshold",
                "Inference Time (ms)",
            ]
        )

    rows = []
    for r in history:
        # Format candy counts string: "Snickers (3), Skittles (2)"
        candies_str = ", ".join(
            f"{cat}: {cnt}" for cat, cnt in r["candy_counts"].items()
        ) if r["candy_counts"] else "None detected"

        rows.append(
            {
                "ID": r["id"],
                "Timestamp": r["timestamp"],
                "Filename": r["filename"],
                "Total Candies": r["total_candies"],
                "Unique Categories": r["unique_categories"],
                "Identified Candies": candies_str,
                "Confidence Threshold": f"{r['confidence_threshold']:.2f}",
                "IoU Threshold": f"{r['iou_threshold']:.2f}",
                "Inference Time (ms)": f"{r['inference_ms']:.1f}",
            }
        )

    return pd.DataFrame(rows)

