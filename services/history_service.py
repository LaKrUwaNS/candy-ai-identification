"""History management service for session-based detection audit trail."""

from typing import Any, Dict, List, Optional
import pandas as pd
from PIL import Image

from core.candy_counter import format_candy_summary_str
from core.detector import DetectionResult
from core.image_processor import create_preview_thumbnail
from utils.helpers import format_percentage, format_timestamp, generate_unique_id

SESSION_HISTORY_KEY = "candy_ai_history"
SESSION_IMAGES_KEY = "candy_ai_images"


class HistoryService:
    """Manages detection history storage, retrieval, and exports."""

    @staticmethod
    def _get_session():
        """Retrieve Streamlit session_state safely."""
        try:
            import streamlit as st
            return st.session_state
        except ImportError:
            # Fallback in-memory dictionary when tested outside Streamlit
            if not hasattr(HistoryService, "_fallback_session"):
                HistoryService._fallback_session = {}
            return HistoryService._fallback_session

    @classmethod
    def init_history(cls) -> None:
        """Initialize session state data containers if absent."""
        session = cls._get_session()
        if SESSION_HISTORY_KEY not in session:
            session[SESSION_HISTORY_KEY] = []
        if SESSION_IMAGES_KEY not in session:
            session[SESSION_IMAGES_KEY] = {}

    @classmethod
    def get_history(cls) -> List[Dict[str, Any]]:
        """Return all history records sorted newest first."""
        cls.init_history()
        session = cls._get_session()
        return session[SESSION_HISTORY_KEY]

    @classmethod
    def add_record(
        cls,
        filename: str,
        result: DetectionResult,
        confidence: float,
        iou: float,
    ) -> str:
        """
        Store a detection run in the session history.
        Generates thumbnail and caches full images separately to optimize memory.
        """
        cls.init_history()
        session = cls._get_session()

        record_id = generate_unique_id()
        timestamp = format_timestamp()

        # Cache full resolution images
        session[SESSION_IMAGES_KEY][record_id] = {
            "annotated": result.annotated_image,
            "original": result.original_image,
        }

        # Create thumbnail for lightweight session storage
        thumbnail = create_preview_thumbnail(result.annotated_image)

        # Convert detection items to dictionaries
        det_dicts = [d.to_dict() for d in result.detections]

        record = {
            "id": record_id,
            "filename": filename,
            "timestamp": timestamp,
            "total_candies": result.total_candies,
            "unique_categories": result.unique_categories,
            "candy_counts": result.candy_counts,
            "detections": det_dicts,
            "thumbnail": thumbnail,
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "inference_ms": result.inference_speed.inference_ms,
        }

        # Prepend newest record
        session[SESSION_HISTORY_KEY].insert(0, record)
        return record_id

    @classmethod
    def get_record(cls, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific record by its ID."""
        for record in cls.get_history():
            if record["id"] == record_id:
                return record
        return None

    @classmethod
    def get_stored_image(cls, record_id: str) -> Optional[Dict[str, Image.Image]]:
        """Retrieve original and annotated images from session cache."""
        cls.init_history()
        session = cls._get_session()
        return session[SESSION_IMAGES_KEY].get(record_id)

    @classmethod
    def clear_history(cls) -> None:
        """Purge all session detection records and image cache."""
        session = cls._get_session()
        session[SESSION_HISTORY_KEY] = []
        session[SESSION_IMAGES_KEY] = {}

    @classmethod
    def to_dataframe(cls) -> pd.DataFrame:
        """Generate clean Pandas DataFrame of all history records for export or table views."""
        records = cls.get_history()
        if not records:
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
        for r in records:
            summary = format_candy_summary_str(r["candy_counts"])
            rows.append(
                {
                    "ID": r["id"],
                    "Timestamp": r["timestamp"],
                    "Filename": r["filename"],
                    "Total Candies": r["total_candies"],
                    "Unique Categories": r["unique_categories"],
                    "Identified Candies": summary,
                    "Confidence Threshold": f"{r['confidence_threshold']:.2f}",
                    "IoU Threshold": f"{r['iou_threshold']:.2f}",
                    "Inference Time (ms)": f"{r['inference_ms']:.1f}",
                }
            )

        return pd.DataFrame(rows)

