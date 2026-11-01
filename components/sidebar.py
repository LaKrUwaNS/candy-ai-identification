"""Centralized sidebar UI component with branding and inference controls."""

from typing import Tuple
import streamlit as st

from config.settings import (
    APP_SUBTITLE,
    APP_TITLE,
    CONFIDENCE_STEP,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    IOU_STEP,
    LOGO_PATH,
    MAX_CONFIDENCE,
    MAX_IOU,
    MIN_CONFIDENCE,
    MIN_IOU,
    SUPPORTED_RESOLUTIONS,
)
from services.history_service import HistoryService
from services.statistics_service import StatisticsService
from utils.file_handler import check_model_exists


def render_sidebar_branding() -> None:
    """Render application logo, title, and model readiness status in sidebar."""
    with st.sidebar:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=100)

        st.markdown(
            f"""
            <div style="margin-bottom: 16px;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #FFFFFF;">Candy AI</h2>
                <span style="font-size: 12px; color: #94A3B8;">{APP_SUBTITLE}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_ok, _ = check_model_exists()
        if model_ok:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;">
                    <span style="color: #34D399; font-size: 12px; font-weight: 700;">● MODEL READY (YOLOv8n)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;">
                    <span style="color: #F87171; font-size: 12px; font-weight: 700;">● MODEL NOT FOUND</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Session Metrics Pill
        history = HistoryService.get_history()
        metrics = StatisticsService.calculate_metrics(history)
        st.markdown("---")
        st.markdown("##### 📌 Session Snapshot")
        s1, s2 = st.columns(2)
        s1.metric("Images", metrics["total_images"])
        s2.metric("Candies", metrics["total_candies"])


def render_inference_controls(key_prefix: str = "main") -> Tuple[float, float, int]:
    """Render confidence, IoU, and image size controls in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Inference Controls")
        conf_val = st.slider(
            "Confidence Threshold",
            min_value=MIN_CONFIDENCE,
            max_value=MAX_CONFIDENCE,
            value=float(DEFAULT_CONFIDENCE),
            step=CONFIDENCE_STEP,
            key=f"{key_prefix}_conf",
            help="Minimum confidence probability for an object to be detected.",
        )
        iou_val = st.slider(
            "IoU (NMS) Threshold",
            min_value=MIN_IOU,
            max_value=MAX_IOU,
            value=float(DEFAULT_IOU),
            step=IOU_STEP,
            key=f"{key_prefix}_iou",
            help="Intersection over Union threshold for Non-Maximum Suppression overlap filtering.",
        )
        imgsz_val = st.select_slider(
            "Inference Resolution (px)",
            options=SUPPORTED_RESOLUTIONS,
            value=DEFAULT_IMAGE_SIZE,
            key=f"{key_prefix}_imgsz",
            help="Input dimension fed into YOLOv8. 640 is standard default.",
        )
    return conf_val, iou_val, imgsz_val

