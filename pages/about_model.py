"""About Model page for Candy AI.
Displays architecture, weights specifications, hardware acceleration, and class mapping.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import torch

from components.header import render_page_header
from config.settings import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    RELATIVE_MODEL_PATH,
)
from core.model_loader import extract_class_mapping, get_device_info, get_model
from utils.file_handler import check_model_exists, get_file_size_mb, resolve_model_path


def render_about_model() -> None:
    """Render the model architecture and specifications page."""
    render_page_header(
        title="ℹ️ About the YOLOv8 Candy Model",
        subtitle="Technical architecture, weights specifications, device configuration, and class registry",
    )

    model_ok, model_msg = check_model_exists()
    resolved_path = resolve_model_path()
    dev_info = get_device_info()

    # Specs Grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="spec-item">
                <div class="spec-title">Model Architecture</div>
                <div class="spec-value">YOLOv8n (Nano)</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Model Task</div>
                <div class="spec-value">Custom Object Detection</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Class Capacity</div>
                <div class="spec-value">20 Candy Categories</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Primary Framework</div>
                <div class="spec-value">Ultralytics YOLO / PyTorch {torch.__version__}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        file_size_mb = (
            f"{get_file_size_mb(resolved_path):.2f} MB"
            if resolved_path.is_file()
            else "Missing"
        )
        st.markdown(
            f"""
            <div class="spec-item">
                <div class="spec-title">Weights File Location</div>
                <div class="spec-value">{RELATIVE_MODEL_PATH}</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Weights File Size</div>
                <div class="spec-value">{file_size_mb}</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Active Compute Device</div>
                <div class="spec-value">{dev_info['device']} ({dev_info['device_name']})</div>
            </div>
            <div class="spec-item">
                <div class="spec-title">Supported Formats</div>
                <div class="spec-value">JPG, JPEG, PNG, WEBP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Inference Configuration Reference
    st.subheader("⚙️ Standard Inference Defaults")
    inf_c1, inf_c2, inf_c3 = st.columns(3)
    with inf_c1:
        st.metric("Default Confidence", f"{DEFAULT_CONFIDENCE:.2f}")
    with inf_c2:
        st.metric("Default IoU (NMS)", f"{DEFAULT_IOU:.2f}")
    with inf_c3:
        st.metric("Input Size (imgsz)", f"{DEFAULT_IMAGE_SIZE} × {DEFAULT_IMAGE_SIZE}")

    st.markdown("---")

    # Actual Class Mapping from model.names
    st.subheader("🏷️ Registered Class Mapping (Source of Truth)")
    st.caption("Extracted directly from `model.names` within `Model/best.pt`.")

    if not model_ok:
        st.error(f"Cannot load class mapping: {model_msg}")
        return

    model = get_model()
    class_mapping = extract_class_mapping(model)

    mapping_rows = [
        {"Class ID": cid, "Category Label": cname}
        for cid, cname in class_mapping.items()
    ]
    df_mapping = pd.DataFrame(mapping_rows)

    mcol1, mcol2 = st.columns([1, 1])
    midpoint = len(mapping_rows) // 2

    with mcol1:
        st.dataframe(df_mapping.iloc[:midpoint], use_container_width=True, hide_index=True)
    with mcol2:
        st.dataframe(df_mapping.iloc[midpoint:], use_container_width=True, hide_index=True)

    st.info(
        "💡 **Inference Notice:** Object detection accuracy depends on illumination, distance, "
        "bounding orientation, and package occlusion. For optimal results, ensure the candy packages "
        "are clearly visible and well lit."
    )


render_about_model()
