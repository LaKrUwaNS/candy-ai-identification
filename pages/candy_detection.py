"""Candy Detection page for Candy AI.
Handles image upload, multi-candy YOLO detection, bounding boxes, counting, and reporting.
"""

import streamlit as st

from components.charts import render_horizontal_bar_chart
from components.detection_viewer import (
    render_candy_report_section,
    render_detections_table,
    render_image_tabs,
)
from components.header import render_page_header
from components.sidebar import render_inference_controls
from services.detection_service import DetectionService
from utils.file_handler import check_model_exists


def render_candy_detection() -> None:
    """Render the candy detection module."""
    render_page_header(
        title="🔍 Candy Image Detection & Counting",
        subtitle="Upload candy imagery to localize packages, classify 20 categories, and generate count reports",
    )

    # Check model presence first
    model_ok, model_msg = check_model_exists()
    if not model_ok:
        st.error(f"⚠️ **Model Initialization Error**: {model_msg}")
        st.stop()

    # Reusable Inference Controls in Sidebar
    conf_val, iou_val, imgsz_val = render_inference_controls(key_prefix="img_det")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a candy image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        help="Upload clear photographs of individual candies or multiple candies on a surface.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div style="border: 2px dashed #CBD5E1; border-radius: 12px; padding: 40px; text-align: center; background: #F8FAFC;">
                <p style="font-size: 36px; margin: 0;">📸</p>
                <h3 style="color: #334155; margin-top: 8px;">Upload Image to Begin Analysis</h3>
                <p style="color: #64748B; font-size: 14px; max-width: 500px; margin: 0 auto;">
                    Supports single or multi-candy photos. Our custom YOLOv8 model detects 20 brands
                    including Bounty, M&Ms, Mars, Snickers, Skittles, Twix, and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Validate uploaded file through service
    val_err = DetectionService.validate_upload(uploaded_file)
    if val_err:
        st.error(f"❌ Invalid image: {val_err}")
        return

    # Action row: Run Detection
    run_col1, _ = st.columns([1, 3])
    with run_col1:
        detect_button = st.button("🚀 Run Candy Detection", use_container_width=True)

    # Cache key for current image + settings to avoid re-running on simple UI reruns
    current_key = f"det_{uploaded_file.name}_{uploaded_file.size}_{conf_val}_{iou_val}_{imgsz_val}"

    if detect_button or current_key not in st.session_state:
        with st.spinner("Analyzing image with YOLOv8 Candy Detector..."):
            try:
                result = DetectionService.process_image(
                    image_input=uploaded_file,
                    filename=uploaded_file.name,
                    confidence=conf_val,
                    iou=iou_val,
                    imgsz=imgsz_val,
                    log_to_history=True,
                )
                st.session_state[current_key] = result
            except Exception as e:
                st.error(f"❌ Inference failed: {str(e)}")
                return
    else:
        result = st.session_state[current_key]

    # Display Results
    st.markdown("---")

    # High-level summary metrics
    inf_time = result.inference_speed.inference_ms
    orig_img = result.original_image

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Total Candies Found", result.total_candies)
    with mcol2:
        st.metric("Unique Categories", result.unique_categories)
    with mcol3:
        st.metric("Inference Time", f"{inf_time:.1f} ms")
    with mcol4:
        st.metric("Resolution", f"{orig_img.width} × {orig_img.height}")

    # Reusable Image Visualization Tabs
    render_image_tabs(
        original_image=result.original_image,
        annotated_image=result.annotated_image,
        filename=uploaded_file.name,
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Section D: Candy Counting System & Report
    st.markdown("### 📋 Candy Counting Report")
    rep_col1, rep_col2 = st.columns([1, 1])

    with rep_col1:
        render_candy_report_section(
            candy_counts=result.candy_counts,
            total_candies=result.total_candies,
            unique_categories=result.unique_categories,
            filename=f"candy_report_{uploaded_file.name}",
        )

    with rep_col2:
        if result.candy_counts:
            render_horizontal_bar_chart(
                result.candy_counts,
                title="Candy Quantities Detected",
            )
        else:
            st.info("No candies identified in this image with current threshold settings.")

    # Section C: Professional Detection Details Table
    st.markdown("### 📑 Detailed Object Localization & Confidence")
    render_detections_table(
        detections=result.detections,
        filename=f"detections_{uploaded_file.name}",
    )


render_candy_detection()
