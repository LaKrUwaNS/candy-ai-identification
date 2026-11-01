"""Real-Time Webcam Detection page for Candy AI.
Provides continuous live video feed recognition and instant camera snapshot modes.
"""

import time
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from components.detection_viewer import render_candy_report_section
from components.header import render_page_header
from config.settings import DEFAULT_CONFIDENCE, DEFAULT_IOU
from services.camera_service import CameraService, FPSCalculator
from services.detection_service import DetectionService
from utils.file_handler import check_model_exists


def render_realtime_detection() -> None:
    """Render the real-time webcam detection module."""
    render_page_header(
        title="📹 Real-Time Webcam Candy Detection",
        subtitle="Continuous video inference and snapshot candy identification via connected camera",
    )

    model_ok, model_msg = check_model_exists()
    if not model_ok:
        st.error(f"⚠️ **Model Initialization Error**: {model_msg}")
        st.stop()

    # Sidebar parameters
    with st.sidebar:
        st.markdown("### ⚙️ Camera Inference Settings")
        conf_val = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=float(DEFAULT_CONFIDENCE),
            step=0.05,
            key="webcam_conf",
        )
        iou_val = st.slider(
            "IoU (NMS) Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_IOU),
            step=0.05,
            key="webcam_iou",
        )
        camera_idx = st.selectbox(
            "Camera Device Index",
            options=[0, 1, 2],
            index=0,
            help="Default camera is 0. If external webcam is plugged in, try 1 or 2.",
        )

    cam_tab1, cam_tab2 = st.tabs(
        ["🎥 Live Camera Feed (Continuous)", "📸 Browser Camera Snapshot"]
    )

    # -------------------------------------------------------------
    # TAB 1: Continuous Live Camera Stream (OpenCV)
    # -------------------------------------------------------------
    with cam_tab1:
        st.markdown("#### Live Video Stream Recognition")
        st.caption(
            "Uses your local computer camera to continuously track and identify candies in real-time."
        )

        ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 2])
        with ctrl_col1:
            start_stream = st.button("▶️ Start Camera", type="primary", key="btn_start_cam")
        with ctrl_col2:
            stop_stream = st.button("⏹️ Stop Camera", key="btn_stop_cam")

        if "is_streaming" not in st.session_state:
            st.session_state.is_streaming = False

        if start_stream:
            st.session_state.is_streaming = True
        if stop_stream:
            st.session_state.is_streaming = False

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if st.session_state.is_streaming:
            cap = CameraService.open_camera(device_index=camera_idx, width=640, height=480)

            if cap is None:
                st.session_state.is_streaming = False
                status_placeholder.error(
                    f"❌ **Could not open Camera (Index {camera_idx})**.\n\n"
                    "- Please ensure your webcam is plugged in and not in use by another app.\n"
                    "- Verify Windows Camera privacy permissions: *Windows Settings > Privacy & Security > Camera*.\n"
                    "- Or switch to the **📸 Browser Camera Snapshot** tab above."
                )
            else:
                fps_calc = FPSCalculator(interval_seconds=1.0)

                try:
                    while st.session_state.is_streaming:
                        ret, frame = CameraService.read_frame(cap)
                        if not ret or frame is None:
                            status_placeholder.warning("⚠️ Failed to grab video frame.")
                            break

                        current_fps = fps_calc.tick()

                        # Convert BGR to RGB PIL
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(rgb_frame)

                        # Run detection via service (without logging every frame to history)
                        result = DetectionService.process_image(
                            image_input=pil_frame,
                            confidence=conf_val,
                            iou=iou_val,
                            imgsz=480,
                            log_to_history=False,
                        )

                        # Display annotated image
                        frame_placeholder.image(
                            result.annotated_image,
                            caption=f"Live Feed — FPS: {current_fps:.1f} | Detections: {result.total_candies}",
                            use_container_width=True,
                        )

                        # Live metric update
                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Live Candies", result.total_candies)
                            m2.metric("Unique Types", result.unique_categories)
                            m3.metric("Live FPS", f"{current_fps:.1f}")

                            if result.candy_counts:
                                badges_html = " ".join(
                                    f"<span class='candy-badge' style='background:#0F172A; border:1px solid #FF6B35; color:#FBBF24;'>{cat}: {cnt}</span>"
                                    for cat, cnt in result.candy_counts.items()
                                )
                                st.markdown(f"**Detected:** {badges_html}", unsafe_allow_html=True)

                        time.sleep(0.03)

                except Exception as exc:
                    status_placeholder.error(f"Stream error: {str(exc)}")
                finally:
                    CameraService.release_camera(cap)
                    st.session_state.is_streaming = False
                    status_placeholder.info("Camera stream stopped.")
        else:
            frame_placeholder.info(
                "Click **▶️ Start Camera** to begin real-time detection, or switch to the **📸 Browser Camera Snapshot** tab."
            )

    # -------------------------------------------------------------
    # TAB 2: Browser Camera Snapshot
    # -------------------------------------------------------------
    with cam_tab2:
        st.markdown("#### High-Resolution Camera Snapshot")
        st.caption("Capture an instant snapshot directly from your browser's webcam.")

        camera_img = st.camera_input("Take a photo of candies to detect")

        if camera_img is not None:
            with st.spinner("Processing camera snapshot..."):
                snap_result = DetectionService.process_image(
                    image_input=camera_img,
                    filename="webcam_snapshot.jpg",
                    confidence=conf_val,
                    iou=iou_val,
                    imgsz=640,
                    log_to_history=True,
                )

            c_res1, c_res2 = st.columns([1, 1])
            with c_res1:
                st.image(
                    snap_result.annotated_image,
                    caption=f"Snapshot Result: {snap_result.total_candies} candies found",
                    use_container_width=True,
                )

            with c_res2:
                render_candy_report_section(
                    candy_counts=snap_result.candy_counts,
                    total_candies=snap_result.total_candies,
                    unique_categories=snap_result.unique_categories,
                    filename="snapshot_report",
                )

                if snap_result.detections:
                    det_df = pd.DataFrame(
                        [
                            {
                                "ID": f"#{d.detection_id}",
                                "Candy": d.candy_category,
                                "Confidence": d.confidence_percent,
                                "Box (W×H)": f"{d.box_width}×{d.box_height} px",
                            }
                            for d in snap_result.detections
                        ]
                    )
                    st.dataframe(det_df, use_container_width=True, hide_index=True)


render_realtime_detection()
