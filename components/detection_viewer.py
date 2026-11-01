"""Detection visualization and results viewer components."""

from io import BytesIO
from typing import Dict, List
import pandas as pd
from PIL import Image
import streamlit as st

from core.candy_counter import format_ascii_report
from core.detector import DetectionItem
from utils.image_utils import pil_to_bytes


def render_image_tabs(
    original_image: Image.Image,
    annotated_image: Image.Image,
    filename: str = "candy_scan",
) -> None:
    """Render image viewer with Annotated, Side-by-Side, and Original tabs."""
    tab_ann, tab_sbs, tab_orig = st.tabs(
        ["🎯 Annotated Detections", "↔️ Side-by-Side Comparison", "🖼️ Original Image"]
    )

    with tab_ann:
        st.image(annotated_image, caption="Annotated Detection Output", use_container_width=True)
        img_bytes = pil_to_bytes(annotated_image, format="PNG")
        st.download_button(
            label="💾 Download Annotated Image",
            data=img_bytes,
            file_name=f"detected_{filename}.png",
            mime="image/png",
        )

    with tab_sbs:
        c1, c2 = st.columns(2)
        with c1:
            st.image(original_image, caption="Original Input", use_container_width=True)
        with c2:
            st.image(annotated_image, caption="Annotated Output", use_container_width=True)

    with tab_orig:
        st.image(original_image, caption="Original Input", use_container_width=True)


def render_candy_report_section(
    candy_counts: Dict[str, int],
    total_candies: int,
    unique_categories: int,
    filename: str = "candy_report",
) -> None:
    """Render ASCII report container and export button."""
    ascii_rep = format_ascii_report(candy_counts, total_candies, unique_categories)
    st.markdown(
        f'<div class="candy-report-container">{ascii_rep}</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label="📄 Export Report (TXT)",
        data=ascii_rep,
        file_name=f"{filename}.txt",
        mime="text/plain",
    )


def render_detections_table(
    detections: List[DetectionItem],
    filename: str = "detections",
) -> None:
    """Render structured table of localized candy objects with CSV export."""
    if not detections:
        st.warning(
            "⚠️ **Zero Objects Detected**: Try lowering the Confidence Threshold slider in the sidebar."
        )
        return

    table_rows = []
    for det in detections:
        # Support either DetectionItem or dict
        if isinstance(det, DetectionItem):
            d_id = det.detection_id
            cat = det.candy_category
            cid = det.class_id
            conf = det.confidence
            conf_pct = det.confidence_percent
            x1, y1, x2, y2 = det.bbox
            bw = det.box_width
            bh = det.box_height
        else:
            d_id = det.get("detection_id", 0)
            cat = det.get("candy_category", "")
            cid = det.get("class_id", 0)
            conf = det.get("confidence", 0.0)
            conf_pct = det.get("confidence_percent", "")
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
            bw = det.get("box_width", x2 - x1)
            bh = det.get("box_height", y2 - y1)

        table_rows.append(
            {
                "Detection ID": f"#{d_id}",
                "Category": cat,
                "Class ID": cid,
                "Confidence Score": f"{conf:.4f}",
                "Confidence (%)": conf_pct,
                "Box (x1, y1)": f"({x1}, {y1})",
                "Box (x2, y2)": f"({x2}, {y2})",
                "Dimensions (W×H)": f"{bw} × {bh} px",
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Export Detections (CSV)",
        data=csv_data,
        file_name=f"{filename}.csv",
        mime="text/csv",
    )

