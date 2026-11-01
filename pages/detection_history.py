"""Detection History page for Candy AI.
Review past session detection runs, inspect annotated images, export logs, and clear history.
"""

import pandas as pd
import streamlit as st

from components.detection_viewer import (
    render_candy_report_section,
    render_detections_table,
)
from components.header import render_page_header
from services.history_service import HistoryService


def render_detection_history() -> None:
    """Render the session history management page."""
    HistoryService.init_history()
    history = HistoryService.get_history()

    render_page_header(
        title="📜 Session Detection History",
        subtitle="Audit trail of all processed images, detected candy breakdowns, and historical logs",
    )

    if not history:
        st.info(
            "📂 **No detection history found in this session.**\n\n"
            "Process images in the **🔍 Candy Detection** or **📹 Real-Time Detection** pages "
            "to automatically populate session records."
        )
        return

    # Header actions: Clear history & Export CSV
    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])

    with top_col1:
        st.markdown(f"**Total Records Stored:** `{len(history)}`")

    with top_col2:
        df_hist = HistoryService.to_dataframe()
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_bytes,
            file_name="candy_ai_session_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with top_col3:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            HistoryService.clear_history()
            st.success("Session history cleared successfully.")
            st.rerun()

    st.markdown("---")

    # History Table Summary
    st.subheader("📋 Detection Records")
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # Detailed Drill-down Section
    st.markdown("### 🔎 Inspect Past Detection Run")

    record_options = [
        f"#{r['id']} — {r['filename']} ({r['timestamp']}) [Candies: {r['total_candies']}]"
        for r in history
    ]
    selected_idx = st.selectbox(
        "Select record to inspect",
        options=range(len(record_options)),
        format_func=lambda idx: record_options[idx],
    )

    selected_record = history[selected_idx]
    record_id = selected_record["id"]

    st.markdown(f"#### Run Details: `#{record_id}` — {selected_record['filename']}")

    detail_col1, detail_col2 = st.columns([1, 1])

    with detail_col1:
        cached_imgs = HistoryService.get_stored_image(record_id)
        if cached_imgs and "annotated" in cached_imgs:
            st.image(
                cached_imgs["annotated"],
                caption=f"Annotated Run #{record_id} — {selected_record['total_candies']} detected candies",
                use_container_width=True,
            )
        elif selected_record.get("thumbnail") is not None:
            st.image(
                selected_record["thumbnail"],
                caption="Thumbnail Preview",
                use_container_width=True,
            )
        else:
            st.caption("No image preview stored for this run.")

    with detail_col2:
        st.markdown(
            f"""
            - **Timestamp:** `{selected_record['timestamp']}`
            - **Source File:** `{selected_record['filename']}`
            - **Total Candies:** `{selected_record['total_candies']}`
            - **Unique Categories:** `{selected_record['unique_categories']}`
            - **Confidence Threshold:** `{selected_record['confidence_threshold']:.2f}`
            - **IoU Threshold:** `{selected_record['iou_threshold']:.2f}`
            - **Inference Speed:** `{selected_record['inference_ms']:.1f} ms`
            """
        )

        render_candy_report_section(
            candy_counts=selected_record["candy_counts"],
            total_candies=selected_record["total_candies"],
            unique_categories=selected_record["unique_categories"],
            filename=f"report_{record_id}",
        )

    # Object coordinates table for this selected run
    detections = selected_record.get("detections", [])
    if detections:
        st.markdown("##### Detected Object Coordinates")
        render_detections_table(detections, filename=f"detections_{record_id}")


render_detection_history()
