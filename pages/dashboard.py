"""Dashboard page for Candy AI.
Displays dynamic session statistics, category distribution, and recent activity.
"""

import pandas as pd
import streamlit as st

from components.charts import (
    render_donut_chart,
    render_horizontal_bar_chart,
    render_timeline_chart,
)
from components.header import render_page_header
from components.metric_cards import render_kpi_cards
from services.history_service import HistoryService
from services.statistics_service import StatisticsService


def render_dashboard() -> None:
    """Render the dashboard module."""
    history = HistoryService.get_history()
    metrics = StatisticsService.calculate_metrics(history)

    # Reusable Page Header
    render_page_header(
        title="📊 Executive Detection Dashboard",
        subtitle="Live analytics and real-time statistics computed from current session activity",
        badge_text="YOLOv8n Active",
    )

    # Reusable KPI Metric Cards
    render_kpi_cards(
        total_images=metrics["total_images"],
        total_candies=metrics["total_candies"],
        unique_categories=metrics["unique_categories_count"],
        top_candy=metrics["most_frequent_candy"],
        top_count=metrics["most_frequent_count"],
        avg_candies=metrics["avg_candies_per_image"],
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Empty State Guide if no session data yet
    if metrics["total_images"] == 0:
        st.info(
            "👋 **Welcome to Candy AI!** No detections have been recorded yet in this session.\n\n"
            "- Navigate to **🔍 Candy Detection** in the sidebar to upload candy photos (JPG, PNG, WEBP).\n"
            "- Or visit **📹 Real-Time Detection** to test live camera recognition.\n"
            "- All statistics on this dashboard are dynamically updated with zero fake data."
        )
        return

    # Visual Analytics Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("🍩 Category Distribution")
        render_donut_chart(
            metrics["category_distribution"],
            title="Candy Proportions in Current Session",
        )

    with chart_col2:
        st.subheader("📊 Category Frequency Breakdown")
        render_horizontal_bar_chart(
            metrics["category_distribution"],
            title="Total Count by Candy Type",
        )

    # Detection Timeline
    if len(history) >= 2:
        st.subheader("📈 Detection Activity Over Time")
        render_timeline_chart(history)

    # Recent Detection Activity Section
    st.subheader("🕒 Recent Detection Activity")
    recent_items = history[:5]

    activity_rows = []
    for item in recent_items:
        items_summary = ", ".join(
            f"{cat} ({cnt})" for cat, cnt in item["candy_counts"].items()
        ) if item["candy_counts"] else "None detected"

        activity_rows.append(
            {
                "Scan ID": f"#{item['id']}",
                "Timestamp": item["timestamp"],
                "Image Source": item["filename"],
                "Candies Found": item["total_candies"],
                "Unique Types": item["unique_categories"],
                "Detected Items": items_summary,
                "Inference Speed": f"{item['inference_ms']:.1f} ms",
            }
        )

    df_recent = pd.DataFrame(activity_rows)
    st.dataframe(df_recent, use_container_width=True, hide_index=True)


render_dashboard()
