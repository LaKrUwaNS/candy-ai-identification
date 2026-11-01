"""Candy Categories page for Candy AI.
Displays the 20 supported candy categories with class IDs and session detection counts.
"""

import pandas as pd
import streamlit as st

from components.header import render_page_header
from config.settings import get_color_for_class, get_hex_for_class
from core.model_loader import extract_class_mapping, get_model
from services.history_service import HistoryService
from services.statistics_service import StatisticsService
from utils.file_handler import check_model_exists


def render_candy_categories() -> None:
    """Render the candy categories catalog page."""
    render_page_header(
        title="🍬 Supported Candy Categories",
        subtitle="Comprehensive catalog of all 20 classes identified by the trained YOLOv8 model",
        badge_text="20 Classes",
    )

    model_ok, model_msg = check_model_exists()
    if not model_ok:
        st.error(f"⚠️ **Model Initialization Error**: {model_msg}")
        st.stop()

    model = get_model()
    class_mapping = extract_class_mapping(model)

    # Compute actual session detection frequency
    history = HistoryService.get_history()
    metrics = StatisticsService.calculate_metrics(history)
    session_counts = metrics["category_session_counts"]

    # Filter/Search controls
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_term = st.text_input(
            "🔎 Search Categories",
            placeholder="e.g. Twix, MMs, Snickers...",
        ).strip().lower()

    with search_col2:
        sort_mode = st.selectbox(
            "Sort By",
            options=["Class ID (Ascending)", "Category Name (A-Z)", "Session Detections (High to Low)"],
            index=0,
        )

    # Build data items
    category_items = []
    for cid, cname in class_mapping.items():
        cnt = session_counts.get(cname, 0)
        rgb = get_color_for_class(cid)
        hex_code = get_hex_for_class(cid)
        category_items.append(
            {
                "class_id": cid,
                "category_name": cname,
                "session_count": cnt,
                "color_rgb": rgb,
                "color_hex": hex_code,
            }
        )

    # Apply search filter
    if search_term:
        category_items = [
            item for item in category_items
            if search_term in item["category_name"].lower() or search_term in str(item["class_id"])
        ]

    # Apply sorting
    if sort_mode == "Class ID (Ascending)":
        category_items.sort(key=lambda x: x["class_id"])
    elif sort_mode == "Category Name (A-Z)":
        category_items.sort(key=lambda x: x["category_name"].lower())
    elif sort_mode == "Session Detections (High to Low)":
        category_items.sort(key=lambda x: (-x["session_count"], x["class_id"]))

    st.markdown(f"**Showing {len(category_items)} Categories**")

    # Render in responsive 2-column card grid
    grid_col1, grid_col2 = st.columns(2)

    for idx, item in enumerate(category_items):
        target_col = grid_col1 if idx % 2 == 0 else grid_col2
        with target_col:
            st.markdown(
                f"""
                <div class="category-card" style="border-left: 5px solid {item['color_hex']};">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 24px; height: 24px; border-radius: 6px; background-color: {item['color_hex']}; box-shadow: 0 2px 4px rgba(0,0,0,0.15);"></div>
                        <div>
                            <span style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.5px;">CLASS #{item['class_id']}</span>
                            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">{item['category_name']}</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 11px; color: #64748B; display: block;">Session Scans</span>
                        <span style="font-size: 18px; font-weight: 800; color: {'#FF6B35' if item['session_count'] > 0 else '#94A3B8'};">
                            {item['session_count']}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Detailed Table View Expander
    with st.expander("📋 View Categories Data Table", expanded=False):
        df_cats = pd.DataFrame(
            [
                {
                    "Class ID": it["class_id"],
                    "Category Name": it["category_name"],
                    "Session Detections": it["session_count"],
                    "Box Color (HEX)": it["color_hex"],
                }
                for it in category_items
            ]
        )
        st.dataframe(df_cats, use_container_width=True, hide_index=True)


render_candy_categories()
