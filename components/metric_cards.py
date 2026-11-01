"""Reusable KPI metric card components."""

import streamlit as st


def render_kpi_cards(
    total_images: int,
    total_candies: int,
    unique_categories: int,
    top_candy: str,
    top_count: int,
    avg_candies: float,
) -> None:
    """Render a responsive 4-column row of styled metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-orange">
                <div class="metric-label">Images Processed</div>
                <div class="metric-value">{total_images}</div>
                <div class="metric-subtext">Total scans in session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-blue">
                <div class="metric-label">Candies Detected</div>
                <div class="metric-value">{total_candies}</div>
                <div class="metric-subtext">Avg {avg_candies:.1f} per image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-green">
                <div class="metric-label">Unique Categories</div>
                <div class="metric-value">{unique_categories} <span style="font-size: 16px; color: #64748B;">/ 20</span></div>
                <div class="metric-subtext">Discovered classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-purple">
                <div class="metric-label">Most Frequent Candy</div>
                <div class="metric-value" style="font-size: 22px; line-height: 1.4; word-break: break-word;">{top_candy}</div>
                <div class="metric-subtext">{top_count} object(s) detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

