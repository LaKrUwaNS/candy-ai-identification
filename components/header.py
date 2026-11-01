"""Reusable page header component."""

from typing import Optional
import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    badge_text: Optional[str] = None,
) -> None:
    """Render a modern gradient header with title, subtitle, and optional status badge."""
    badge_html = ""
    if badge_text:
        badge_html = f"""
        <div style="text-align: right;">
            <span style="background: rgba(255, 107, 53, 0.2); border: 1px solid #FF6B35; color: #FFAA00; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">
                {badge_text}
            </span>
        </div>
        """

    st.markdown(
        f"""
        <div class="main-header-container">
            <div>
                <h1 class="main-header-title">{title}</h1>
                <p class="main-header-subtitle">{subtitle}</p>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

