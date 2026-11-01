"""Candy AI – Intelligent Candy Identification System
Main Application Entrypoint and Navigation Router.
"""

from pathlib import Path
import streamlit as st

from components.sidebar import render_sidebar_branding
from config.settings import APP_ICON, APP_TITLE, STYLES_PATH
from services.history_service import HistoryService

# 1. Global Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Inject Custom CSS
def inject_custom_css() -> None:
    """Inject assets/styles.css into the page DOM."""
    target_path = STYLES_PATH if STYLES_PATH.is_file() else STYLES_PATH.parent / "style.css"
    if target_path.is_file():
        with open(target_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


inject_custom_css()
HistoryService.init_history()

# 3. Sidebar Branding & Session Status
render_sidebar_branding()

# 4. Modern Streamlit Navigation Router
pages = {
    "Overview": [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "Detection Engine": [
        st.Page("pages/candy_detection.py", title="Candy Detection", icon="🔍"),
        st.Page("pages/realtime_detection.py", title="Real-Time Detection", icon="📹"),
    ],
    "Catalog & Audit": [
        st.Page("pages/candy_categories.py", title="Candy Categories", icon="🍬"),
        st.Page("pages/detection_history.py", title="Detection History", icon="📜"),
    ],
    "System": [
        st.Page("pages/about_model.py", title="About Model", icon="ℹ️"),
    ],
}

pg = st.navigation(pages)
pg.run()
