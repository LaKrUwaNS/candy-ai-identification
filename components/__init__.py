"""UI components package for Candy AI."""

from components.charts import (
    render_donut_chart,
    render_horizontal_bar_chart,
    render_timeline_chart,
)
from components.detection_viewer import (
    render_candy_report_section,
    render_detections_table,
    render_image_tabs,
)
from components.header import render_page_header
from components.metric_cards import render_kpi_cards
from components.sidebar import render_inference_controls, render_sidebar_branding

__all__ = [
    "render_page_header",
    "render_kpi_cards",
    "render_sidebar_branding",
    "render_inference_controls",
    "render_donut_chart",
    "render_horizontal_bar_chart",
    "render_timeline_chart",
    "render_image_tabs",
    "render_candy_report_section",
    "render_detections_table",
]

