"""Interactive Plotly chart components for Candy AI."""

from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.statistics_service import StatisticsService


def render_donut_chart(
    category_counts: Dict[str, int],
    title: str = "Candy Category Distribution",
) -> None:
    """Render an interactive Plotly donut chart."""
    if not category_counts:
        return

    labels = list(category_counts.keys())
    values = list(category_counts.values())

    colors = [
        "#FF6B35", "#F7931A", "#FBBF24", "#34D399", "#38BDF8",
        "#818CF8", "#A78BFA", "#F472B6", "#FB7185", "#E11D48",
        "#D97706", "#059669", "#0284C7", "#4F46E5", "#7C3AED",
        "#DB2777", "#B91C1C", "#92400E", "#047857", "#0369A1"
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                textinfo="label+value",
                hoverinfo="label+value+percent",
                marker=dict(
                    colors=colors[: len(labels)],
                    line=dict(color="#ffffff", width=2),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
    )

    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)


def render_horizontal_bar_chart(
    category_counts: Dict[str, int],
    title: str = "Category Count Breakdown",
) -> None:
    """Render a horizontal bar chart of category counts."""
    if not category_counts:
        return

    df = pd.DataFrame(
        list(category_counts.items()), columns=["Category", "Count"]
    ).sort_values("Count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["Count"],
            y=df["Category"],
            orientation="h",
            marker=dict(
                color=df["Count"],
                colorscale=[
                    [0.0, "#FBBF24"],
                    [0.5, "#F97316"],
                    [1.0, "#EA580C"],
                ],
                line=dict(color="#C2410C", width=1),
            ),
            text=df["Count"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        margin=dict(t=50, b=30, l=10, r=20),
        xaxis=dict(
            title="Number of Candies",
            gridcolor="#E2E8F0",
            dtick=1,
            tick0=0,
        ),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(280, len(df) * 28 + 80),
    )

    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)


def render_timeline_chart(history: List[dict]) -> None:
    """Render detection volume trend over time."""
    if not history or len(history) < 2:
        return

    timestamps, counts, filenames = StatisticsService.get_timeline_series(history)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=counts,
            mode="lines+markers",
            name="Candies Detected",
            line=dict(color="#FF6B35", width=3, shape="spline"),
            marker=dict(size=8, color="#0F172A", line=dict(color="#FF6B35", width=2)),
            text=filenames,
            hovertemplate="<b>%{text}</b><br>Time: %{x}<br>Candies: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Session Detection Activity Timeline</b>",
            font=dict(size=16, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        xaxis=dict(title="Detection Timestamp", gridcolor="#E2E8F0"),
        yaxis=dict(title="Candies Detected", gridcolor="#E2E8F0", dtick=1),
        margin=dict(t=50, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )

    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)

