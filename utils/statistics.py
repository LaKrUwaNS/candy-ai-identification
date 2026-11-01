"""Statistics and report generation utilities for Candy AI.
Computes metrics dynamically from detection results and session history.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def calculate_session_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate high-level dashboard metrics from session history.
    Strictly calculates from actual history; no hardcoded or fake statistics.
    """
    if not history:
        return {
            "total_images": 0,
            "total_candies": 0,
            "unique_categories_count": 0,
            "most_frequent_candy": "None",
            "most_frequent_count": 0,
            "category_distribution": {},
            "category_session_counts": {},
            "avg_candies_per_image": 0.0,
            "avg_inference_time_ms": 0.0,
        }

    total_images = len(history)
    total_candies = sum(record.get("total_candies", 0) for record in history)

    # Aggregate counts across all session records
    aggregated_counts: Counter = Counter()
    total_inference_time = 0.0

    for record in history:
        counts = record.get("candy_counts", {})
        aggregated_counts.update(counts)
        total_inference_time += record.get("inference_ms", 0.0)

    unique_categories_count = len(aggregated_counts)

    if aggregated_counts:
        top_candy, top_count = aggregated_counts.most_common(1)[0]
    else:
        top_candy, top_count = "None", 0

    avg_candies = round(total_candies / total_images, 1) if total_images > 0 else 0.0
    avg_inference = round(total_inference_time / total_images, 1) if total_images > 0 else 0.0

    # Sort distribution by count descending
    sorted_dist = dict(
        sorted(aggregated_counts.items(), key=lambda x: (-x[1], x[0]))
    )

    return {
        "total_images": total_images,
        "total_candies": total_candies,
        "unique_categories_count": unique_categories_count,
        "most_frequent_candy": top_candy,
        "most_frequent_count": top_count,
        "category_distribution": sorted_dist,
        "category_session_counts": dict(aggregated_counts),
        "avg_candies_per_image": avg_candies,
        "avg_inference_time_ms": avg_inference,
    }


def format_candy_report(
    candy_counts: Dict[str, int],
    total_candies: int,
    unique_categories: int,
) -> str:
    """
    Format a dynamic ASCII candy detection report matching the user specification.
    """
    lines = [
        "========================================",
        "        CANDY DETECTION REPORT          ",
        "========================================",
        "",
    ]

    if not candy_counts:
        lines.append("No candy objects detected.")
    else:
        max_name_len = max((len(name) for name in candy_counts.keys()), default=15)
        name_width = max(max_name_len + 2, 18)

        for name, count in candy_counts.items():
            lines.append(f"{name.ljust(name_width)}: {count}")

    lines.append("")
    lines.append("----------------------------------------")
    lines.append(f"Total Candies      : {total_candies}")
    lines.append(f"Unique Categories  : {unique_categories}")
    lines.append("========================================")

    return "\n".join(lines)


def create_donut_chart(
    category_counts: Dict[str, int],
    title: str = "Candy Category Distribution",
) -> Optional[go.Figure]:
    """Generate a modern Plotly donut chart for candy distribution."""
    if not category_counts:
        return None

    labels = list(category_counts.keys())
    values = list(category_counts.values())

    # Premium candy color palette
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

    return fig


def create_horizontal_bar_chart(
    category_counts: Dict[str, int],
    title: str = "Category Count Breakdown",
) -> Optional[go.Figure]:
    """Generate a clean horizontal bar chart for category counts."""
    if not category_counts:
        return None

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

    return fig


def create_confidence_distribution_chart(
    detections: List[Dict[str, Any]],
) -> Optional[go.Figure]:
    """Plot confidence score distribution for detected items."""
    if not detections:
        return None

    df = pd.DataFrame(detections)

    fig = px.bar(
        df,
        x="candy_category",
        y="confidence",
        color="candy_category",
        hover_data=["confidence_percent", "detection_id"],
        title="<b>Detection Confidence by Category</b>",
        labels={"confidence": "Confidence Score", "candy_category": "Category"},
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        margin=dict(t=50, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=320,
    )

    return fig


def create_history_timeline_chart(
    history: List[Dict[str, Any]],
) -> Optional[go.Figure]:
    """Plot detection volume over time across the current session."""
    if not history or len(history) < 2:
        return None

    # Reverse to show chronological order
    chrono = list(reversed(history))
    timestamps = [r["timestamp"] for r in chrono]
    totals = [r["total_candies"] for r in chrono]
    labels = [r["filename"] for r in chrono]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=totals,
            mode="lines+markers",
            name="Candies Detected",
            line=dict(color="#FF6B35", width=3, shape="spline"),
            marker=dict(size=8, color="#0F172A", line=dict(color="#FF6B35", width=2)),
            text=labels,
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

    return fig

