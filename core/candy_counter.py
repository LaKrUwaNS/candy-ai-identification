"""Candy counting, ranking, and report generation logic."""

from collections import Counter
from typing import Dict, List, Optional, Tuple


def count_candies(category_names: List[str]) -> Dict[str, int]:
    """
    Tally occurrences of detected candy categories.
    Returns a sorted dictionary: primary sort by count descending, secondary alphabetically.
    """
    if not category_names:
        return {}

    raw_counter = Counter(category_names)
    sorted_items = sorted(raw_counter.items(), key=lambda item: (-item[1], item[0]))
    return dict(sorted_items)


def get_top_candy(counts: Dict[str, int]) -> Tuple[str, int]:
    """Return the most frequently detected candy category and its count."""
    if not counts:
        return "None", 0
    top_name = next(iter(counts))
    return top_name, counts[top_name]


def format_candy_summary_str(counts: Dict[str, int]) -> str:
    """Format category counts as a concise comma-separated string."""
    if not counts:
        return "None detected"
    return ", ".join(f"{name} ({count})" for name, count in counts.items())


def format_ascii_report(
    candy_counts: Dict[str, int],
    total_candies: int,
    unique_categories: int,
) -> str:
    """
    Format dynamic ASCII candy detection report matching the project specification:

    ========================================
            CANDY DETECTION REPORT          
    ========================================
    Bounty             : 2
    Mars               : 1
    ...
    ----------------------------------------
    Total Candies      : 12
    Unique Categories  : 5
    ========================================
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

