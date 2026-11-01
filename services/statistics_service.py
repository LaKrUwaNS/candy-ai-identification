"""Statistics computation service for session metrics and category trends."""

from collections import Counter
from typing import Any, Dict, List, Tuple


class StatisticsService:
    """Computes session metrics and aggregations from actual detection data."""

    @staticmethod
    def calculate_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute high-level session KPI metrics from detection history.
        Strictly computes from actual history; zero fake or hardcoded values.
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
        total_candies = sum(r.get("total_candies", 0) for r in history)

        aggregated_counts: Counter = Counter()
        total_inference_time = 0.0

        for r in history:
            counts = r.get("candy_counts", {})
            aggregated_counts.update(counts)
            total_inference_time += r.get("inference_ms", 0.0)

        unique_categories_count = len(aggregated_counts)

        if aggregated_counts:
            top_candy, top_count = aggregated_counts.most_common(1)[0]
        else:
            top_candy, top_count = "None", 0

        avg_candies = round(total_candies / total_images, 1) if total_images > 0 else 0.0
        avg_inf = round(total_inference_time / total_images, 1) if total_images > 0 else 0.0

        # Sort distribution by count descending, then alphabetically
        sorted_dist = dict(
            sorted(aggregated_counts.items(), key=lambda item: (-item[1], item[0]))
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
            "avg_inference_time_ms": avg_inf,
        }

    @staticmethod
    def get_timeline_series(
        history: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[int], List[str]]:
        """Extract chronological arrays (timestamps, counts, filenames) for time-series charts."""
        chrono = list(reversed(history))
        timestamps = [r["timestamp"] for r in chrono]
        counts = [r["total_candies"] for r in chrono]
        filenames = [r["filename"] for r in chrono]
        return timestamps, counts, filenames

