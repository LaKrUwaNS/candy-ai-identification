"""Unit tests for candy counting, ranking, and report generation."""

import unittest
from core.candy_counter import (
    count_candies,
    format_ascii_report,
    format_candy_summary_str,
    get_top_candy,
)


class TestCandyCounter(unittest.TestCase):
    def test_count_candies_empty(self):
        counts = count_candies([])
        self.assertEqual(counts, {})

    def test_count_candies_tally_and_sort(self):
        categories = [
            "Snickers", "Bounty", "Snickers", "Twix_clasic",
            "Mars", "Snickers", "Bounty", "Skittles"
        ]
        counts = count_candies(categories)

        expected = {
            "Snickers": 3,
            "Bounty": 2,
            "Mars": 1,
            "Skittles": 1,
            "Twix_clasic": 1,
        }
        self.assertEqual(counts, expected)
        # Verify first item is indeed Snickers
        top_name, top_cnt = get_top_candy(counts)
        self.assertEqual(top_name, "Snickers")
        self.assertEqual(top_cnt, 3)

    def test_get_top_candy_empty(self):
        name, count = get_top_candy({})
        self.assertEqual(name, "None")
        self.assertEqual(count, 0)

    def test_format_candy_summary_str(self):
        counts = {"Snickers": 3, "Mars": 1}
        summary = format_candy_summary_str(counts)
        self.assertEqual(summary, "Snickers (3), Mars (1)")

        self.assertEqual(format_candy_summary_str({}), "None detected")

    def test_format_ascii_report(self):
        counts = {"Snickers": 3, "Skittles": 2}
        report = format_ascii_report(counts, total_candies=5, unique_categories=2)

        self.assertIn("CANDY DETECTION REPORT", report)
        self.assertIn("Snickers", report)
        self.assertIn("Skittles", report)
        self.assertIn("Total Candies      : 5", report)
        self.assertIn("Unique Categories  : 2", report)

    def test_format_ascii_report_empty(self):
        report = format_ascii_report({}, total_candies=0, unique_categories=0)
        self.assertIn("No candy objects detected.", report)
        self.assertIn("Total Candies      : 0", report)


if __name__ == "__main__":
    unittest.main()

