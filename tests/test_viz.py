"""Tests for visualization utilities."""
import pytest


class TestVisualization:
    def test_temperature_curve_empty(self):
        from tree_of_thoughts.viz import ascii_temperature_curve
        result = ascii_temperature_curve([])
        assert "[No temperature data]" in result

    def test_temperature_curve_basic(self):
        from tree_of_thoughts.viz import ascii_temperature_curve
        result = ascii_temperature_curve([1.0, 0.8, 0.6, 0.4, 0.2])
        assert "|" in result
        assert "*" in result

    def test_acceptance_bar_empty(self):
        from tree_of_thoughts.viz import ascii_acceptance_bar
        result = ascii_acceptance_bar([])
        assert "[No acceptance data]" in result

    def test_acceptance_bar_basic(self):
        from tree_of_thoughts.viz import ascii_acceptance_bar
        result = ascii_acceptance_bar(
            [True, True, False, True, False]
        )
        assert "Accept" in result
        assert "Reject" in result

    def test_search_tree_empty(self):
        from tree_of_thoughts.viz import search_tree_text
        result = search_tree_text([])
        assert "[No thoughts to display]" in result

    def test_search_tree_basic(self):
        from tree_of_thoughts.viz import search_tree_text
        thoughts = [
            {"thought": "best answer", "evaluation": 0.95},
            {"thought": "ok answer", "evaluation": 0.60},
            {"thought": "bad answer", "evaluation": 0.20},
        ]
        result = search_tree_text(thoughts)
        assert "best answer" in result
        assert "0.95" in result
        assert "bad answer" in result

    def test_format_search_result_with_sa_data(self):
        from tree_of_thoughts.base import SearchResult
        from tree_of_thoughts.viz import format_search_result

        result = SearchResult(
            final_answer="42",
            best_score=0.95,
            execution_time=5.0,
            total_api_calls=10,
            strategy="SA",
            parameters={"temp": 1.0},
            temperature_curve=[1.0, 0.9, 0.8],
            acceptance_log=[True, False, True],
            thought_graph=[
                {"thought": "a", "evaluation": 0.9},
            ],
        )
        output = format_search_result(result)
        assert "SA" in output
        assert "42" in output
        assert "0.95" in output
