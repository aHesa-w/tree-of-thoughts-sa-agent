"""Tests for tree_of_thoughts.string_to_dict and evaluator.

Note: tree_of_thoughts.agent imports heavy deps (swarms, swarm_models).
Tests that only need base abstractions import from tree_of_thoughts.base directly,
bypassing the __init__.py chain.
"""

import pytest
from tree_of_thoughts.base import string_to_dict


class TestStringToDict:
    def test_json_parse(self):
        result = string_to_dict(
            '{"thought": "test answer", "evaluation": 0.85}'
        )
        assert result["thought"] == "test answer"
        assert result["evaluation"] == 0.85

    def test_json_with_markdown_fence(self):
        result = string_to_dict(
            '```json\n{"thought": "answer", "evaluation": 0.9}\n```'
        )
        assert result["thought"] == "answer"
        assert result["evaluation"] == 0.9

    def test_regex_fallback(self):
        result = string_to_dict(
            'Some prefix text "thought": "fallback answer", "evaluation": 0.75'
        )
        assert result["thought"] == "fallback answer"
        assert result["evaluation"] == 0.75

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            string_to_dict("completely invalid string")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            string_to_dict(123)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            string_to_dict("")


class TestEvaluator:
    def test_auto_detect_code(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type(
            "write a function to sort a list"
        ) == "code"
        assert AutoEvaluator.detect_task_type(
            "fix bug in calculate_total"
        ) == "code"

    def test_auto_detect_math(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type(
            "calculate 2 + 2"
        ) == "math"
        assert AutoEvaluator.detect_task_type(
            "solve equation x + 5 = 10"
        ) == "math"

    def test_auto_detect_reasoning(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type(
            "what is the meaning of life"
        ) == "reasoning"

    def test_math_evaluator_valid(self):
        from tree_of_thoughts.evaluator import MathEvaluator
        score = MathEvaluator.evaluate("(6 / (3 - 2)) * 4 = 24")
        assert score == 1.0

    def test_math_evaluator_invalid(self):
        from tree_of_thoughts.evaluator import MathEvaluator
        score = MathEvaluator.evaluate("2 + 2 = 5")
        assert score is not None
        assert score < 0.5

    def test_code_evaluator_valid(self):
        from tree_of_thoughts.evaluator import CodeEvaluator
        score = CodeEvaluator.evaluate(
            "implement a function",
            "def add(a, b):\n    return a + b",
        )
        assert score == 1.0

    def test_code_evaluator_invalid(self):
        from tree_of_thoughts.evaluator import CodeEvaluator
        score = CodeEvaluator.evaluate(
            "implement a function",
            "def broken(",
        )
        assert score == 0.0


class TestSearchBase:
    def test_search_result_defaults(self):
        from tree_of_thoughts.base import SearchResult
        r = SearchResult()
        assert r.final_answer == ""
        assert r.best_score == 0.0
        assert r.strategy == ""
        assert r.temperature_curve == []

    def test_search_result_fields(self):
        from tree_of_thoughts.base import SearchResult
        r = SearchResult(
            final_answer="42",
            best_score=0.95,
            strategy="SA",
            total_api_calls=10,
            execution_time=5.0,
            parameters={"temp": 1.0},
        )
        assert r.final_answer == "42"
        assert r.best_score == 0.95
        assert r.strategy == "SA"
        assert r.total_api_calls == 10

    def test_search_history(self):
        from tree_of_thoughts.base import SearchHistory
        h = SearchHistory()
        h.steps.append("step1")
        h.evaluations.append(0.5)
        h.decisions.append("accept")
        assert len(h.steps) == 1
