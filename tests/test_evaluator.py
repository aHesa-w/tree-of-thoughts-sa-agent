"""Tests for AutoEvaluator detection and evaluation logic."""

import pytest


class TestAutoEvaluator:
    def test_detect_code_keywords(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type(
            "write a function to sort list"
        ) == "code"
        assert AutoEvaluator.detect_task_type(
            "fix bug in the calculator app"
        ) == "code"
        assert AutoEvaluator.detect_task_type(
            "implement a binary search tree"
        ) == "code"

    def test_detect_math_keywords(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type("solve 2x + 5 = 15") == "math"
        assert AutoEvaluator.detect_task_type(
            "calculate compound interest"
        ) == "math"

    def test_detect_reasoning(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        assert AutoEvaluator.detect_task_type(
            "what is the capital of france"
        ) == "reasoning"

    def test_evaluate_math_valid(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        score = AutoEvaluator.evaluate(
            "calculate 2+2",
            "2 + 2 = 4",
        )
        assert 0.5 <= score <= 1.0

    def test_evaluate_math_invalid(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        score = AutoEvaluator.evaluate(
            "calculate 2+2",
            "2 + 2 = 5",
        )
        assert score <= 0.5

    def test_evaluate_code_valid(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        score = AutoEvaluator.evaluate(
            "write a function add(a,b)",
            "```python\ndef add(a, b):\n    return a + b\n```",
        )
        assert score == 1.0

    def test_evaluate_reasoning_fallback(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        score = AutoEvaluator.evaluate(
            "why is the sky blue",
            "because of rayleigh scattering",
        )
        assert score == -1.0  # fallback signal

    def test_evaluate_no_match_returns_negative_one(self):
        from tree_of_thoughts.evaluator import AutoEvaluator
        score = AutoEvaluator.evaluate(
            "what is 2+2",
            "just some text without numbers",
        )
        # task type is math but thought has no math -> returns None
        # which should be -1
        assert score == -1.0
