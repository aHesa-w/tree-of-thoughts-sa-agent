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


class TestTotAgentRun:
    """TotAgent.run() 端到端测试 — 覆盖 3 条执行路径 + 异常输入。

    TotAgent.run() 接收 LLM 生成的 JSON 思维字符串，完成解析和评分覆盖。
    三条执行路径：
      - 路径1：自定义 evaluator 覆盖评分
      - 路径2：auto_detect_evaluator=True → AutoEvaluator 覆盖
      - 路径3：无覆盖 → 保留 LLM 自评（string_to_dict 原始值）
    """

    VALID_JSON = '{"thought": "2 + 2 = 4", "evaluation": 0.8}'
    VALID_JSON_LOW = '{"thought": "2 + 2 = 5", "evaluation": 0.9}'
    MATH_TASK = "calculate 2 + 2"
    MATH_MARKDOWN = '```json\n{"thought": "2 + 2 = 4", "evaluation": 0.8}\n```'
    REGEX_INPUT = 'Some text "thought": "guess answer", "evaluation": 0.4'

    # ── 路径3：无覆盖（默认），保留 LLM 自评 ──

    def test_run_default_preserves_evaluation(self):
        """默认 TotAgent：string_to_dict 解析后直接返回，保留 raw evaluation。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        result = agent.run(self.VALID_JSON)
        assert result["thought"] == "2 + 2 = 4"
        assert result["evaluation"] == 0.8  # 保留 LLM 自评

    def test_run_markdown_fence(self):
        """Markdown code fence 内 JSON 正常解析。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        result = agent.run(self.MATH_MARKDOWN)
        assert result["thought"] == "2 + 2 = 4"
        assert result["evaluation"] == 0.8

    def test_run_regex_fallback(self):
        """非标准 JSON 走正则回退路径。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        result = agent.run(self.REGEX_INPUT)
        assert result["thought"] == "guess answer"
        assert result["evaluation"] == 0.4

    # ── 路径1：自定义 evaluator 覆盖 ──

    def test_run_with_evaluator_overrides_score(self):
        """自定义 evaluator(lambda) 返回的评分覆盖 LLM 自评。"""
        from tree_of_thoughts.agent import TotAgent

        def my_eval(task, thought):
            return 1.0 if "4" in thought else 0.0

        agent = TotAgent(evaluator=my_eval)
        result = agent.run(self.VALID_JSON)
        assert result["thought"] == "2 + 2 = 4"
        # 原始 evaluation 是 0.8，但被 evaluator 覆盖为 1.0
        assert result["evaluation"] == 1.0

    def test_run_with_evaluator_low_score(self):
        """evaluator 对错误结果打低分。"""
        from tree_of_thoughts.agent import TotAgent

        def my_eval(task, thought):
            return 0.0 if "5" in thought else 1.0

        agent = TotAgent(evaluator=my_eval)
        # "2 + 2 = 5" 中包含 "5"
        result = agent.run(self.VALID_JSON_LOW)
        assert result["evaluation"] == 0.0

    # ── 路径2：auto_detect_evaluator=True → AutoEvaluator 覆盖 ──

    def test_run_auto_eval_math_valid(self):
        """数学任务 + auto_detect=True → MathEvaluator 打高分。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent(auto_detect_evaluator=True)
        # task 是 "calculate 2 + 2"，thought 是 "2 + 2 = 4"
        result = agent.run(self.VALID_JSON)
        assert result["thought"] == "2 + 2 = 4"
        # AutoEvaluator 检测为 math → MathEvaluator 评分 1.0
        assert result["evaluation"] == 1.0

    def test_run_auto_eval_math_invalid_thought(self):
        """auto_detect: thought 不含数学内容 → AutoEvaluator 返回 -1 → 保留原始评分。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent(auto_detect_evaluator=True)
        # task 含 math 关键词，但 thought 无数学内容
        result = agent.run('{"thought": "I think the answer is 4", "evaluation": 0.5}')
        # AutoEvaluator.evaluate 返回 -1（无匹配），所以保留原始 0.5
        assert result["evaluation"] == 0.5

    def test_run_auto_eval_code_valid_function(self):
        """代码任务 + auto_detect=True → CodeEvaluator 评分 1.0。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent(auto_detect_evaluator=True)
        thought_json = '{"thought": "```python\\ndef add(a, b):\\n    return a + b\\n```", "evaluation": 0.7}'
        result = agent.run(thought_json)
        assert result["evaluation"] == 1.0

    def test_run_auto_eval_reasoning_fallback(self):
        """推理任务 + auto_detect=True → AutoEvaluator 返回 -1 → 保留原始评分。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent(auto_detect_evaluator=True)
        thought_json = '{"thought": "The sky is blue because of Rayleigh scattering", "evaluation": 0.6}'
        result = agent.run(thought_json)
        assert result["evaluation"] == 0.6  # -1 回退，保留原始值

    # ── 异常输入 ──

    def test_run_invalid_json_raises(self):
        """无效 JSON 字符串 → string_to_dict 抛 ValueError。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        with pytest.raises(ValueError):
            agent.run("this is not valid json at all")

    def test_run_empty_string_raises(self):
        """空字符串 → ValueError。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        with pytest.raises(ValueError):
            agent.run("")

    def test_run_non_string_raises(self):
        """非字符串参数 → ValueError。"""
        from tree_of_thoughts.agent import TotAgent
        agent = TotAgent()
        with pytest.raises(ValueError):
            agent.run(12345)


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
