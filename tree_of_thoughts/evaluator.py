"""Task-type-aware evaluators for Tree of Thoughts.

AutoEvaluator examines the task string and routes to the best evaluator:
- Math tasks: evaluates arithmetic expression correctness
- Code tasks: evaluates code parse-ability and structure
- Reasoning tasks: falls back to LLM self-evaluation (returns -1)
"""

import ast
import re
from typing import Optional


class MathEvaluator:
    """Evaluates arithmetic/math thought correctness."""

    @staticmethod
    def evaluate(thought: str) -> Optional[float]:
        """Return 1.0 if the thought contains a valid math expression that
        can be safely parsed, 0.0 if clearly malformed, None if uncertain."""
        # Extract the most math-like expression
        # Look for equations after "=" or standalone arithmetic
        lines = thought.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            # Remove trailing/leading noise
            line = re.sub(r"^[^0-9(+\-]*", "", line)
            line = re.sub(r"[^0-9)\s]*$", "", line)
            if "=" in line:
                # Try evaluating both sides
                parts = line.split("=", 1)
                try:
                    left = eval(parts[0].strip(), {"__builtins__": {}}, {})
                    right = eval(parts[1].strip(), {"__builtins__": {}}, {})
                    if abs(left - right) < 1e-9:
                        return 1.0
                    return 0.3
                except Exception:
                    continue
            # No '=' sign — try as standalone expression
            try:
                result = eval(line, {"__builtins__": {}}, {})
                if isinstance(result, (int, float)):
                    return 0.8  # valid expression, but no verification target
            except Exception:
                continue
        return None


class CodeEvaluator:
    """Evaluates code thought correctness by checking parse-ability."""

    @staticmethod
    def evaluate(task: str, thought: str) -> Optional[float]:
        """Return score based on code structure:
        1.0 — cleanly parses as valid Python
        0.5 — partial code with some structure
        0.0 — clearly malformed
        None — not code-like
        """
        code = CodeEvaluator._extract_code_block(thought)
        if not code:
            # No code block found; maybe the whole thought is code
            code = thought.strip()

        if not code:
            return None

        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            # Count signs of "close to valid"
            lines = code.strip().split("\n")
            good_lines = sum(
                1 for l in lines if l.strip() and not l.strip().startswith("#")
            )
            if good_lines > 2:
                return 0.5
            return 0.0

    @staticmethod
    def _extract_code_block(text: str) -> Optional[str]:
        """Extract content from markdown code fences."""
        match = re.search(
            r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL
        )
        return match.group(1).strip() if match else None


class AutoEvaluator:
    """Automatically selects evaluator based on task type detection."""

    _CODE_KEYWORDS = [
        "code", "function", "implement", "bug", "refactor",
        "def ", "class ", "import ", "return ", "async ",
        "write a", "fix", "debug",
    ]
    _MATH_KEYWORDS = [
        "calculate", "equation", "math", "arithmetic",
        "+", "-", "*", "/", "=", "solve",
    ]

    @staticmethod
    def detect_task_type(task: str) -> str:
        task_lower = task.lower()
        code_score = sum(
            1 for kw in AutoEvaluator._CODE_KEYWORDS if kw in task_lower
        )
        math_score = sum(
            1 for kw in AutoEvaluator._MATH_KEYWORDS if kw in task_lower
        )
        if code_score > math_score:
            return "code"
        if math_score >= code_score and math_score > 0:
            return "math"
        return "reasoning"

    @staticmethod
    def evaluate(task: str, thought: str) -> float:
        """Run the best-guess evaluator. Returns -1 if no evaluator applies
        (caller should fall back to LLM self-evaluation)."""
        task_type = AutoEvaluator.detect_task_type(task)
        if task_type == "math":
            score = MathEvaluator.evaluate(thought)
            if score is not None:
                return score
        elif task_type == "code":
            score = CodeEvaluator.evaluate(task, thought)
            if score is not None:
                return score
        return -1.0
