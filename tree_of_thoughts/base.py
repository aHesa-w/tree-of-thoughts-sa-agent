"""Base abstractions for Tree of Thoughts search strategies.

Provides SearchStrategy (ABC), SearchResult, SearchHistory, and string_to_dict
that are shared across DFS, BFS, and SA implementations.
"""

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


def string_to_dict(thought_string: str) -> dict:
    """
    Safely parse a thought string into a dict.
    Uses json.loads() with fallback regex extraction.
    No heavy dependencies -- safe to import anywhere.
    """
    if not isinstance(thought_string, str):
        raise ValueError(f"Expected string, got {type(thought_string)}")

    s = thought_string.strip()
    # Remove markdown code block fences
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)

    # Try JSON parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    thought_match = re.search(r'"thought"\s*:\s*"([^"]+)"', s)
    eval_match = re.search(r'"evaluation"\s*:\s*([\d.]+)', s)
    if thought_match and eval_match:
        return {
            "thought": thought_match.group(1),
            "evaluation": float(eval_match.group(1)),
        }

    # Last resort: try ast.literal_eval (safe eval of simple dict literals)
    if s.startswith("{") and s.endswith("}"):
        try:
            result = ast.literal_eval(s)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError, MemoryError):
            pass

    raise ValueError(
        f"Cannot parse thought string (first 200 chars): {thought_string[:200]}"
    )


@dataclass
class SearchResult:
    """Standard result returned by any search strategy."""

    final_answer: str = ""
    best_score: float = 0.0
    execution_time: float = 0.0
    total_api_calls: int = 0
    strategy: str = ""
    parameters: dict = field(default_factory=dict)
    thought_graph: List[dict] = field(default_factory=list)
    # SA-specific fields (empty for BFS/DFS)
    temperature_curve: List[float] = field(default_factory=list)
    acceptance_log: List[bool] = field(default_factory=list)
    iteration_scores: List[float] = field(default_factory=list)


@dataclass
class SearchHistory:
    """Track per-step decisions for post-hoc analysis."""

    steps: List[str] = field(default_factory=list)
    evaluations: List[float] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class SearchStrategy(ABC):
    """Abstract base class for all ToT search strategies."""

    @abstractmethod
    def search(self, initial_state: str) -> SearchResult:
        """Run the search strategy and return a structured result."""
        ...

    def get_history(self) -> SearchHistory:
        """Return per-step history for post-hoc analysis."""
        return SearchHistory()
