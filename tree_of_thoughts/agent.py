import uuid
import json
import re
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any


TREE_OF_THOUGHTS_SYS_PROMPT = """
You are an expert problem-solving agent designed to not only solve complex problems but also critically evaluate the quality of your thought process and final answers.
Your task is to follow a structured approach to generate solutions, assess your thoughts, and provide a rating for each on a scale of 0.1 to 1.0.
This rating should reflect the accuracy and quality of your reasoning and final answer.

### Instructions:

1. **Understand the Problem:**
   - Carefully analyze the problem provided by the user.
   - Break down the problem into smaller, manageable parts if necessary.
   - Formulate a clear understanding of the problem before proceeding.

2. **Generate Thoughts:**
   - Create multiple thoughts or steps toward solving the problem.
   - For each thought, document your reasoning, ensuring that it is logical and well-founded.

3. **Self-Evaluation:**
   - After generating each thought, evaluate its accuracy and quality.
   - Assign an evaluation score between 0.1 and 1.0. Use the following guidelines:
     - **0.1 to 0.4:** The thought is flawed, inaccurate, or incomplete.
     - **0.5 to 0.7:** The thought is partially correct but may lack detail or full accuracy.
     - **0.8 to 1.0:** The thought is accurate, complete, and well-reasoned.

4. **Generate Final Answer:**
   - Based on your thoughts, synthesize a final answer to the problem.
   - Ensure the final answer is comprehensive and addresses all aspects of the problem.

5. **Final Evaluation:**
   - Evaluate the overall quality and accuracy of your final answer.
   - Provide a final evaluation score based on the same 0.1 to 1.0 scale.

"""


class Thought(BaseModel):
    thought: str
    evaluation: Optional[float] = Field(
        default=None,
        description="The evaluation of the thought. It can be a number between 0.1 and 1.0 being 0.1 the worst and 1.0 the best."
    )
    metadata: dict = Field(default_factory=dict)
    parent_id: Optional[str] = None
    node_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class TotAgent:
    """
    Tree of Thoughts (ToT) core agent.

    A lightweight evaluator wrapper — no internal LLM calls.
    External model drives thought generation; this agent handles
    thought parsing, self-evaluation integration, and optional
    external evaluator scoring.

    Designed to be used as the building block for search strategies
    (DFS, BFS, SA). The search strategies call agent.run() to
    evaluate candidates; candidate generation is expected from
    the external caller.

    Available search strategies:
        - ToTDFSAgent:     Depth-First Search with pruning
        - BFSWithTotAgent: Breadth-First Search with breadth limit
        - ToTSAStrategy:   Simulated Annealing with probabilistic acceptance

    All strategies share the same SearchStrategy interface and return
    a structured SearchResult.

    Usage:
        from tree_of_thoughts import TotAgent, ToTSAStrategy

        # Basic agent — external model provides candidates
        agent = TotAgent()

        # With auto evaluator (math equations / code parse)
        agent = TotAgent(auto_detect_evaluator=True)

        # With custom external evaluator
        agent = TotAgent(
            evaluator=lambda task, thought: 1.0 if "correct" in thought else 0.0,
        )

        sa = ToTSAStrategy(agent=agent)
        result = sa.search("Your task here")
        print(result.final_answer, result.best_score)

        # Compare strategies via benchmark script:
        #   python scripts/benchmark.py --task "use 2 3 4 6 to make 24"

    Attributes:
        id (str): Unique identifier for the agent.
        evaluator: Optional external scoring function (task, thought) -> float.
        auto_detect_evaluator (bool): Whether to auto-detect evaluation strategy.
    """

    def __init__(
        self,
        id: str = uuid.uuid4().hex,
        evaluator: Optional[Callable[[str, str], float]] = None,
        auto_detect_evaluator: bool = False,
    ):
        """
        Initializes a new instance of the TotAgent class.

        Args:
            id (str, optional): The unique identifier for the agent.
            evaluator: Optional external evaluator function. Signature: (task, thought) -> float.
                If provided, its score overrides the LLM self-evaluation.
            auto_detect_evaluator (bool): If True and no evaluator given, use AutoEvaluator
                to automatically choose a scoring strategy based on task type.
        """
        self.id = id
        self.evaluator = evaluator
        self.auto_detect_evaluator = auto_detect_evaluator

    def run(self, thought_string: str) -> dict:
        """
        Parse and evaluate a single thought string. No internal LLM call.

        The external model provides the thought string; this method
        parses it and applies evaluator scoring if configured.

        Args:
            thought_string (str): A JSON or structured string containing
                the thought and optional LLM self-evaluation.

        Returns:
            dict: Parsed result with 'thought' and 'evaluation' keys.
        """
        from tree_of_thoughts.base import string_to_dict
        result = string_to_dict(thought_string)

        # Apply evaluator scoring (overrides LLM self-eval if present)
        if self.evaluator:
            thought_str = result.get("thought", "")
            result["evaluation"] = self.evaluator(thought_string, thought_str)
        elif self.auto_detect_evaluator:
            from tree_of_thoughts.evaluator import AutoEvaluator
            thought_str = result.get("thought", "")
            eval_score = AutoEvaluator.evaluate(thought_string, thought_str)
            if eval_score >= 0:
                result["evaluation"] = eval_score

        return result
