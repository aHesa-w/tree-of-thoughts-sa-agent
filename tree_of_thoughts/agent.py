import uuid
from pydantic import BaseModel, Field
from typing import Optional, Callable
from swarms import Agent
from swarm_models import OpenAIFunctionCaller
from typing import Any
import os
from dotenv import load_dotenv

load_dotenv()


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
    Represents a Tree of Thoughts (ToT) agent.

    Attributes:
        id (str): The unique identifier for the agent.
        max_loops (int): The maximum number of loops the agent can run.
        model (OpenAIFunctionCaller): The OpenAI function caller for the agent.
        agent (Agent): The agent responsible for running tasks.

    Methods:
        run(task: str) -> dict: Runs a task using the agent and returns the output as a dictionary.
    """

    def __init__(
        self,
        id: str = uuid.uuid4().hex,
        max_loops: int = None,
        use_openai_caller: bool = True,
        model: Optional[Any] = None,
        evaluator: Optional[Callable[[str, str], float]] = None,
        auto_detect_evaluator: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes a new instance of the TotAgent class.

        Args:
            id (str, optional): The unique identifier for the agent. Defaults to a randomly generated UUID.
            max_loops (int, optional): The maximum number of loops the agent can run. Defaults to None.
            use_openai_caller (bool): Whether to use the default OpenAI function caller.
            model: Optional custom model instance.
            evaluator: Optional external evaluator function. Signature: (task, thought) -> float.
                If provided, its score overrides the LLM self-evaluation.
            auto_detect_evaluator (bool): If True and no evaluator given, use AutoEvaluator
                to automatically choose a scoring strategy based on task type.
        """
        self.id = id
        self.max_loops = max_loops
        self.model = model
        self.evaluator = evaluator
        self.auto_detect_evaluator = auto_detect_evaluator

        if use_openai_caller:
            self.model = OpenAIFunctionCaller(
                system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
                base_model=Thought,
                parallel_tool_calls=False,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=3000,
            )

        self.agent = Agent(
            agent_name=f"ToT-Agent-{self.id}",
            system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
            llm=self.model,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"tot_agent{self.id}.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            *args,
            **kwargs,
        )

    def run(self, task: Any) -> dict:
        """
        Runs a task using the agent and returns the output as a dictionary.

        If an external evaluator is registered, its score overrides the LLM self-evaluation.
        If auto_detect_evaluator is enabled and no evaluator is set, AutoEvaluator is used.

        Args:
            task (str): The task to be run by the agent.

        Returns:
            dict: The output of the agent as a dictionary.
        """
        agent_output = self.agent.run(task)
        from tree_of_thoughts.base import string_to_dict
        result = string_to_dict(agent_output)

        if self.evaluator:
            task_str = task if isinstance(task, str) else str(task)
            thought_str = result.get("thought", "")
            result["evaluation"] = self.evaluator(task_str, thought_str)
        elif self.auto_detect_evaluator:
            from tree_of_thoughts.evaluator import AutoEvaluator
            task_str = task if isinstance(task, str) else str(task)
            thought_str = result.get("thought", "")
            eval_score = AutoEvaluator.evaluate(task_str, thought_str)
            if eval_score >= 0:
                result["evaluation"] = eval_score

        return result
