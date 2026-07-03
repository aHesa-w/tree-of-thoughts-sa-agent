import uuid
import json
import time
import os
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from tree_of_thoughts.agent import TotAgent
from tree_of_thoughts.base import SearchStrategy, SearchResult, SearchHistory


class ToTDFSAgent(SearchStrategy):
    """
    Depth-First Search using the TotAgent, with pruning based on evaluation scores.

    Implements SearchStrategy so it can be used interchangeably with BFS and SA.
    The original run() method is preserved for backward compatibility.
    """

    def __init__(
        self,
        agent: TotAgent,
        threshold: float,
        max_loops: int,
        prune_threshold: float = 0.5,
        number_of_agents: int = 3,
        autosave_on: bool = True,
        id: str = uuid.uuid4().hex,
        *args,
        **kwargs,
    ):
        self.id = id
        self.agent = agent
        self.threshold = threshold
        self.max_loops = max_loops
        self.prune_threshold = prune_threshold
        self.all_thoughts: List[Dict[str, Any]] = []
        self.pruned_branches: List[Dict[str, Any]] = []
        self.number_of_agents = number_of_agents
        self.autosave_on = autosave_on
        self._api_calls: int = 0
        self._history = SearchHistory()

        self.agent.max_loops = max_loops

    def dfs(self, state: str, step: int = 0) -> Optional[Dict[str, Any]]:
        logger.info(f"Starting DFS for state: {state}")

        if step >= self.max_loops:
            return None

        logger.info(
            f"Generating {self.number_of_agents} thoughts for state: {state}"
        )

        with ThreadPoolExecutor(max_workers=self.number_of_agents) as executor:
            next_thoughts = list(
                executor.map(self.agent.run, [state] * self.number_of_agents)
            )
        self._api_calls += self.number_of_agents

        next_thoughts.sort(key=lambda x: x["evaluation"], reverse=False)

        for thought in next_thoughts:
            if thought["evaluation"] > self.prune_threshold:
                self.all_thoughts.append(thought)
                result = self.dfs(thought["thought"], step + 1)

                if result and result["evaluation"] > self.threshold:
                    return result
            else:
                self._prune_thought(thought)

        logger.info(f"Finished DFS for state: {state}")
        return None

    def _prune_thought(self, thought: Dict[str, Any]):
        self.pruned_branches.append(
            {
                "thought": thought["thought"],
                "evaluation": thought["evaluation"],
                "reason": "Evaluation score below threshold",
            }
        )

    def search(self, initial_state: str) -> SearchResult:
        """Implement SearchStrategy.search(), returning a structured SearchResult."""
        start_time = time.time()
        self.all_thoughts = []
        self.pruned_branches = []
        self._api_calls = 0

        initial_thoughts = self.dfs(initial_state)

        for i in range(1, self.max_loops):
            if initial_thoughts:
                next_task = initial_thoughts["thought"]
                initial_thoughts = self.dfs(next_task, step=i)
            else:
                break

        self.all_thoughts.sort(key=lambda x: x["evaluation"], reverse=False)
        elapsed = time.time() - start_time

        best_thought = self.all_thoughts[-1] if self.all_thoughts else None

        result = SearchResult(
            final_answer=best_thought["thought"] if best_thought else "",
            best_score=best_thought["evaluation"] if best_thought else 0.0,
            execution_time=elapsed,
            total_api_calls=self._api_calls,
            strategy="DFS",
            parameters={
                "threshold": self.threshold,
                "max_loops": self.max_loops,
                "prune_threshold": self.prune_threshold,
                "number_of_agents": self.number_of_agents,
            },
            thought_graph=[
                {
                    "thought": t["thought"],
                    "evaluation": t["evaluation"],
                }
                for t in self.all_thoughts
            ],
        )

        if self.autosave_on:
            output_dir = "tree_of_thoughts_runs"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(
                output_dir, f"tree_of_thoughts_run{self.id}.json"
            )
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "final_thoughts": self.all_thoughts,
                        "pruned_branches": self.pruned_branches,
                        "highest_rated_thought": best_thought,
                    },
                    f,
                    indent=4,
                )

        return result

    def run(self, task: str, *args, **kwargs) -> str:
        """Original run() method. Returns JSON string for backward compatibility."""
        result = self.search(task)
        tree_dict = {
            "final_thoughts": self.all_thoughts,
            "pruned_branches": self.pruned_branches,
            "highest_rated_thought": (
                self.all_thoughts[-1] if self.all_thoughts else None
            ),
        }
        json_string = json.dumps(tree_dict, indent=4)

        if self.autosave_on:
            output_dir = "tree_of_thoughts_runs"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(
                output_dir, f"tree_of_thoughts_run{self.id}.json"
            )
            with open(file_path, "w") as f:
                f.write(json_string)

        return json_string
