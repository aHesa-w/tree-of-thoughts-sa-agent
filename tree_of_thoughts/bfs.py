import json
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from tree_of_thoughts.agent import TotAgent
from tree_of_thoughts.base import string_to_dict
from tree_of_thoughts.base import SearchStrategy, SearchResult, SearchHistory

load_dotenv()


class BFSWithTotAgent(SearchStrategy):
    """
    Breadth-First Search using the TotAgent, based on the ToT-BFS algorithm.

    Implements SearchStrategy so it can be used interchangeably with DFS and SA.
    The original run() method is preserved for backward compatibility.
    """

    def __init__(
        self,
        agent: TotAgent,
        max_loops: int,
        breadth_limit: int,
        number_of_agents: int = 3,
        autosave_on: bool = True,
        id: str = uuid.uuid4().hex,
    ):
        self.id = id
        self.agent = agent
        self.max_loops = max_loops
        self.breadth_limit = breadth_limit
        self.number_of_agents = number_of_agents
        self.autosave_on = autosave_on
        self.all_thoughts: List[Dict[str, Any]] = []
        self._api_calls: int = 0
        self._history = SearchHistory()

    def bfs(self, state: str) -> Optional[Dict[str, Any]]:
        """Perform BFS with breadth limit based on evaluation scores."""
        S = [state]

        for t in range(1, self.max_loops + 1):
            logger.info(f"Step {t}/{self.max_loops}: Expanding states.")
            S_prime = self._generate_new_states(S)

            if not S_prime:
                logger.info(
                    f"No valid thoughts generated at step {t}. Stopping BFS."
                )
                break

            V = self._evaluate_states(S_prime)
            self._log_and_store_thoughts(S_prime, V)
            S = self._select_best_states(S_prime, V)

        return self._generate_final_answer(S)

    def _generate_new_states(self, S: List[str]) -> List[Dict[str, Any]]:
        S_prime = []
        for s in S:
            with ThreadPoolExecutor() as executor:
                new_thoughts = list(
                    executor.map(self.agent.run, [s] * self.number_of_agents)
                )
                self._api_calls += self.number_of_agents
                S_prime.extend(
                    [
                        [s, thought]
                        for thought in new_thoughts
                        if thought is not None
                    ]
                )
        return S_prime

    def _evaluate_states(self, S_prime: List[Dict[str, Any]]) -> List[float]:
        return [thought["evaluation"] for _, thought in S_prime]

    def _log_and_store_thoughts(
        self, S_prime: List[Dict[str, Any]], V: List[float]
    ):
        for i, (_, thought) in enumerate(S_prime):
            self.all_thoughts.append(thought)

    def _select_best_states(
        self, S_prime: List[Dict[str, Any]], V: List[float]
    ) -> List[str]:
        state_evaluation_pairs = list(zip(S_prime, V))
        state_evaluation_pairs.sort(key=lambda x: x[1], reverse=True)
        best_states = [
            pair[0][1]["thought"]
            for pair in state_evaluation_pairs[: self.breadth_limit]
        ]
        return best_states

    def _generate_final_answer(self, S: List[str]) -> Optional[Dict[str, Any]]:
        if not S:
            return None
        final_state = max(S, key=lambda s: self.agent.run(s)["evaluation"])
        self._api_calls += 1
        return self.agent.run(final_state)

    def _run_agent(self, task: str) -> Optional[Dict[str, Any]]:
        try:
            return self.agent.run(task)
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
        return None

    def search(self, initial_state: str) -> SearchResult:
        """Implement SearchStrategy.search(), returning a structured SearchResult."""
        start_time = time.time()
        self.all_thoughts = []
        self._api_calls = 0

        final_thought = self.bfs(initial_state)

        self.all_thoughts.sort(key=lambda x: x["evaluation"], reverse=False)
        elapsed = time.time() - start_time

        best_thought = self.all_thoughts[-1] if self.all_thoughts else None

        result = SearchResult(
            final_answer=(
                final_thought["thought"]
                if final_thought
                else (best_thought["thought"] if best_thought else "")
            ),
            best_score=(
                final_thought["evaluation"]
                if final_thought
                else (best_thought["evaluation"] if best_thought else 0.0)
            ),
            execution_time=elapsed,
            total_api_calls=self._api_calls,
            strategy="BFS",
            parameters={
                "max_loops": self.max_loops,
                "breadth_limit": self.breadth_limit,
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

        return result

    def run(self, task: str) -> str:
        """Original run() method. Returns JSON string for backward compatibility."""
        final_thought = self.bfs(task)

        self.all_thoughts.sort(key=lambda x: x["evaluation"], reverse=False)

        tree_dict = {
            "all_thoughts": self.all_thoughts,
            "final_thought": final_thought,
        }

        return json.dumps(tree_dict, indent=4)
