"""Simulated Annealing search strategy for Tree of Thoughts.

ToTSAStrategy combines the ToT multi-candidate generation with SA's
probabilistic acceptance of sub-optimal solutions to escape local optima.
"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from loguru import logger

from tree_of_thoughts.agent import TotAgent
from tree_of_thoughts.base import SearchStrategy, SearchResult, SearchHistory
from tree_of_thoughts.evaluator import AutoEvaluator


class ToTSAStrategy(SearchStrategy):
    """Simulated Annealing search strategy for Tree of Thoughts.

    At each iteration, parallel_candidates thoughts are generated from the
    current state. The best candidate is evaluated; if it improves the score
    it is always accepted. If it is worse, it is accepted with a probability
    that decreases as temperature cools, allowing escape from local optima.

    Args:
        agent: TotAgent instance used to generate and score thoughts.
        initial_temperature: Starting temperature (default 1.0).
        cooling_rate: Multiplicative cooling factor per iteration (default 0.95).
        min_temperature: Temperature floor (default 0.01).
        parallel_candidates: How many candidates to generate per iteration (default 3).
        reheat_threshold: Number of consecutive no-improvement iterations before reheat.
        reheat_factor: How much to reheat, as fraction of the gap to 1.0.
        early_stop_patience: Consecutive no-improvement + low temp to force stop.
        max_iterations: Hard limit on iterations.
        use_auto_evaluator: If True, apply AutoEvaluator on each candidate.
    """

    def __init__(
        self,
        agent: TotAgent,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        parallel_candidates: int = 3,
        reheat_threshold: int = 5,
        reheat_factor: float = 0.5,
        early_stop_patience: int = 10,
        max_iterations: int = 100,
        use_auto_evaluator: bool = False,
    ):
        self.agent = agent
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.parallel_candidates = parallel_candidates
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.early_stop_patience = early_stop_patience
        self.max_iterations = max_iterations
        self.use_auto_evaluator = use_auto_evaluator
        self._api_calls = 0
        self._history = SearchHistory()
        self._temperature_curve: List[float] = []
        self._acceptance_log: List[bool] = []
        self._iteration_scores: List[float] = []

    def search(self, initial_state: str) -> SearchResult:
        """Run the SA search and return a structured SearchResult."""
        start_time = time.time()
        self._reset_state()

        # --- initial solution ---
        current = self.agent.run(initial_state)
        self._api_calls += 1
        current_thought = current.get("thought", initial_state)
        current_score = self._resolve_score(initial_state, current)

        best_thought = current_thought
        best_score = current_score
        no_improve_count = 0
        temperature = self.initial_temperature

        self._record_iteration(temperature, current_score, True)

        # --- main SA loop ---
        for iteration in range(1, self.max_iterations + 1):
            if temperature < self.min_temperature:
                logger.info(
                    f"Temperature {temperature:.4f} below min. Stopping."
                )
                break

            # Generate candidates
            candidates = self._generate_candidates(initial_state)
            if not candidates:
                break

            # Pick the best candidate
            candidate = self._pick_best(candidates, initial_state)
            new_thought = candidate["thought"]
            new_score = candidate["evaluation"]

            # SA acceptance criterion
            delta = new_score - current_score
            accepted = False
            if delta > 0:
                accepted = True
            else:
                prob = math.exp(delta / max(temperature, 1e-9))
                if random.random() < prob:
                    accepted = True

            if accepted:
                current_thought = new_thought
                current_score = new_score
                if new_score > best_score:
                    best_thought = new_thought
                    best_score = new_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            self._record_iteration(temperature, current_score, accepted)

            # Cool down
            temperature *= self.cooling_rate

            # Reheat if stuck
            if no_improve_count >= self.reheat_threshold and temperature < 0.5:
                temperature += self.reheat_factor * (1.0 - temperature)
                logger.info(
                    f"Reheat triggered at iter {iteration}, "
                    f"T -> {temperature:.4f}"
                )
                no_improve_count = 0

            # Early stop
            if (
                no_improve_count >= self.early_stop_patience
                and temperature < 0.1
            ):
                logger.info(
                    f"Early stop at iter {iteration} "
                    f"(no improve for {no_improve_count}, T={temperature:.4f})"
                )
                break

        elapsed = time.time() - start_time

        return SearchResult(
            final_answer=best_thought,
            best_score=best_score,
            execution_time=elapsed,
            total_api_calls=self._api_calls,
            strategy="SA",
            parameters={
                "initial_temperature": self.initial_temperature,
                "cooling_rate": self.cooling_rate,
                "min_temperature": self.min_temperature,
                "parallel_candidates": self.parallel_candidates,
                "reheat_threshold": self.reheat_threshold,
                "reheat_factor": self.reheat_factor,
                "early_stop_patience": self.early_stop_patience,
                "max_iterations": self.max_iterations,
                "use_auto_evaluator": self.use_auto_evaluator,
            },
            temperature_curve=list(self._temperature_curve),
            acceptance_log=list(self._acceptance_log),
            iteration_scores=list(self._iteration_scores),
        )

    def get_history(self) -> SearchHistory:
        return self._history

    # ---- internal helpers ----

    def _reset_state(self):
        self._api_calls = 0
        self._temperature_curve = []
        self._acceptance_log = []
        self._iteration_scores = []
        self._history = SearchHistory()

    def _generate_candidates(self, task: str) -> List[Dict[str, Any]]:
        """Generate parallel_candidates thoughts and return them."""
        with ThreadPoolExecutor(
            max_workers=self.parallel_candidates
        ) as executor:
            results = list(
                executor.map(
                    self.agent.run, [task] * self.parallel_candidates
                )
            )
        self._api_calls += self.parallel_candidates
        return [r for r in results if r is not None]

    def _pick_best(
        self, candidates: List[Dict[str, Any]], task: str
    ) -> Dict[str, Any]:
        """Pick the candidate with the highest evaluation score."""
        # Apply auto-evaluator if enabled
        if self.use_auto_evaluator:
            for c in candidates:
                score = AutoEvaluator.evaluate(task, c.get("thought", ""))
                if score >= 0:
                    c["evaluation"] = score

        candidates.sort(key=lambda x: x.get("evaluation", 0), reverse=True)
        return candidates[0]

    def _resolve_score(self, task: str, result: Dict[str, Any]) -> float:
        """Get score with optional AutoEvaluator override."""
        if self.use_auto_evaluator:
            score = AutoEvaluator.evaluate(
                task, result.get("thought", "")
            )
            if score >= 0:
                return score
        return result.get("evaluation", 0.0)

    def _record_iteration(
        self, temperature: float, score: float, accepted: bool
    ):
        self._temperature_curve.append(temperature)
        self._iteration_scores.append(score)
        self._acceptance_log.append(accepted)
        self._history.steps.append(str(score))
        self._history.evaluations.append(score)
        self._history.decisions.append("accept" if accepted else "reject")
