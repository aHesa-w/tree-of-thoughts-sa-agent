"""Tests for SA strategy (mocked LLM calls)."""

from unittest.mock import MagicMock, patch


class MockAgent:
    """Mimics TotAgent.run() for SA tests."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def run(self, task):
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return {"thought": "mock thought", "evaluation": 0.5}

    def __call__(self, task):
        return self.run(task)


class TestToTSAStrategy:
    def test_search_returns_result(self):
        from tree_of_thoughts.sa import ToTSAStrategy

        agent = MockAgent()
        sa = ToTSAStrategy(
            agent=agent,
            initial_temperature=1.0,
            cooling_rate=0.9,
            min_temperature=0.5,
            parallel_candidates=1,
            max_iterations=3,
        )
        result = sa.search("test task")
        assert result.strategy == "SA"
        assert result.total_api_calls > 0
        assert result.execution_time >= 0
        # max_iterations=3 means initial + up to 3 loop iters
        assert len(result.temperature_curve) >= 1

    def test_accepts_better_solution(self):
        from tree_of_thoughts.sa import ToTSAStrategy

        # Responses: first is 0.5, then 0.9 (better -> must accept)
        agent = MockAgent(responses=[
            {"thought": "initial", "evaluation": 0.5},
            {"thought": "better", "evaluation": 0.9},
        ])
        sa = ToTSAStrategy(
            agent=agent,
            initial_temperature=1.0,
            cooling_rate=0.5,
            min_temperature=0.01,
            parallel_candidates=1,
            max_iterations=2,
        )
        result = sa.search("test")
        assert result.best_score == 0.9

    def test_temperature_cooling(self):
        from tree_of_thoughts.sa import ToTSAStrategy

        agent = MockAgent()
        sa = ToTSAStrategy(
            agent=agent,
            initial_temperature=1.0,
            cooling_rate=0.5,
            min_temperature=0.01,
            parallel_candidates=1,
            max_iterations=5,
        )
        sa.search("test")
        assert len(sa._temperature_curve) > 0
        last_temp = sa._temperature_curve[-1]
        assert last_temp < sa._temperature_curve[0]

    def test_acceptance_log_populated(self):
        from tree_of_thoughts.sa import ToTSAStrategy

        agent = MockAgent()
        sa = ToTSAStrategy(
            agent=agent,
            initial_temperature=1.0,
            max_iterations=3,
            parallel_candidates=1,
        )
        result = sa.search("test")
        assert len(result.acceptance_log) > 0
