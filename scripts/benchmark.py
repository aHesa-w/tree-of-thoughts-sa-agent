#!/usr/bin/env python3
"""Benchmark script for comparing ToT search strategies.

Usage:
    python scripts/benchmark.py --strategies dfs bfs sa

Output:
    a comparison table and individual SearchResult details.
"""

import argparse
from typing import Dict

from tree_of_thoughts import TotAgent, ToTDFSAgent, BFSWithTotAgent, ToTSAStrategy
from tree_of_thoughts.base import SearchResult


def run_benchmark(
    task: str,
    strategies: list,
    evaluator: str = "self",
) -> Dict[str, SearchResult]:
    """Run all requested strategies on the same task and return results."""
    agent = TotAgent()
    if evaluator == "auto":
        agent.auto_detect_evaluator = True

    results: Dict[str, SearchResult] = {}

    if "dfs" in strategies:
        print(f"\n{'='*60}")
        print(f"  Running DFS...")
        print(f"{'='*60}")
        dfs = ToTDFSAgent(
            agent=agent,
            threshold=0.8,
            max_loops=2,
            prune_threshold=0.5,
            number_of_agents=3,
        )
        results["DFS"] = dfs.search(task)

    if "bfs" in strategies:
        print(f"\n{'='*60}")
        print(f"  Running BFS...")
        print(f"{'='*60}")
        bfs = BFSWithTotAgent(
            agent=agent,
            max_loops=2,
            breadth_limit=2,
            number_of_agents=3,
        )
        results["BFS"] = bfs.search(task)

    if "sa" in strategies:
        print(f"\n{'='*60}")
        print(f"  Running SA...")
        print(f"{'='*60}")
        sa = ToTSAStrategy(
            agent=agent,
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.01,
            parallel_candidates=3,
            reheat_threshold=5,
            early_stop_patience=10,
            max_iterations=30,
            use_auto_evaluator=(evaluator == "auto"),
        )
        results["SA"] = sa.search(task)

    return results


def print_comparison_table(results: Dict[str, SearchResult]):
    """Print a markdown comparison table."""
    print(f"\n\n{'─'*60}")
    print("  BENCHMARK COMPARISON")
    print(f"{'─'*60}")
    print(
        f"  {'Strategy':<10} {'Score':<8} {'Time(s)':<10} "
        f"{'API Calls':<12} {'Best Answer':<30}"
    )
    print(f"  {'─'*8:<10} {'─'*6:<8} {'─'*7:<10} {'─'*9:<12} {'─'*29:<30}")
    for name, r in results.items():
        answer = r.final_answer[:28] + ".." if len(r.final_answer) > 30 else r.final_answer
        print(
            f"  {name:<10} {r.best_score:<8.3f} {r.execution_time:<10.2f} "
            f"{r.total_api_calls:<12} {answer:<30}"
        )
    print(f"{'─'*60}")

    # Print SA-specific details if available
    if "SA" in results:
        sa = results["SA"]
        if sa.temperature_curve:
            temp_start = sa.temperature_curve[0]
            temp_end = sa.temperature_curve[-1]
            accept_rate = (
                sum(sa.acceptance_log) / len(sa.acceptance_log)
                if sa.acceptance_log
                else 0
            )
            print(f"\n  SA Details:")
            print(f"    Iterations: {len(sa.temperature_curve)}")
            print(f"    Temperature: {temp_start:.2f} -> {temp_end:.4f}")
            print(f"    Acceptance rate: {accept_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ToT search strategies on a task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=(
            "Your task: use 4 numbers and basic arithmetic (+-*/) "
            "to obtain 24 in 1 equation, return only the math."
        ),
        help="The task to run all strategies on.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="dfs,bfs,sa",
        help="Comma-separated list of strategies: dfs, bfs, sa",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="self",
        choices=["self", "auto"],
        help="Evaluator mode: 'self' (LLM self-eval) or 'auto' (AutoEvaluator)",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    print(f"Task: {args.task}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Evaluator: {args.evaluator}")

    results = run_benchmark(args.task, strategies, args.evaluator)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
