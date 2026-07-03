"""
Usage: python examples/sa.py

No external LLM required. The SA strategy generates candidates via
TotAgent.run() which parses and evaluates thought strings.
"""

from tree_of_thoughts import ToTSAStrategy
from tree_of_thoughts.agent import TotAgent
from tree_of_thoughts.viz import format_search_result


# Create the agent (no internal LLM — external model drives generation)
agent = TotAgent()

# SA strategy
sa_strategy = ToTSAStrategy(
    agent=agent,
    initial_temperature=1.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    parallel_candidates=3,
    reheat_threshold=5,
    early_stop_patience=10,
    max_iterations=30,
)

initial_state = """
Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math.
"""

result = sa_strategy.search(initial_state)

print("\n" + "=" * 60)
print("  SA Search Result")
print("=" * 60)
print(format_search_result(result))
