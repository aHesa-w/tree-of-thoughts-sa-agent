"""Example: using SA search strategy with Tree of Thoughts.

Make sure OPENAI_API_KEY is set in your environment or .env file.
"""
from dotenv import load_dotenv

from tree_of_thoughts import TotAgent, ToTSAStrategy
from tree_of_thoughts.viz import format_search_result

load_dotenv()

# Create the agent (set use_openai_caller=True if you have OPENAI_API_KEY)
tot_agent = TotAgent(use_openai_caller=False)

# SA strategy
sa_strategy = ToTSAStrategy(
    agent=tot_agent,
    initial_temperature=1.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    parallel_candidates=3,
    reheat_threshold=5,
    early_stop_patience=10,
    max_iterations=30,
    use_auto_evaluator=False,
)

initial_state = """
Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math.
"""

result = sa_strategy.search(initial_state)

print("\n" + "=" * 60)
print("  SA Search Result")
print("=" * 60)
print(format_search_result(result))
