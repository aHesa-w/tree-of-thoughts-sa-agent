"""
Usage: python examples/bfs.py

No external LLM required. The BFS strategy generates candidates via
TotAgent.run() which parses and evaluates thought strings.
"""
from tree_of_thoughts import TotAgent, BFSWithTotAgent


# Create the agent
tot_agent = TotAgent()

# BFS strategy with breadth limit
bfs_agent = BFSWithTotAgent(
    agent=tot_agent,
    max_loops=3,
    breadth_limit=2,
    number_of_agents=3,
)

initial_state = """
Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math.
"""

final_thought = bfs_agent.run(initial_state)
print(final_thought)
