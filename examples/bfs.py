"""
Usage: python examples/bfs.py

Make sure OPENAI_API_KEY is set in your environment or .env file.
"""
from tree_of_thoughts import TotAgent, BFSWithTotAgent
from dotenv import load_dotenv

load_dotenv()

# Create the agent
tot_agent = TotAgent(use_openai_caller=False)

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
