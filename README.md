
![Tree of Thoughts Banner](images/treeofthoughts.png)

[Paper link](https://arxiv.org/pdf/2305.10601.pdf)
[Author's implementation](https://github.com/princeton-nlp/tree-of-thought-llm)

## Introduction

Tree of Thoughts (ToT) is a powerful and flexible algorithm that significantly advances model reasoning. This project implements three search strategies — **DFS**, **BFS**, and **SA (Simulated Annealing)** — under a unified `SearchStrategy` interface.

Key features:
- **Model-agnostic**: No embedded LLM dependency; TotAgent is a lightweight evaluator wrapper that parses and scores thought strings from any external model.
- **Unified interface**: All strategies share `SearchStrategy.search(initial_state) -> SearchResult`.
- **AutoEvaluator**: Built-in task-type-aware scoring for math (expression validation) and code (AST parse) — no LLM needed for evaluation.
- **No external API required**: Run with mock data or real LLM output in JSON format.

---

## Install

```bash
$ pip3 install -U tree-of-thoughts
```

### From source

```bash
$ git clone git@github.com:aHesa-w/tree-of-thoughts-sa-agent.git
$ cd tree-of-thoughts
$ pip3 install -e .
```

No `.env` file required.

---

## Quick Start

### SA strategy with mock data (no LLM needed)

```bash
$ python examples/sa.py
```

### SA strategy with custom evaluator

```python
from tree_of_thoughts import TotAgent, ToTSAStrategy

agent = TotAgent(
    evaluator=lambda task, thought: 1.0 if "correct" in thought else 0.0,
)
sa = ToTSAStrategy(agent=agent, max_iterations=5)
result = sa.search("Your task here")
print(result.final_answer, result.best_score)
```

### SA with AutoEvaluator (math/code scoring)

```python
from tree_of_thoughts import TotAgent, ToTSAStrategy

agent = TotAgent(auto_detect_evaluator=True)
sa = ToTSAStrategy(agent=agent, parallel_candidates=3)
result = sa.search("calculate 2+2")
```

### Full self-contained mock demo

```bash
$ python demo_sa_mock.py
```

This demo runs three parameter configurations (default / fast cool / explore mode) and prints a comparison report.

---

## Search Strategies

All strategies inherit from `tree_of_thoughts.base.SearchStrategy` and return `SearchResult`.

| Strategy | Class | File | Description |
|----------|-------|------|-------------|
| DFS | `ToTDFSAgent` | `tree_of_thoughts/dfs.py` | Depth-First Search with pruning |
| BFS | `BFSWithTotAgent` | `tree_of_thoughts/bfs.py` | Breadth-First Search with breadth limit |
| SA | `ToTSAStrategy` | `tree_of_thoughts/sa.py` | Simulated Annealing with probabilistic acceptance |

### SA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_temperature` | 1.0 | Starting temperature; higher = more exploration |
| `cooling_rate` | 0.95 | Temperature multiplier per iteration |
| `parallel_candidates` | 3 | Candidates generated per iteration via ThreadPool |
| `reheat_threshold` | 5 | Consecutive no-improvement before reheating |
| `reheat_factor` | 0.5 | Reheat magnitude |
| `early_stop_patience` | 10 | No-improvement + low temp to force stop |
| `max_iterations` | 100 | Hard iteration limit |
| `use_auto_evaluator` | False | Use AutoEvaluator for scoring |

---

## Evaluators

| Evaluator | Trigger | Scoring Logic |
|-----------|---------|---------------|
| `MathEvaluator` | Task contains math keywords (`+`, `-`, `=`, `solve`, etc.) | Safely `eval()` arithmetic expressions, check equality |
| `CodeEvaluator` | Task contains code keywords (`def`, `function`, `import`, etc.) | `ast.parse()` for valid Python syntax |
| `AutoEvaluator` | Auto-detected from task string | Routes to Math or Code evaluator; returns -1 for fallback |

---

## Testing

```bash
$ python -m pytest tests/ -v
```

47 tests across:
- `test_agent.py` — string_to_dict, TotAgent.run() evaluator routing, SearchResult
- `test_evaluator.py` — AutoEvaluator detection, Math/Code scoring
- `test_sa.py` — SA strategy with mocked LLM
- `test_viz.py` — ASCII visualization utilities

---

## Benchmark

Compare all strategies on a given task:

```bash
$ python scripts/benchmark.py --task "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"
```

---

## Todo

- [x] Unify DFS/BFS/SA under SearchStrategy interface
- [x] Decouple TotAgent from OpenAI / swarms dependencies
- [x] Add AutoEvaluator with math and code scoring
- [x] Add mock demo for SA strategy
- [ ] Add DFS/BFS unit tests
- [ ] Implement ThoughtGenerator protocol for clean external model integration
- [ ] Visual tree rendering from SearchResult

---

# Acknowledgements

Thanks to Shunyu Yao (Princeton University), Dian Yu (Google DeepMind), Jeffrey Zhao (Google DeepMind), Izhak Shafran (Google DeepMind), Thomas L. Griffiths (Princeton University), Yuan Cao (Google DeepMind), Karthik Narasimhan (Princeton University) for the Tree of Thoughts paper.

# License

Apache 2.0
