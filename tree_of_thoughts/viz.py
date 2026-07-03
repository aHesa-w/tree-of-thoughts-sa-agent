"""Visualization utilities for Tree of Thoughts search strategies.

Provides:
    - ASCII temperature curve for SA
    - Acceptance rate charts
    - Search tree text-based visualization
"""

from typing import List, Optional


def ascii_temperature_curve(
    temperatures: List[float],
    width: int = 40,
    height: int = 8,
) -> str:
    """Render an ASCII line chart of temperature over iterations.

    Args:
        temperatures: List of temperature values per iteration.
        width: Character width of the chart.
        height: Character height of the chart.

    Returns:
        Multi-line string with the chart.
    """
    if not temperatures:
        return "[No temperature data]"

    t_min = min(temperatures)
    t_max = max(temperatures)
    t_range = t_max - t_min if t_max != t_min else 1.0

    lines = []
    # Y-axis labels and plot area
    for row in range(height):
        y_ratio = 1.0 - (row / (height - 1))
        y_val = t_min + y_ratio * t_range
        label = f"{y_val:6.3f} |"

        line_chars = []
        for col in range(width):
            idx = int(col / width * (len(temperatures) - 1))
            norm = (temperatures[idx] - t_min) / t_range
            y_pos = 1.0 - norm
            row_pos = row / (height - 1)
            if abs(y_pos - row_pos) < 1.0 / height:
                line_chars.append("*")
            elif col == 0:
                line_chars.append("|")
            else:
                line_chars.append(" ")
        lines.append(label + "".join(line_chars))

    # X-axis
    x_axis = " " * 7 + "+" + "-" * (width - 1)
    lines.append(x_axis)
    lines.append(
        f" {'0':>7} {'iterations':>{width // 2}} {len(temperatures) - 1:>{width // 3}}"
    )

    return "\n".join(lines)


def ascii_acceptance_bar(acceptance_log: List[bool]) -> str:
    """Render an acceptance/rejection bar chart.

    Args:
        acceptance_log: List of booleans (True=accept, False=reject).

    Returns:
        Multi-line string.
    """
    if not acceptance_log:
        return "[No acceptance data]"

    total = len(acceptance_log)
    accepts = sum(acceptance_log)
    rejects = total - accepts
    width = 30

    accept_ratio = accepts / total if total else 0
    reject_ratio = rejects / total if total else 0

    accept_bar = int(accept_ratio * width)
    reject_bar = width - accept_bar

    lines = [
        "  Acceptance / Rejection",
        f"  Accept: {accepts}/{total} ({accept_ratio:.1%})",
        f"  Reject: {rejects}/{total} ({reject_ratio:.1%})",
        "  " + "█" * accept_bar + "░" * reject_bar,
    ]
    return "\n".join(lines)


def search_tree_text(
    thoughts: List[dict],
    max_depth: Optional[int] = None,
    max_items: int = 20,
) -> str:
    """Render a text-based search tree from a flat thought list.

    Since ToT thoughts don't always carry parent_id, we render them
    as a sorted list showing evaluation scores.

    Args:
        thoughts: List of dicts with at least 'thought' and 'evaluation'.
        max_depth: Not used (flat list rendering), but kept for interface compat.
        max_items: Max number of items to show.

    Returns:
        Multi-line string.
    """
    if not thoughts:
        return "[No thoughts to display]"

    sorted_thoughts = sorted(
        thoughts, key=lambda x: x.get("evaluation", 0), reverse=True
    )

    lines = ["  Thoughts (sorted by score, top first):", ""]
    for i, t in enumerate(sorted_thoughts[:max_items]):
        thought = t.get("thought", "")
        eval_score = t.get("evaluation", 0)
        thought_short = thought[:60] + ".." if len(thought) > 62 else thought
        lines.append(f"  {i+1:>2}. [{eval_score:.2f}] {thought_short}")

    if len(sorted_thoughts) > max_items:
        lines.append(f"  ... ({len(sorted_thoughts) - max_items} more)")

    return "\n".join(lines)


def format_search_result(result) -> str:
    """Format a SearchResult into a human-readable report."""
    lines = [
        f"  Strategy: {result.strategy}",
        f"  Best score: {result.best_score:.3f}",
        f"  Time: {result.execution_time:.2f}s",
        f"  API calls: {result.total_api_calls}",
        f"  Answer: {result.final_answer}",
        "",
    ]

    if result.temperature_curve:
        lines.append("  Temperature curve:")
        lines.append(ascii_temperature_curve(result.temperature_curve))
        lines.append("")

    if result.acceptance_log:
        lines.append(ascii_acceptance_bar(result.acceptance_log))
        lines.append("")

    if result.thought_graph:
        lines.append(search_tree_text(result.thought_graph, max_items=10))
        lines.append("")

    return "\n".join(lines)
