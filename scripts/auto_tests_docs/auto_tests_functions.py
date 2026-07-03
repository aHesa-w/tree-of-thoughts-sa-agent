"""Auto-generate test skeletons for functions in tree_of_thoughts.

No external model API calls. Uses inspect to generate skeleton test files.
"""

import inspect
import os
import sys
import threading

from tree_of_thoughts import (
    TotAgent,
    ToTDFSAgent,
    BFSWithTotAgent,
    ToTSAStrategy,
)
from tree_of_thoughts.evaluator import AutoEvaluator, MathEvaluator, CodeEvaluator
from tree_of_thoughts.base import string_to_dict


TEST_SKELETON = '''"""
Tests for {func_name}.
"""

import pytest


class Test{func_camel}:
    """Tests for {func_name}."""

    def test_basic(self):
        """TODO: implement basic test."""
        pass
'''


def to_camel_case(name: str) -> str:
    parts = name.replace("_", " ").split()
    return "".join(p.capitalize() for p in parts)


def generate_test_skeleton(func) -> str:
    return TEST_SKELETON.format(
        func_name=func.__name__,
        func_camel=to_camel_case(func.__name__),
    )


def create_test_file(func, output_dir: str = "tests/utils"):
    skeleton = generate_test_skeleton(func)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"test_{func.__name__.lower()}.py")
    with open(file_path, "w") as f:
        f.write(skeleton)

    print(f"  [CREATED] {file_path}")


def get_module_functions() -> list:
    functions = []

    # Functions from base module
    functions.append(string_to_dict)

    # Static methods from evaluator module
    for cls in [AutoEvaluator, MathEvaluator, CodeEvaluator]:
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                functions.append(method)

    return functions


def main():
    functions = get_module_functions()
    print(f"Generating test skeletons for {len(functions)} functions...")

    threads = []
    for func in functions:
        thread = threading.Thread(target=create_test_file, args=(func,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nDone. Test skeletons generated in 'tests/utils/'.")


if __name__ == "__main__":
    main()
