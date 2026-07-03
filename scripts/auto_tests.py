"""Auto-generate pytest skeleton files from the actual module classes.

Generates test templates by inspecting the current classes in
tree_of_thoughts. No external model API calls.
"""

import inspect
import os
import threading
from typing import List, Type

from tree_of_thoughts import TotAgent, ToTDFSAgent, BFSWithTotAgent, ToTSAStrategy


TEST_SKELETON = '''"""
Tests for {class_name}.
"""

import pytest


class Test{class_name}:
    """Tests for {class_name}.{doc_short}"""

    def test_init(self):
        """Test initialization with default parameters."""
        # TODO: implement
        pass

    def test_run_returns_expected_type(self):
        """Test that run() returns the correct type."""
        # TODO: implement
        pass
'''


def generate_test_skeleton(cls: Type) -> str:
    doc = inspect.getdoc(cls) or ""
    doc_first_line = doc.split("\n")[0] if doc else ""
    return TEST_SKELETON.format(
        class_name=cls.__name__,
        doc_short=doc_first_line,
    )


def create_test_file(cls: Type, output_dir: str = "tests/tot"):
    skeleton = generate_test_skeleton(cls)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"test_{cls.__name__.lower()}.py")
    with open(file_path, "w") as f:
        f.write(skeleton)

    print(f"  [CREATED] {file_path}")


def main():
    classes = [
        TotAgent,
        ToTDFSAgent,
        BFSWithTotAgent,
        ToTSAStrategy,
    ]

    print(f"Generating test skeletons for {len(classes)} classes...")
    threads = []
    for cls in classes:
        thread = threading.Thread(target=create_test_file, args=(cls,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nDone. Test skeletons generated in 'tests/tot/'.")


if __name__ == "__main__":
    main()
