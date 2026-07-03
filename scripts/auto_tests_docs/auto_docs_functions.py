"""Auto-generate documentation skeletons for functions in tree_of_thoughts.

No external model API calls. Uses inspect to generate skeleton docs.
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


DOC_SKELETON = """# {func_name}

{doc}

## Signature

```python
{signature}
```

## Usage

```python
# TODO: add usage example
```
"""


def generate_doc_markdown(func) -> str:
    doc = inspect.getdoc(func) or "(no documentation)"
    try:
        sig = inspect.signature(func)
        sig_str = f"def {func.__name__}{sig}"
    except (ValueError, TypeError):
        sig_str = f"def {func.__name__}(...)"

    return DOC_SKELETON.format(
        func_name=func.__name__,
        doc=doc,
        signature=sig_str,
    )


def create_doc_file(func, output_dir: str = "docs/tot/functions"):
    markdown = generate_doc_markdown(func)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{func.__name__.lower()}.md")
    with open(file_path, "w") as f:
        f.write(markdown)

    print(f"  [CREATED] {file_path}")


def get_functions() -> list:
    functions = [string_to_dict]

    # Static methods from evaluators
    for cls in [AutoEvaluator, MathEvaluator, CodeEvaluator]:
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                functions.append(method)

    return functions


def main():
    functions = get_functions()
    print(f"Generating function docs for {len(functions)} functions...")

    threads = []
    for func in functions:
        thread = threading.Thread(target=create_doc_file, args=(func,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nDone. Function docs generated in 'docs/tot/functions/'.")


if __name__ == "__main__":
    main()
