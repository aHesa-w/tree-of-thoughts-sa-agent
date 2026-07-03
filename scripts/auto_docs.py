"""Auto-generate documentation skeleton files from the actual module classes.

Generates doc markdown templates by inspecting the current classes in
tree_of_thoughts. No external model API calls.
"""

import inspect
import os
import threading
from typing import List, Type

from tree_of_thoughts import TotAgent, ToTDFSAgent, BFSWithTotAgent, ToTSAStrategy


DOC_SKELETON = """# {class_name}

{doc}

## Constructor

```python
{signature}
```

## Methods

| Method | Description |
|--------|-------------|
{method_rows}

## Usage Example

```python
from tree_of_thoughts import {class_name}

# TODO: add example
```

"""


def generate_doc_markdown(cls: Type) -> str:
    doc = inspect.getdoc(cls) or "(no documentation)"
    sig_parts = []
    try:
        sig = inspect.signature(cls.__init__)
        sig_parts.append(f"def __init__{sig}")
    except (ValueError, TypeError):
        sig_parts.append("def __init__(self, ...)")

    signature = "\n    ".join(sig_parts)

    method_rows = ""
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        method_doc = inspect.getdoc(method) or ""
        first_line = method_doc.split("\n")[0] if method_doc else ""
        method_rows += f"| `{name}()` | {first_line} |\n"

    return DOC_SKELETON.format(
        class_name=cls.__name__,
        doc=doc,
        signature=signature,
        method_rows=method_rows,
    )


def create_doc_file(cls: Type, output_dir: str = "docs/tot"):
    markdown = generate_doc_markdown(cls)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{cls.__name__.lower()}.md")
    with open(file_path, "w") as f:
        f.write(markdown)

    print(f"  [CREATED] {file_path}")


def main():
    classes = [
        TotAgent,
        ToTDFSAgent,
        BFSWithTotAgent,
        ToTSAStrategy,
    ]

    print(f"Generating doc skeletons for {len(classes)} classes...")
    threads = []
    for cls in classes:
        thread = threading.Thread(target=create_doc_file, args=(cls,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nDone. Doc skeletons generated in 'docs/tot/'.")


if __name__ == "__main__":
    main()
