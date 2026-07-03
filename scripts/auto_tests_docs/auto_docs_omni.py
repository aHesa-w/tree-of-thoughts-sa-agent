"""Auto-generate documentation skeletons from inspect.

No external model API calls. Uses inspect to gather info from module classes.
"""

import inspect
import os
import threading

from tree_of_thoughts import (
    TotAgent,
    ToTDFSAgent,
    BFSWithTotAgent,
    ToTSAStrategy,
)
from tree_of_thoughts.evaluator import AutoEvaluator, MathEvaluator, CodeEvaluator


DOC_SKELETON = """# {item_name}

{doc}

## Signature

```python
{signature}
```

## Public Methods

| Method | Description |
|--------|-------------|
{method_rows}
```
"""


def generate_doc(item, module: str = "tree_of_thoughts"):
    doc = inspect.getdoc(item) or "(no documentation)"
    is_class = inspect.isclass(item)
    item_name = item.__name__

    sig_str = ""
    if is_class:
        try:
            sig = inspect.signature(item.__init__)
            sig_str = f"class {item_name}{sig}"
        except (ValueError, TypeError):
            sig_str = f"class {item_name}"
    else:
        try:
            sig = inspect.signature(item)
            sig_str = f"def {item_name}{sig}"
        except (ValueError, TypeError):
            sig_str = f"def {item_name}(...)"

    method_rows = ""
    if is_class:
        for name, method in inspect.getmembers(item, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            method_doc = inspect.getdoc(method) or ""
            first_line = method_doc.split("\n")[0] if method_doc else ""
            method_rows += f"| `{name}()` | {first_line} |\n"

    return DOC_SKELETON.format(
        item_name=item_name,
        doc=doc,
        signature=sig_str,
        method_rows=method_rows,
    )


def process_item(item, docs_folder_path: str):
    markdown = generate_doc(item)
    os.makedirs(docs_folder_path, exist_ok=True)

    file_path = os.path.join(docs_folder_path, f"{item.__name__.lower()}.md")
    with open(file_path, "w") as f:
        f.write(markdown)

    print(f"  [CREATED] {file_path}")


def main(docs_folder_path: str = "docs/tot"):
    items = [
        TotAgent,
        ToTDFSAgent,
        BFSWithTotAgent,
        ToTSAStrategy,
        AutoEvaluator,
        MathEvaluator,
        CodeEvaluator,
    ]
    print(f"Generating docs for {len(items)} items in '{docs_folder_path}'...")

    threads = []
    for item in items:
        thread = threading.Thread(target=process_item, args=(item, docs_folder_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nDone. Documentation generated in '{docs_folder_path}'.")


if __name__ == "__main__":
    main()
