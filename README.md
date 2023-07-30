[![pypi](https://img.shields.io/pypi/v/pyoak.svg)](https://pypi.org/project/pyoak/)
[![pypi](https://img.shields.io/pypi/l/pyoak.svg)](https://pypi.org/project/pyoak/)
[![python](https://img.shields.io/pypi/pyversions/pyoak.svg)](https://pypi.org/project/pyoak/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Status](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml/badge.svg)](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml)

Commerial-grade, well tested, documented and typed Python library for modeling, building, traversing, transforming, transferring and even pattern matching abstract syntax trees (AST) for arbtirary languages.

## Features

* 🌳 Easy to use dataclasses based, strictly typed, pseudo-immutable AST modeling
* 📝 "Magic", auto-maintained node registry, allowing node cross-referncing and retrievel
* 📚 Source origin tracking for each node
* 📺 Zero-setup pretty printing using [Rich](https://github.com/willmcgugan/rich)
* 💾 json, msgpack, yaml or plain dict (de)serialization
* 🏃‍♀️ AST traversal: depth-first or breadth-first, top down or bottom up, with filtering and pruning
* 🎯 Xpath-like AST search (top to the node)
* 👯‍♂️ Node pattern matching with ability to capture specific sub-trees or attributes
* ... and more!

## Installation

```bash
pip install pyoak
```

## Basic Usage

```python
from dataclasses import dataclass

from pyoak.node import ASTNode
from pyoak.origin import CodeOrigin, TextSource, get_code_range
from rich import print


@dataclass
class Name(ASTNode):
    identifier: str


@dataclass
class NumberLiteral(ASTNode):
    value: int


@dataclass
class Stmt(ASTNode):
    pass


@dataclass
class AssignStmt(Stmt):
    name: Name
    value: NumberLiteral


def parse_assignment(code: str) -> AssignStmt:
    # This should be a real parse logic
    s = TextSource(source_uri="important_place", source_type="text/plain", _raw=code)
    stmt_o = CodeOrigin(source=s, position=get_code_range(0, 1, 0, 5, 1, 5))
    x_o = CodeOrigin(source=s, position=get_code_range(0, 1, 0, 1, 1, 1))
    num_o = CodeOrigin(source=s, position=get_code_range(4, 1, 4, 5, 1, 5))
    return AssignStmt(
        name=Name(identifier="x", origin=x_o),
        value=NumberLiteral(value=1, origin=num_o),
        origin=stmt_o,
    )


node = parse_assignment("x = 1\nother_stuff")
print(node)  # Prints a pretty tree
print(
    f"Original source code: {node.origin.get_raw()}"
)  # prints `Original source code: x = 1`
assert node is AssignStmt.get(node.id)  # you can always get the node using only its id
dup_node = node.duplicate()  # returns a deep copy of the node
assert node is not dup_node  # they are different, including id, but...
dup_node.original_id = node.id  # ...they can be linked together
assert dup_node == node  # True
# which is the same as
assert (
    dup_node.content_id == node.content_id
)  # content id uniquely represents the subtree

# Now let's get an iterable of all Name, NumberLiteral nodes
# this will traverse all the way down the tree, not just the direct children
some_children = node.gather((Name, NumberLiteral))

for subtree in some_children:
    print(subtree)
    print(f"Original source code: {subtree.origin.get_raw()}")
```

and this is just the tip of the iceberg!

## Documentation

TBD

## Credits

* The (de)serialization code is based on the awesome [mashumaro](https://github.com/Fatal1ty/mashumaro)
* Pattern matching definitions and engine runs on [lark-parser](https://github.com/lark-parser/lark)

## Links

* GitHub: <https://github.com/mishamsk/pyoak>
* PyPI: <https://pypi.org/project/pyoak/>
* Free software: MIT
