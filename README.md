[![pypi](https://img.shields.io/pypi/v/pyoak.svg)](https://pypi.org/project/pyoak/)
[![pypi](https://img.shields.io/pypi/l/pyoak.svg)](https://pypi.org/project/pyoak/)
[![python](https://img.shields.io/pypi/pyversions/pyoak.svg)](https://pypi.org/project/pyoak/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Status](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml/badge.svg)](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml)

Commerial-grade, well tested, documented and typed Python library for modeling, building, traversing, transforming, transferring and even pattern matching abstract syntax trees (AST) for arbtirary languages.

## Features <!-- omit from toc -->

* 🌳 Easy to use dataclasses based, strictly typed, pseudo-immutable AST modeling
* 📝 "Magic", auto-maintained node registry, allowing node cross-referncing and retrievel
* 📚 Source origin tracking for each node
* 📺 Zero-setup pretty printing using [Rich](https://github.com/willmcgugan/rich)
* 💾 json, msgpack, yaml or plain dict (de)serialization
* 🏃‍♀️ AST traversal: depth-first or breadth-first, top down or bottom up, with filtering and pruning
* 🎯 Xpath-like AST search (top to the node)
* 👯‍♂️ Node pattern matching with ability to capture specific sub-trees or attributes
* ... and more!

## Feature Roadmap <!-- omit from toc -->
* Pattern matcher rewrite to a bespoke engine to avoid limitations of the current implementation
* Context-aware pattern matching
* Prettyfiyng pattern matching language to make it more friendly
* Strict-mode with runtime field value type checking
* Frozen version of ASTNode
* Make orjson, pyyaml, msgpack & chardet optional dependencies
* ~~rustify some parts for performance~~ well, that's too far-fetched

## Table of Contents <!-- omit from toc -->
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Documentation](#documentation)
    - [Defining a model](#defining-a-model)
    - [Creating a node](#creating-a-node)
    - [Traversal](#traversal)
        - [Helpers](#helpers)
    - [Xpath \& Pattern Matching](#xpath--pattern-matching)
    - [Visitors \& Transformers](#visitors--transformers)
    - [Serialization \& Deserialization](#serialization--deserialization)
        - [Debugging \& Reporting Deserialization Errors for Deeply Nested Trees](#debugging--reporting-deserialization-errors-for-deeply-nested-trees)
- [Credits](#credits)
- [Links](#links)


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

The docs are not a book on code parsing, semantic analysis or AST transformations. It is assumed that you already know what ASTs are and looking for a library to make it a bit easier and more convenient to work with them in Python.

pyoak code is strictly typed and heavily documented. Most public (and private) methods and system fields have extensive docstrings, thus it should be easy to use the library without reading the docs.

It is strongly encouraged to use mypy/pyright (or your type checker of choice) as it will make working with ASTs a breeze, as well as help you to avoid many common mistakes.

### Defining a model

pyoak AST nodes are regular Python [dataclasses](https://docs.python.org/3/library/dataclasses.html), with ~~a lot of~~ some additional magic.

To create a node you just need create a dataclass that inherits from `pyoak.node.ASTNode`.

```python
from dataclasses import dataclass
from pyoak.node import ASTNode

@dataclass
class MyNode(ASTNode):
    attribute: int
    child: ASTNode | None
    more_children: tuple[ASTNode, ...]
```

Any field that is either a union of `ASTNode` subclasses (including optional `None`), or a tuple/list of `ASTNode` subclasses will be considered as a child field. Anything else, including `dict[str, ASTNode]` won't become a child!

Since node instances are assumed to me pseudo-immuateble (more on that below), it is not expected that you'll need anything beyond a tuple of children. Even list's are supported for legacy reasons and may/will be removed in the future.

### Creating a node

### Traversal

#### Helpers

### Xpath & Pattern Matching

### Visitors & Transformers

### Serialization & Deserialization

#### Debugging & Reporting Deserialization Errors for Deeply Nested Trees

## Credits

* The (de)serialization code is based on the awesome [mashumaro](https://github.com/Fatal1ty/mashumaro)
* Pattern matching definitions and engine runs on [lark-parser](https://github.com/lark-parser/lark)

## Links

* GitHub: <https://github.com/mishamsk/pyoak>
* PyPI: <https://pypi.org/project/pyoak/>
* Free software: MIT
