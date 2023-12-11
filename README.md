[![pypi](https://img.shields.io/pypi/v/pyoak.svg)](https://pypi.org/project/pyoak/)
[![pypi](https://img.shields.io/pypi/l/pyoak.svg)](https://pypi.org/project/pyoak/)
[![python](https://img.shields.io/pypi/pyversions/pyoak.svg)](https://pypi.org/project/pyoak/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Status](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml/badge.svg)](https://github.com/mishamsk/pyoak/actions/workflows/dev.yml)

Commerial-grade, well tested, documented and typed Python library for modeling, building, traversing, transforming, transferring and even pattern matching abstract syntax trees (AST) for arbtirary languages.

## Features<!-- omit from toc -->

* 🌳 Easy to use dataclasses based, strictly typed, pseudo-immutable AST modeling
* 📝 "Magic", auto-maintained node registry, allowing node cross-referncing and retrievel
* 📚 Source origin tracking for each node
* 📺 Zero-setup pretty printing using [Rich](https://github.com/willmcgugan/rich)
* 💾 json, msgpack, yaml or plain dict (de)serialization
* 🏃‍♀️ AST traversal: depth-first or breadth-first, top down or bottom up, with filtering and pruning
* 🎯 Xpath-like AST search (top to the node)
* 👯‍♂️ Node pattern matching with ability to capture specific sub-trees or attributes
* ... and more!

## Feature Roadmap<!-- omit from toc -->
* Pattern matcher rewrite to a bespoke engine to avoid limitations of the current implementation
* Context-aware pattern matching
* Prettyfiyng pattern matching language to make it more friendly
* Strict-mode with runtime field value type checking
* Frozen version of ASTNode
* ~~rustify some parts for performance~~ well, that's too far-fetched

## Table of Contents <!-- omit from toc -->
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Documentation](#documentation)
    - [Attached \& Detached Nodes, Registry](#attached--detached-nodes-registry)
    - [Defining a Model](#defining-a-model)
        - [Inheritance](#inheritance)
    - [Creating a Node](#creating-a-node)
    - [~~Mutating~~ Applying Changes to a Node](#mutating-applying-changes-to-a-node)
    - [Traversal](#traversal)
    - [Other Helpers](#other-helpers)
    - [Xpath \& Pattern Matching](#xpath--pattern-matching)
        - [Xpath](#xpath)
        - [Pattern Matching](#pattern-matching)
    - [Visitors \& Transformers](#visitors--transformers)
        - [ASTVisitor](#astvisitor)
        - [ASTTransformVisitor](#asttransformvisitor)
        - [ASTTransformer](#asttransformer)
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

The docs are not a book on code parsing, semantic analysis or AST transformations. It is assumed that you already know what ASTs are and are looking for a library to make it a bit easier and more convenient to work with them in Python.

pyoak code is strictly typed and heavily documented. Most public (and private) methods and system fields have extensive docstrings, thus it should be easy to use the library without reading the docs.

It is strongly encouraged to use mypy/pyright (or your type checker of choice) as it will make working with ASTs a breeze, as well as help you to avoid many common mistakes.

### Attached & Detached Nodes, Registry

The first and potentially the most important thing to understand is that whenever you create a node instance by default it becomes "attached" to a global registry and gets a unique id.

Attached nodes not only are guaranteed to have a unique id, they ensure uniqueness of their children, get a reference to their parent and maintain this state even when [changes are made](#mutating-applying-changes-to-a-node) to the tree. And of course, you can always get a node by its id, thus allowing you to have external cross-references, such as semantic graph.

To ensure correctness, attached nodes are assumed to be pseudo-immutable and should only be changed via [APIs](#mutating-applying-changes-to-a-node) or [transforms](#visitors--transformers). Since there is no true way to enforce this in Python, it is up to you to follow this rule, but there are a number of checks that will raise exceptions - see docs for particular APIs for more details.

### Defining a Model

pyoak AST nodes are regular Python [dataclasses](https://docs.python.org/3/library/dataclasses.html), with ~~a lot of~~ some additional magic.

To create a node you just need to create a dataclass that inherits from `pyoak.node.ASTNode`.

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

Since node instances are assumed to be pseudo-immutable, it is not expected that you'll need anything beyond a tuple of children. Even lists are supported for legacy reasons and may/will be removed in the future.

> ⚠️ Frozen dataclasses are not supported yet (see [the roadmap](#feature-roadmap)), but you should treat the library as generally geared towards treating nodes as if they are

> 💡 You may use dataclasses.field, e.g. to define defaults, provide serialization hints via metadata, or mark some fields as non-compare. pyoak doesn't restrict you in any way

#### Inheritance

It may be convenient to use inheritance. Especially when combined with the fact that dataclasses can override fields of their parent classes. See the example below:

```python
@dataclass
class Expr(ASTNode):
    """Base for all expressions"""
    pass

@dataclass
class Literal(Expr):
    """Base for all literals"""
    value: Any

@dataclass
class NumberLiteral(Literal):
    value: int

@dataclass
class NoneLiteral(Literal):
    value: None = field(default=None, init=False)

@dataclass
class BinExpr(Expr):
    """Base for all binary expressions"""
    left: Expr
    right: Expr
```

Not only have you defined the model for binary expressions, but also:
* narrowed down the value type for `NumberLiteral` to `int`
* saved yourself time by not having to define `value` field for `NoneLiteral` when you [instantiate](#creating-a-node)

all while maintaining full class compatibility.

> 💡 You may use multiple inheritance as well, say, to define "tag" like classes, that do not carry any data, but are used to mark nodes. E.g. `Unsupported` for structures that wrap unparsed code

### Creating a Node

In its simplest form you just instantiate a class. The only mandatory keyword field that ASTNode adds is `origin`:

```python
int_val = NumberLiteral(value=1, origin=NoOrigin())
```

This will create an attached node with automatically generated id. You may also specify the id explicitly:

```python
int_val = NumberLiteral(value=1, origin=NoOrigin(), id="some_id")
```

in this scenario, pyoak will check if the id is unique. If it is not, it will create a new unique one and will set `id_collision_with` to the id that you've provided.

This is helpful if you have some natural id from the parsing source and you want to catch duplicates.

You can also pass `ensure_unique_id=True` to raise an exception instead.

```python
int_val1 = NumberLiteral(value=1, origin=NoOrigin(), id="some_id")
int_val2 = NumberLiteral(value=1, origin=NoOrigin(), id="some_id", ensure_unique_id=True) # will raise ASTNodeIDCollisionError
```

There are other optional init-only arguments to create a detached node, to mark id collision as a duplicate rather than a collision. There are also more exceptions that may be raised, e.g. if you try to create a node with children that are already children of another node. Refer to the docstrings for more details.

### ~~Mutating~~ Applying Changes to a Node

If you need to change values of a node, there is a `replace` method. It will create and return a new node with changes applied to it.

This works similarly to dataclasses.replace, but with some differences:
* It will automatically update the content_id of the node and all of its parents if needed.
* It is not allowed to change the following attributes: id, content_id, original_id, id_collision_with, fields with init=False.
* In addition to creating a new instance, it will also replace the node in the AST registry and within its parent if it has one.

Currently, `replace` doesn't validate the types of the changes you pass in to be compatible with the types of the fields you are trying to change, but runtime type checks are [planned for the future](#feature-roadmap).

There is also a second method: `replace_with`. It allows you to replace the entire node with a new one or None (remove the node).

Unlike `replace`, it does a runtime check against the parent if a node you are changing has one. Thus ensuring that the change is type safe.

### Traversal

Each node, even a detached one, knows about its children:

```python
for c in node.get_child_nodes():
    print(c)

# or

list_of_children = node.children
```

and attached nodes also know about their parents:

```python
parent = node.parent
```

But that's not all. There are multiple helpers to traverse down the tree:

```python
for num_lits in node.dfs(skip_self=False, filter=lambda n: isinstance(n, NumberLiteral)):
    print(num_lits.value)

# or using a shorthand

for num_lits in node.gather(NumberLiteral):
    print(num_lits.value)

# breadth-first search also available
list(node.bfs(filter=important_nodes, prune=stop_at))
```

check the docstring for full parameter documentation.

There are also helpers & shorthands for traversing up the tree:

```python
for ancestor in node.ancestors():
    print(ancestor)

parent_stmt = expr_node.get_first_ancestor_of_type(Stmt)
```

### Other Helpers

Besides the traversal methods mentioned above, there are also some other helpers:

* `duplicate` - create a deep copy of the given subtree, recursively duplicating all of its children
* `to_properties_dict` - that gives a mapping of field names to values only for the properties of the node, without the children
* `is_ancestor` - check if a node is an ancestor of another node
* `get_depth` - get the depth of the node in the tree, with an optional `relative_to` parameter to get the depth relative to another node

and others, that are less commonly used.

### Xpath & Pattern Matching

pyoak.match package provides two ways to match nodes in the tree: "xpath" and pattern matching.

Xpath can be used to match nodes by their path from the root of the tree using a syntax similar to xpath in xml.

Pattern matching is concerned with matching nodes by their content, including their children.

#### Xpath

To match, you must create an ASTXpath object and call its `match` method:

```python
from pyoak.match.xpath import ASTXpath

xpath = ASTXpath("//IntLiteral")
assert IntLiteral(value=1).match(xpath)
```

**Syntax**

Each path is in the format `@parent_field_name[index]type_name` and is separated by a slash.

- The slash is optional at the start of the XPath. If omitted, it is the same as using `//the rest of the path`.
- `//` is a wildcard (anywhere) that matches any path.
- `@parent_field_name` and `index` are optional.
- `type_name` is optional, except for the last path in the XPath.

Types are instance comparisons, so any subclass matches a type.

**More examples**

- `CallExpr/@arguments[0]Id` matches the first argument of a call expression at any level in the tree, which is of type `Id`.
- `CallExpr/@arguments[*]Id` matches any argument of type `Id`.
- `@arguments[*]CallExpr` matches any CallExpr as long as it is an argument (stored in a field with the said name) of a parent expression.
- `CallExpr` matches any call expression at any level in the tree.

#### Pattern Matching

Pattern matching is done using a "matcher" object. You first create it with a list of patterns you'd like to match against and then execute `match` against a node. The first pattern that matches will be returned.

```python
matcher = PatternMatcher(
    [("rule1", '(Literal #[value="re.*ex"])'), ("rule2", '(IntLiteral #[value="1"])')]
)
match = matcher.match(node)
```

the result will be either `None`, meaning no pattern matched, or a tuple of the name of the pattern that matched and the match dict.

The match dict will contain a mapping from capture keys to values. Capture keys are the names you can embed within the pattern to store something within the matched node. E.g.:

```python
matcher = PatternMatcher([("rule", "(ParentType @[child_tuple_field =[(Literal) !! -> first_child, * -> remaining_children]])")])
```

this pattern matches any subclass of `ParentType` that has a child field `child_tuple_field` which is a sequence, first child of type Literal and then zero or more children of any type and shape.

The capture key `first_child` will be mapped to the first child of the node that matched the pattern and `remaining_children` will be mapped to a tuple of the remaining children.

> :warning: pattern matching is known to have some quirks, so use with caution. Upgrade of the internal engine is the main item on the [feature roadmap](#feature-roadmap).

### Visitors & Transformers

pyoak provides 3 classes for visiting and transforming the tree.

#### ASTVisitor
A base class for all visitors, and the one to use if you want to visit the tree without changing it.

It provides a `visit` method that will call the appropriate `visit_<node_type>` method on the visitor object or `generic_visit` if one is not available.

Important notes:
* Visitor doesn't visit children by default. It is up to you to call `visit` on the children you want to visit.
* By default, visitor methods are matched by the type of the node, if not found the next type in the MRO is checked, and so on. Going back to our examples above `visit_Expr` will be triggered for any subclass of Expr, unless a more specific visitor is defined.
  * If you'd like visitor methods to be matched only by strict type equality, change the class variable `strict` to True in your visitor class.
* You can pass `validate=True` when inheriting from `ASTVisitor` like so `class MyVisitor(ASTVisitor, validate=True)` to validate that all visitor methods match the call signature. I.e. if you have a visitor method `visit_Expr(self, node: IntExpr)` and `validate=True` is passed, an exception will be raised.

#### ASTTransformVisitor

A base class for all visitors that need to transform the tree. It inherits from `ASTVisitor`, thus all of the notes above also apply to it.

> ‼️ This is the recommended way of transforming the tree when you need to do a lot of changes. For simple changes, see `ASTTransformer` below or use `replace_with` directly.

Instead of calling `visit`, you'd call the `transform` method (although `visit` will do the same thing).

Unlike regular visitors, this one always works on a detached copy of the tree. This is necessary to ensure that the original tree is only modified and replaced on successful transformation.

The major implication of this is that you can't traverse `up` the tree inside visitor methods, as the parent pointers are not set on the copy.

#### ASTTransformer

Strictly speaking, this is not a visitor, but just a thin wrapper that traverses the tree bottom up, depth-first (using the `ASTNode.dfs` method) and applying `ASTNode.replace_with` when necessary.

Instead of passing functions to filters and writing replacement logic yourself, it allows you to inherit from `ASTTransformer`, redefine the `transform` method (and also optional `filter` and `prune` methods), and then run everything via the `execute` method.

Unlike ASTTransformVisitor, this class works on the original tree, so you can traverse `up` the tree inside the `transform` method.

```python
class SomeTransformer(ASTTransformer):
    def filter(self, node: ASTNode) -> bool:
        return isinstance(node, TypeOfInterest)

    def transform(self, node: ASTNode) -> ASTNode | None:
        # In reality, based on the filter nodes will be of TypeOfInterest, but we can't specify that in the signature
        if smth:
            return None
        else:
            return node.replace(attr=new_value)
```

### Serialization & Deserialization

ASTNode is itself based on `pyoak.serialize.DataClassSerializeMixin`, which itself is an extension of [mashumaro's](https://github.com/Fatal1ty/mashumaro) `DataClassDictMixin` with Json, Yaml & MsgPack (de)serialization support.

Most of the mashumaro capabilities are available verbatim, including `omit` metadata. There are a couple of differences though:

* You should use `as_dict`/`as_obj` instead of mashumari's `to_dict`/`from_dict`. The former correctly handles the type tag and all additional features below
* `as_dict`/`as_obj`, `to/from_json`, `to/from_msgpack`, and `to_from_yaml` methods provide extra argument `serialization_options`. There are a couple of built-in ones to serialize the list of child field names, to sort keys, to omit the type tag, and to serialize origin source values in an optimized way. You can also use this to further customize the serialization/deserialization process.
    * For more information about built-in options, check `pyoak.serialize` and `pyoak.origin` modules docstrings

#### Debugging & Reporting Deserialization Errors for Deeply Nested Trees

Trees can get pretty deep (we've seen a real-world usage of this library with millions of nodes, hundreds of levels deep).

By default, mashumaro will report an error at the top of the tree with the full source as context. This is not only unhelpful but also impossible to work with when your source is 100s of MBs.

For this reason, pyoak provides a helper `unwrap_invalid_field_exception` in `pyoak.serialize`. It will unwrap the exception and provide a tuple with a path to the invalid field and the inner exception that caused the error.

## Credits

* The (de)serialization code is based on the awesome [mashumaro](https://github.com/Fatal1ty/mashumaro)
* Pattern matching definitions and engine runs on [lark-parser](https://github.com/lark-parser/lark)

## Links

* GitHub: <https://github.com/mishamsk/pyoak>
* PyPI: <https://pypi.org/project/pyoak/>
* Free software: MIT
