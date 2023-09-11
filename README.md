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
* Make orjson, pyyaml, msgpack & chardet optional dependencies
* ~~rustify some parts for performance~~ well, that's too far-fetched

## Table of Contents <!-- omit from toc -->
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Documentation](#documentation)
    - [Defining a Model](#defining-a-model)
        - [Inheritance](#inheritance)
    - [Creating a Node](#creating-a-node)
        - [Runtime Type Checks](#runtime-type-checks)
        - [ID's \& Registry](#ids--registry)
    - [~~Mutating~~ Applying Changes to a Node](#mutating-applying-changes-to-a-node)
    - [Traversal](#traversal)
        - [Going Down the Tree](#going-down-the-tree)
        - [Going Up the Tree](#going-up-the-tree)
    - [Other Helpers](#other-helpers)
    - [Xpath \& Pattern Matching](#xpath--pattern-matching)
        - [Xpath](#xpath)
        - [Pattern Matching](#pattern-matching)
    - [Visitors \& Transformers](#visitors--transformers)
        - [ASTVisitor](#astvisitor)
        - [ASTTransformVisitor](#asttransformvisitor)
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


@dataclass(frozen=True)
class Name(ASTNode):
    identifier: str


@dataclass(frozen=True)
class NumberLiteral(ASTNode):
    value: int


@dataclass(frozen=True)
class Stmt(ASTNode):
    pass


@dataclass(frozen=True)
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

### Defining a Model

pyoak AST nodes are regular Python [dataclasses](https://docs.python.org/3/library/dataclasses.html), with ~~a lot of~~ some additional magic.

To create a node you just need to create a frozen dataclass that inherits from `pyoak.node.ASTNode`.

```python
from dataclasses import dataclass
from pyoak.node import ASTNode

@dataclass(frozen=True)
class MyNode(ASTNode):
    attribute: int
    child: ASTNode | None
    more_children: tuple[ASTNode, ...]
```


All fields are classified into either child fields or properties. Any subclass of ASTNode in field type marks it as a child field.

Since nodes are assumed to be immutable, pyoak will check type annotations on the best effort basis, to prevent usage of mutable types by raising an exception at class defintion. The following rules are enforced:
* Child fields may only be a union of `ASTNode` subclasses (including optional `None`), or a tuple of `ASTNode` subclasses. Anything else will raise an exception.
* For properties, nutable collections are not allowed, but arbitrary types are. It is up to you to ensure that the type is immutable.

> 💡 You may use dataclasses.field, e.g. to define defaults, provide serialization hints via metadata, or mark some fields as non-compare. pyoak doesn't restrict you in any way

#### Inheritance

It may be convenient to use inheritance. Especially when combined with the fact that dataclasses can override fields of their parent classes. See the example below:

```python
@dataclass(frozen=True)
class Expr(ASTNode):
    """Base for all expressions"""
    pass

@dataclass(frozen=True)
class Literal(Expr):
    """Base for all literals"""
    value: Any

@dataclass(frozen=True)
class NumberLiteral(Literal):
    value: int

@dataclass(frozen=True)
class NoneLiteral(Literal):
    value: None = field(default=None, init=False)

@dataclass(frozen=True)
class BinExpr(Expr):
    """Base for all binary expressions"""
    left: Expr
    right: Expr
```

Not only have you defined the model for binary expressions, but also:
* narrowed down the value type for `NumberLiteral` to `int`
* saved yourself time by not having to define `value` field for `NoneLiteral` when you [instantiate](#creating-a-node)

all while maintaining full class compatibility.

> 💡 You may use multiple inheritance as well, say, to define "tag" like classes, that do not carry any data, but are used to mark nodes. E.g. `Unsupported` for structures that wrap unparsed code. However, this will prevent you from using slotted classes

### Creating a Node

In its simplest form you just instantiate a class:

```python
int_val = NumberLiteral(value=1)
```
This will create a node with automatically generated id.

The base ASTNode class defines only one optional field `origin` (which defaults to `NO_ORIGIN` sentinel)

```python
int_val = NumberLiteral(value=1, origin=CodeOrigin(...))
```

#### Runtime Type Checks

pyoak has an optional feature (disabled by default) to do runtime type checks on instance creation. To enable it use the following snippet:

```python
from pyoak import config

config.RUNTIME_TYPE_CHECK = True
```

This feature is convenient during development, to check for typical mistakes, like passing list instead of tuple. However, it is not recommended to use it in production, as it slows down instance creation by 20-40%.

#### ID's & Registry

Every time you create a node, it gets two ids: `id` and `content_id`. The former is a globally unique id for the node, the latter is a hash representing node's content (itself and all of it's children).

Each new node is automatically added to the registry, which is a mapping from `id` to the node. This allows you to retrieve any node by its id, even if you don't have a reference to it.

```python
int_val = NumberLiteral(value=1)
assert int_val is NumberLiteral.get(int_val.id)
# or
assert int_val is ASTNode.get_any(int_val.id)
```

However, registry is a weak mapping, so if you don't have any references to the node anywhere, it will be garbage collected and removed from the registry.

### ~~Mutating~~ Applying Changes to a Node

Nodes are immutable (to the extent possible in Python), thus if you need to change values of a node, use `dataclasses.replace` function.

### Traversal

#### Going Down the Tree

Each node, knows about its children:

```python
for c in node.get_child_nodes():
    print(c)

# or

list_of_children = node.children
```

There are multiple helpers to traverse down the tree:

```python
for num_lits in node.dfs(skip_self=False, filter=lambda n: isinstance(n, NumberLiteral)):
    print(num_lits.value)

# or using a shorthand

for num_lits in node.gather(NumberLiteral):
    print(num_lits.value)

# breadth-first search also available
list(node.bfs(filter=important_nodes, prune=stop_at))
```

Note, that these methods return an generator, so you can stop the traversal at any time. Also, worth mentioning that the node itself is not yielded.

#### Going Up the Tree

If you need to search "up" the tree, you need to use the `Tree` instance created from the root node:

```python
tree = Tree(root_node)
# or
tree =root_node.to_tree()
```

Now you can use multiple methods:
```python
parent = tree.get_parent(node) # to get the parent of the node

for ancestor in tree.get_ancestors(node):
    print(ancestor)

parent_stmt = tree.get_first_ancestor_of_type(expr_node, Stmt)
```

### Other Helpers

Besides the traversal methods mentioned above, there are also some other helpers:

* `ASTNode.duplicate` - create a deep copy of the given subtree, recursively duplicating all of its children
* `ASTNode.to_properties_dict` - that gives a mapping of field names to values only for the properties of the node, without the children
* `Tree.is_ancestor` - check if a node is an ancestor of another node
* `Tree.get_depth` - get the depth of the node in the tree, with an optional `relative_to` parameter to get the depth relative to another node

and others, that are less commonly used.

### Xpath & Pattern Matching

pyoak.match package provides two ways to match nodes in the tree: "xpath" and pattern matching.

Xpath can be used to match nodes by their path from the root of the tree using a syntax similar to xpath in xml.

Pattern matching is concerned with matching nodes by their content, including their children.

#### Xpath

If you want to find all sub-nodes matching a given xpath starting a given node, you can use the `find` and `findall` methods:

```python
root_node = parse("some code")
for node in root_node.findall("//IntLiteral"):
    print(node.value)
```

If you want to match a given node, i.e. check if it's in the path, you must create an ASTXpath object and call its `match` method:

```python
from pyoak.match.xpath import ASTXpath

root_node = parse("some code")
xpath = ASTXpath("//IntLiteral")

...
if xpath.match(root_node, some_node):
    print("Matched!")
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

Pattern matching is done using a "matcher" object. There are two of them:
* `MultiPatternMatcher` for matching against multiple patterns
* `NodeMatcher` for matching against a single pattern

To create a `NodeMatcher`:
```python
macher, msg = NodeMatcher.from_pattern("(RootClass @child_tuple=[(*) -> cap $cap *])")
assert macher is not None, msg
```

and then match:
```python
node = RootClass(child_tuple=(c1, c2, tail)

ok, match_dict = macher.match(node)
assert ok
assert match_dict["cap"] == f1
```

With `MultiPatternMatcher` you'll first create it with a list of patterns you'd like to match against and then execute `match` against a node.

```python
matcher = MultiPatternMatcher(
    [("rule1", '(Literal #value="re.*ex")'), ("rule2", '(IntLiteral #value="1")')]
)
match = matcher.match(node)
```

the result will be either `None`, meaning no pattern matched, or a tuple of the name of the pattern that matched and the match dict.

The match dict will contain a mapping from capture keys to values. Capture keys are the names you can embed within the pattern to store something within the matched node. E.g.:

```python
matcher = MultiPatternMatcher([("rule", "(ParentType @child_tuple_field =[(Literal) -> first_child, * -> remaining_children])")])
```

this pattern matches any subclass of `ParentType` that has a child field `child_tuple_field` which is a sequence, first child of type Literal and then zero or more children of any type and shape.

The capture key `first_child` will be mapped to the first child of the node that matched the pattern and `remaining_children` will be mapped to a tuple of the remaining children.

### Visitors & Transformers

pyoak provides 3 classes for visiting and transforming the tree.

#### ASTVisitor
A base class for all visitors, and the one to use if you want to visit the tree without changing it.

It provides a `visit` method that will call the appropriate `visit_<node_type>` method on the visitor object or `generic_visit` if one is not available.

Important notes:
* The class is generic in visit method return type. It is assumed that the type of all `visit_<node_type>` methods will match the return type. Unfirtunately, this cannot be type checked by mypy.
* It is abstract class. As a minimum you must override `generic_visit`
* Visitor doesn't visit children by default. It is up to you to call `visit` on the children you want to visit.
* By default, visitor methods are matched by the type of the node, if not found the next type in the MRO is checked, and so on. Going back to our examples above `visit_Expr` will be triggered for any subclass of Expr, unless a more specific visitor is defined.
  * If you'd like visitor methods to be matched only by strict type equality, change the class variable `strict` to True in your visitor class.
* You can pass `validate=True` when inheriting from `ASTVisitor` like so `class MyVisitor(ASTVisitor, validate=True)` to validate that all visitor methods match the call signature. I.e. if you have a visitor method `visit_Expr(self, node: IntExpr)` and `validate=True` is passed, an exception will be raised.

#### ASTTransformVisitor

A base class for all visitors that need to transform the tree. It inherits from `ASTVisitor`, thus all of the notes above also apply to it.

Unlike `ASTVisitor`:
* It is not generic. Visitor methods return type is pinned to `ASTNode | None`
* It provides a default implementation of `generic_visit` that will transform all children by default, if it detect any changes it will return a new node with the transformed children. Otherwise, it will return the original node.
* You can use a helper method `_transform_children` to transform children. It will return a dictionary suitable for passing to `dataclasses.replace` to create a new node with the transformed children.
* An alias to `visit` called `transform` is provided for convenience.

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
