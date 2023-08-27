from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, NamedTuple, Type, cast

from lark import Lark, UnexpectedInput
from lark.visitors import Transformer

from ..match.error import ASTXpathDefinitionError
from ..match.helpers import check_and_get_ast_node_type
from ..node import ASTNode, NodeTraversalInfo
from ..tree import Tree
from ..typing import Field

xpath_grammar = """
xpath: element* self

element: "/" field_spec? index_spec? class_spec?

self: "/" field_spec? index_spec? class_spec

field_spec: "@" CNAME

index_spec: "[" DIGIT* "]"

class_spec: CNAME

%import common.WS
%import common.CNAME
%import common.DIGIT

%ignore WS
"""

logger = logging.getLogger(__name__)


class ASTXpathElement(NamedTuple):
    ast_class: type[ASTNode]
    parent_field: str | None
    parent_index: int | None
    anywhere: bool


class XPathTransformer(Transformer[str, list[ASTXpathElement]]):
    def xpath(
        self, args: list[tuple[str | None, int | None, Type[ASTNode] | None]]
    ) -> list[ASTXpathElement]:
        ret: list[ASTXpathElement] = []

        # The following logic parses the xpath in reverse order
        elements = reversed(args)
        for el in elements:
            parent_field, parent_index, ast_class = el

            while ast_class is None:
                next_el = next(elements, None)
                if next_el is None:
                    # We are at the very beginning of the xpath
                    # and it starts with // (or more)
                    # change last element to anywhere and return
                    ret[-1] = ASTXpathElement(
                        ret[-1].ast_class, ret[-1].parent_field, ret[-1].parent_index, True
                    )
                    return ret

                # Change last element to anywhere
                ret[-1] = ASTXpathElement(
                    ret[-1].ast_class, ret[-1].parent_field, ret[-1].parent_index, True
                )

                parent_field, parent_index, ast_class = next_el

            ret.append(
                ASTXpathElement(
                    ast_class=ast_class,
                    parent_field=parent_field,
                    parent_index=parent_index,
                    anywhere=False,
                )
            )

        return ret

    def element(
        self, args: list[Type[ASTNode] | str | int]
    ) -> tuple[str | None, int | None, Type[ASTNode] | None]:
        if len(args) == 0:
            return (None, None, None)

        type_: Type[ASTNode] = ASTNode
        parent_field: str | None = None
        parent_index: int | None = None

        for arg in args:
            if isinstance(arg, type):
                type_ = arg
            elif isinstance(arg, int):
                parent_index = arg if arg > -1 else None
            else:
                parent_field = arg

        return parent_field, parent_index, type_

    def self(
        self, args: list[Type[ASTNode] | str | int]
    ) -> tuple[str | None, int | None, Type[ASTNode] | None]:
        return self.element(args)

    def class_spec(self, args: list[str]) -> Type[ASTNode]:
        type_, msg = check_and_get_ast_node_type(args[0])

        if type_ is None:
            raise ASTXpathDefinitionError(msg)

        return type_

    def index_spec(self, args: list[str]) -> int:
        if len(args) == 0:
            return -1
        return int(args[0])

    def field_spec(self, args: list[str]) -> str:
        return args[0]


xpath_parser = Lark(
    grammar=xpath_grammar, start="xpath", parser="lalr", transformer=XPathTransformer()
)


# Relax version of NodeTraversalInfo allowing root nodes
class _NodeTraversalInfo(NamedTuple):
    node: ASTNode
    parent: ASTNode | None
    field: Field | None
    findex: int | None


def _match_node_element(
    n_info: _NodeTraversalInfo | NodeTraversalInfo, element: ASTXpathElement
) -> bool:
    if (
        isinstance(n_info.node, element.ast_class)
        and (
            element.parent_field is None
            or (n_info.field is not None and element.parent_field == n_info.field.name)
        )
        and (element.parent_index is None or element.parent_index == n_info.findex)
    ):
        return True

    return False


def _match_node_xpath(tree: Tree, node: ASTNode, elements: list[ASTXpathElement]) -> bool:
    element = elements[0]

    # Ok, so we are somewhere in the middle (or start) of the search
    # First we need to match the current node to the current expected element
    c_parent, c_parent_field, c_index = tree.parent_info(node)

    if not _match_node_element(
        _NodeTraversalInfo(node, c_parent, c_parent_field, c_index), element
    ):
        return False

    # The current node matches the current element, so we need to go up the AST
    # Elements are already in reversed order, so we pass the tail of the elements list
    tail = elements[1:]

    if len(tail) == 0:
        # If we werr checking the last element, then there are two options:
        # - it was anywhere => we have a match
        # - it was not anywhere => we need to check if the node has no parent
        #   and if it doesn't then we have a match
        return element.anywhere or c_parent is None

    # Ok, so we have remaining elements to match
    # If no parent then no match
    if c_parent is None:
        return False

    # Otherwise we need to match the remaining elements to the parent
    if element.anywhere:
        # Anywhere means any ancestor can match
        for ancestor in tree.ancestors(node):
            if _match_node_xpath(tree, ancestor, tail):
                return True
    else:
        # Otherwise we need to match only the direct parent
        return _match_node_xpath(tree, c_parent, tail)

    # No match
    return False


_AST_XPATH_CACHE: dict[str, ASTXpath] = {}


# A helper class used in the xpath find method
@dataclass(frozen=True)
class _DUMMY_XPATH_ROOT(ASTNode):
    child: ASTNode


class ASTXpath:
    """A parsed XPath for AST nodes."""

    def __new__(cls, xpath: str) -> ASTXpath:
        if xpath not in _AST_XPATH_CACHE:
            _AST_XPATH_CACHE[xpath] = super().__new__(cls)
        return _AST_XPATH_CACHE[xpath]

    def __init__(self, xpath: str) -> None:
        if not xpath.startswith("/"):
            # Relative path is the same as absolute path starting with "anywehere"
            xpath = "//" + xpath

        try:
            # Reversed list used for matching from the node UP to the root
            self._elements_reversed = cast(
                list[ASTXpathElement],
                xpath_parser.parse(xpath),
            )

            # Normal list used for searching from the root DOWN
            self._elements = list(reversed(self._elements_reversed))
        except UnexpectedInput as e:
            raise ASTXpathDefinitionError(
                f"Incorrect xpath definition. Context:\n{e.get_context(xpath)}"
            ) from None
        except ASTXpathDefinitionError:
            raise
        except Exception as e:
            raise ASTXpathDefinitionError("Incorrect xpath definition") from e

    def match(self, tree_or_root: ASTNode | Tree, node: ASTNode) -> bool:
        """Match the `node` to the xpath in the `tree_or_root`."""
        if not isinstance(tree_or_root, Tree):
            tree_or_root = tree_or_root.to_tree()

        if not tree_or_root.is_in_tree(node):
            raise ValueError("The node is not in the tree")

        return _match_node_xpath(tree_or_root, node, self._elements_reversed)

    def findall(self, root: ASTNode) -> Generator[ASTNode, None, None]:
        """Find all nodes in the `root` that match the xpath.

        Adapted from antlr4-python3-runtime/src/Python3/antlr4/xpath/Xpath.py
        Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
        Use of this file is governed by the BSD 3-clause license that
        can be found in the LICENSE.txt file in the project root.
        """
        # Using dict, because set is not ordered
        work: dict[_NodeTraversalInfo | NodeTraversalInfo, None] = {
            _NodeTraversalInfo(_DUMMY_XPATH_ROOT(root), None, None, None): None
        }

        for el in self._elements:
            new_work: dict[_NodeTraversalInfo | NodeTraversalInfo, None] = {}

            for n_info in work:
                if el.anywhere:
                    for c_info in n_info.node.dfs():
                        if _match_node_element(c_info, el):
                            # Insert into our "ordered set" only if not already in there
                            # this is to prefer first insertion order
                            if c_info not in new_work:
                                new_work[c_info] = None
                else:
                    for c, f, i in n_info.node.get_child_nodes_with_field():
                        c_info = NodeTraversalInfo(c, n_info.node, f, i)
                        if _match_node_element(c_info, el):
                            if c_info not in new_work:
                                new_work[c_info] = None
            work = new_work

        yield from [n_info.node for n_info in new_work.keys()]
