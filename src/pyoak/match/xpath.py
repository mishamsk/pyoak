from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Mapping

from ..node import ASTNode
from ..origin import NO_ORIGIN
from .error import ASTXpathOrPatternDefinitionError
from .parser import Parser

if TYPE_CHECKING:
    from .element import ASTXpathElement

logger = logging.getLogger(__name__)


def _match_node_element(node: ASTNode, element: ASTXpathElement) -> bool:
    if isinstance(element.ast_class_or_pattern, type):
        # If the element is a type, then we need to check if the node is an instance of that type
        if not isinstance(node, element.ast_class_or_pattern):
            return False
    # Otherwise this must be a NodeMatcher
    # We need to check if the node matches the matcher
    else:
        match, _ = element.ast_class_or_pattern.match(node)

        if not match:
            return False

    if (
        element.parent_field is None
        or (node.parent_field is not None and element.parent_field == node.parent_field.name)
    ) and (element.parent_index is None or element.parent_index == node.parent_index):
        return True

    return False


def _match_node_xpath(node: ASTNode, elements: list[ASTXpathElement]) -> bool:
    element = elements[0]

    # Ok, so we are somewhere in the middle (or start) of the search
    # First we need to match the current node to the current expected element
    if not _match_node_element(node, element):
        return False

    # The current node matches the current element, so we need to go up the AST
    # Elements are already in reversed order, so we pass the tail of the elements list
    tail = elements[1:]

    if len(tail) == 0:
        # If we werr checking the last element, then there are two options:
        # - it was anywhere => we have a match
        # - it was not anywhere => we need to check if the node has no parent
        #   and if it doesn't then we have a match
        return element.anywhere or node.parent is None

    # Ok, so we have remaining elements to match
    # If no parent then no match
    if node.parent is None:
        return False

    # Otherwise we need to match the remaining elements to the parent
    if element.anywhere:
        # Anywhere means any ancestor can match
        for ancestor in node.ancestors():
            if _match_node_xpath(ancestor, tail):
                return True
    else:
        # Otherwise we need to match only the direct parent
        return _match_node_xpath(node.parent, tail)

    # No match
    return False


_AST_XPATH_CACHE: dict[str, ASTXpath] = {}


# A helper class used in the xpath find method
@dataclass
class _DUMMY_XPATH_ROOT(ASTNode):
    child: ASTNode


class ASTXpath:
    """A parsed XPath for AST nodes."""

    def __new__(cls, xpath: str) -> ASTXpath:
        if xpath not in _AST_XPATH_CACHE:
            _AST_XPATH_CACHE[xpath] = super().__new__(cls)
        return _AST_XPATH_CACHE[xpath]

    def __init__(self, xpath: str, types: Mapping[str, type[Any]] | None = None) -> None:
        """Initialize the xpath.

        Args:
            xpath: The xpath to parse.
            types: An optional mapping of AST class names to their types. If not provided,
                the default mapping from `pyoak.serialize` is used.

        """
        if types is None:
            # Only import if needed
            from ..serialize import TYPES

            types = TYPES

        if not xpath.startswith("/"):
            # Relative path is the same as absolute path starting with "anywehere"
            xpath = "//" + xpath

        try:
            # Reversed list used for matching from the node UP to the root
            self._elements_reversed = list(Parser(types).parse_xpath(xpath))

            # Normal list used for searching from the root DOWN
            self._elements = list(reversed(self._elements_reversed))
        except ASTXpathOrPatternDefinitionError:
            raise
        except Exception as e:
            logger.debug("Internal error parsing xpath", exc_info=True)
            raise ASTXpathOrPatternDefinitionError(
                "Failed to parse Xpath due to internal error. Please report it!"
            ) from e

    def match(self, node: ASTNode) -> bool:
        """Match the `node` to the xpath."""
        return _match_node_xpath(node, self._elements_reversed)

    def findall(self, root: ASTNode) -> Generator[ASTNode, None, None]:
        """Find all nodes in the `root` that match the xpath.

        Adapted from antlr4-python3-runtime/src/Python3/antlr4/xpath/Xpath.py
        Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
        Use of this file is governed by the BSD 3-clause license that
        can be found in the LICENSE.txt file in the project root.

        """
        # Using dict, because set is not ordered
        # Using node ids as keys, because ASTNode is not hashable
        work: dict[str, ASTNode] = {"_DUMMY_XPATH_ROOT": _DUMMY_XPATH_ROOT(root, origin=NO_ORIGIN)}

        for el in self._elements:
            new_work: dict[str, ASTNode] = {}

            for node in work.values():
                if el.anywhere:
                    for n in node.dfs(skip_self=True):
                        if _match_node_element(n, el):
                            # Insert into our "ordered set" only if not already in there
                            # this is to prefer first insertion order
                            if n.id not in new_work:
                                new_work[n.id] = n
                else:
                    for n in node.get_child_nodes():
                        if _match_node_element(n, el):
                            if n.id not in new_work:
                                new_work[n.id] = n
            work = new_work

        yield from new_work.values()
