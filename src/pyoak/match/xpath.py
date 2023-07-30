from __future__ import annotations

import logging
from typing import NamedTuple, Type, cast

from lark import Lark, UnexpectedInput
from lark.visitors import Transformer

from pyoak.match.error import ASTXpathDefinitionError
from pyoak.match.helpers import check_and_get_ast_node_type
from pyoak.node import ASTNode

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


class ASTXpathAnywhereElement:
    pass


class ASTXpathElement(NamedTuple):
    ast_class: Type[ASTNode]
    parent_field: str | None
    parent_index: int | None
    anywhere: bool


class XPathTransformer(Transformer[str, list[ASTXpathElement | ASTXpathAnywhereElement]]):
    def xpath(
        self, args: list[tuple[str | None, int | None, Type[ASTNode] | None]]
    ) -> list[ASTXpathElement | ASTXpathAnywhereElement]:
        ret: list[ASTXpathElement | ASTXpathAnywhereElement] = []

        # Xpath matching goes from the end of the xpath to the beginning
        elements = reversed(args)
        for el in elements:
            anywhere = False
            parent_field, parent_index, ast_class = el

            while ast_class is None:
                anywhere = True
                next_el = next(elements, None)
                if next_el is None:
                    ret.append(ASTXpathAnywhereElement())
                    return ret

                parent_field, parent_index, ast_class = next_el

            ret.append(
                ASTXpathElement(
                    ast_class=ast_class,
                    parent_field=parent_field,
                    parent_index=parent_index,
                    anywhere=anywhere,
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


def _match_node_xpath(
    node: ASTNode | None, elements: list[ASTXpathElement | ASTXpathAnywhereElement]
) -> bool:
    # Node is None when the function was called from the root of the AST
    if node is None:
        # Only match if the remaining xpath is empty or an anywhere element
        if len(elements) == 0 or isinstance(elements[0], ASTXpathAnywhereElement):
            return True
        else:
            return False

    # Node is not None, but the xpath is empty = no match
    if len(elements) == 0:
        return False

    element = elements[0]

    # ASTXpathAnywhereElement only appears at the beggining of the xpath (`here --> //xpath_tail`)
    # so we may return positive match without going further up the AST
    if isinstance(element, ASTXpathAnywhereElement):
        return True

    if element.anywhere:
        # Anywhere case. Means we need to try matching not only current node
        # but also all of it ancestors
        # and if anyone matches then the whole xpath matches
        for ancestor in node.ancestors():
            if _match_node_xpath(ancestor, elements):
                return True

    # Ok, so we are somewhere in the middle (or start) of the search
    # If it was anywhere element, none of the ancestors matched
    # First we need to match the current node to the current expected element
    c_parent_field = node.parent_field.name if node.parent_field else None
    c_index = node.parent_index

    if (
        isinstance(node, element.ast_class)
        and (element.parent_field is None or element.parent_field == c_parent_field)
        and (element.parent_index is None or element.parent_index == c_index)
    ):
        # The current node matches the current element, so we need to go up the AST
        # Elements are already in reversed order, so we pass the tail of the elements list
        return _match_node_xpath(node.parent, elements[1:])
    else:
        return False


class ASTXpath:
    def __init__(self, xpath: str) -> None:
        if not xpath.startswith("/"):
            # Relative path is the same as absolute path starting with "anywehere"
            xpath = "//" + xpath

        try:
            self._elemetns = cast(
                list[ASTXpathElement | ASTXpathAnywhereElement],
                xpath_parser.parse(xpath),
            )
        except UnexpectedInput as e:
            raise ASTXpathDefinitionError(
                f"Incorrect xpath definition. Context:\n{e.get_context(xpath)}"
            ) from None
        except ASTXpathDefinitionError:
            raise
        except Exception as e:
            raise ASTXpathDefinitionError("Incorrect xpath definition") from e

    def match(self, node: ASTNode) -> bool:
        return _match_node_xpath(node, self._elemetns)
