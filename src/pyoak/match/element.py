from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..node import ASTNode
    from .pattern import BaseMatcher


class ASTXpathElement(NamedTuple):
    ast_class_or_pattern: "type[ASTNode] | BaseMatcher"
    parent_field: str | None
    parent_index: int | None
    anywhere: bool
