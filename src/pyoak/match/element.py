from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..node import ASTNode
    from .pattern import NodeMatcher


class ASTXpathElement(NamedTuple):
    ast_class_or_pattern: type[ASTNode] | NodeMatcher
    parent_field: str | None
    parent_index: int | None
    anywhere: bool
