from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import ASTNode


# Alternative implementations of dunder methods
def hash_fn(node: ASTNode) -> int:
    # We use both id & content_id to increase enthrophy,
    # since the default id/content_id digest is only 8 bytes
    return hash((node.id, node.content_id, node.ref))


def eq_fn(self: ASTNode, other: ASTNode) -> bool:
    # Other typed as ASTNode to make mypy happy

    # We use both id & content_id to increase enthrophy,
    # since the default id/content_id digest is only 8 bytes
    if other.__class__ is self.__class__:
        return self.id == other.id and self.content_id == other.content_id and self.ref == other.ref

    return False
