from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import ASTNode


# Alternative implementations of dunder methods
def hash_fn(node: ASTNode) -> int:
    # We use both id & content_id to increase enthrophy,
    # since the default id/content_id digest is only 8 bytes
    return hash((node.id, node.content_id))


def eq_fn(self: ASTNode, other: ASTNode) -> bool:
    # Other typed as ASTNode to make mypy happy

    # We use both id & content_id to increase enthrophy,
    # since the default id/content_id digest is only 8 bytes
    if other.__class__ is self.__class__:
        return self.id == other.id and self.content_id == other.content_id

    return False


def repr_fn(self: ASTNode) -> str:
    """Shallow representation of the node.

    Avoids printing the whole tree.
    """

    # Add standard id, content_id and ref if exists
    props: dict[str, str] = {"id": self.id, "content_id": self.content_id}

    if self.ref is not None:
        props["ref"] = self.ref

    # Add other properties
    props.update({f.name: repr(val) for val, f in self.get_properties()})

    # Now iterate over children, but only show fields and child type & count
    children: dict[str, str] = {}

    for val, f in self.iter_child_fields():
        if isinstance(val, tuple):
            if len(val) == 0:
                children[f.name] = "()"
            else:
                children[f.name] = f"<{val[0].__class__.__name__}>...({len(val)} total)"
        elif val is None:
            children[f.name] = "None"
        else:
            children[f.name] = f"<{val.__class__.__name__}>"

    # Now combine everything with this node's class and origin at the end
    if children:
        return (
            f"Sub-tree<{self.__class__.__qualname__}>"
            f"(Props:{', '.join(f'{k}={v}' for k, v in props.items())};"
            f"Children:{', '.join(f'{k}={v}' for k, v in children.items())};"
            f"origin={self.origin})"
        )

    # Leaf node
    return (
        f"Leaf<{self.__class__.__qualname__}>"
        f"(Props:{', '.join(f'{k}={v}' for k, v in props.items())};"
        f"origin={self.origin})"
    )
