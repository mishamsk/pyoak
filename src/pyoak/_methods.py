"""Alternative implementations of dunder methods for ASTNode."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import ASTNode


def repr_fn(self: ASTNode) -> str:
    """Shallow representation of the node.

    Avoids printing the whole tree.

    """

    # Add standard id, content_id and ref if exists
    props: dict[str, str | None] = {
        "id": self.id,
        "content_id": self.content_id,
        "original_id": self.original_id,
        "id_collision_with": self.id_collision_with,
    }

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
            f"origin={self.origin!s})"
        )

    # Leaf node
    return (
        f"Leaf<{self.__class__.__qualname__}>"
        f"(Props:{', '.join(f'{k}={v}' for k, v in props.items())};"
        f"origin={self.origin!s})"  # string representation of origin is enough
    )
