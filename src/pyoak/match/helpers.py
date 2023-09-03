from __future__ import annotations

from pyoak.node import ASTNode

from ..serialize import TYPES


def check_and_get_ast_node_type(class_name: str) -> tuple[type[ASTNode] | None, str]:
    if class_name not in TYPES:
        return None, f"Unknown AST type: <{class_name}>"

    type_ = TYPES[class_name]

    if not issubclass(type_, ASTNode):
        return None, f"<{class_name}> is not an AST type"
    else:
        return type_, ""
