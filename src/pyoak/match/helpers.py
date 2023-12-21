from __future__ import annotations

from typing import Any, Mapping, Type

from ..node import ASTNode


def check_and_get_ast_node_type(
    class_name: str, types: Mapping[str, type[Any]]
) -> tuple[Type[ASTNode] | None, str]:
    if class_name not in types:
        return None, f"Unknown AST type: <{class_name}>"

    type_ = types[class_name]

    if not issubclass(type_, ASTNode):
        return None, f"<{class_name}> is not an AST type"
    else:
        return type_, ""
