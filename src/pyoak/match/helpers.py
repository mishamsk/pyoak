from __future__ import annotations

from typing import Mapping, Sequence, Type

from lark import Tree

from pyoak.node import ASTNode

from ..serialize import TYPES


def check_ast_node_type(class_name: str) -> tuple[bool, str]:
    if class_name not in TYPES:
        return False, f"Unknown AST type: <{class_name}>"
    elif not issubclass(TYPES[class_name], ASTNode):
        return False, f"<{class_name}> is not an AST type"
    return True, ""


def check_and_get_ast_node_type(class_name: str) -> tuple[Type[ASTNode] | None, str]:
    if class_name not in TYPES:
        return None, f"Unknown AST type: <{class_name}>"

    type_ = TYPES[class_name]

    if not issubclass(type_, ASTNode):
        return None, f"<{class_name}> is not an AST type"
    else:
        return type_, ""


def get_unique_rule_name(base: str, existing_rules: Sequence[str] | Mapping[str, str]) -> str:
    i = 1
    out = base
    while out in existing_rules:
        out = f"{base}_{i}"
        i += 1
    return out


def maybe_capture_rule(
    maybe_capture_child: str | Tree[str] | None,
    inner_rule: str,
    capture_rule_prefix: str,
    capture_rules: dict[str, str],
) -> str:
    if isinstance((capture := maybe_capture_child), Tree) and capture.data == "capture":
        assert len(capture.children) == 1 and isinstance(capture.children[0], str)
        rule_suffix = get_unique_rule_name(
            base=str(capture.children[0]), existing_rules=capture_rules
        )

        capture_rules[rule_suffix] = inner_rule
        return f"{capture_rule_prefix}{rule_suffix}"
    else:
        return inner_rule
